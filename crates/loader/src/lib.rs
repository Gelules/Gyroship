//! NX executables (NSO0): segments, LZ4 decompression and hash checks, MOD0,
//! dynamic table, symbols and relocations, and export to an ELF file that
//! disassemblers can load.
//!
//! The module image is laid out from address 0 (module-relative); a loader
//! picks the real base address. Executables are untrusted input: every
//! offset and size is bounds-checked.

use sha2::{Digest, Sha256};
use std::fmt;

pub mod link;

const NSO_MAGIC: &[u8; 4] = b"NSO0";
const MOD0_MAGIC: &[u8; 4] = b"MOD0";
const HEADER_SIZE: usize = 0x100;
// ponytail: a module image is held in memory whole; real modules are < 1 GiB.
const MAX_SEGMENT: u32 = 1 << 30;

pub mod dt {
    pub const NULL: i64 = 0;
    pub const NEEDED: i64 = 1;
    pub const PLTRELSZ: i64 = 2;
    pub const HASH: i64 = 4;
    pub const STRTAB: i64 = 5;
    pub const SYMTAB: i64 = 6;
    pub const RELA: i64 = 7;
    pub const RELASZ: i64 = 8;
    pub const STRSZ: i64 = 10;
    pub const SYMENT: i64 = 11;
    pub const SONAME: i64 = 14;
    pub const JMPREL: i64 = 23;
    pub const RELRSZ: i64 = 35;
    pub const RELR: i64 = 36;
    pub const GNU_HASH: i64 = 0x6fff_fef5;
}

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn bad<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

fn bytes<const N: usize>(buf: &[u8], off: u64) -> Result<[u8; N]> {
    usize::try_from(off)
        .ok()
        .and_then(|o| buf.get(o..o.checked_add(N)?))
        .map(|b| b.try_into().expect("slice has length N"))
        .ok_or_else(|| Error(format!("read of {N} bytes at {off:#x} is out of bounds")))
}
fn u16_at(buf: &[u8], off: u64) -> Result<u16> {
    bytes(buf, off).map(u16::from_le_bytes)
}
fn u32_at(buf: &[u8], off: u64) -> Result<u32> {
    bytes(buf, off).map(u32::from_le_bytes)
}
fn u64_at(buf: &[u8], off: u64) -> Result<u64> {
    bytes(buf, off).map(u64::from_le_bytes)
}
fn i32_at(buf: &[u8], off: u64) -> Result<i32> {
    bytes(buf, off).map(i32::from_le_bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// Module-relative address.
    pub addr: u64,
    pub size: u64,
}

impl Segment {
    pub fn end(&self) -> u64 {
        self.addr + self.size
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Mod0 {
    /// Module-relative address of the MOD0 header itself.
    pub addr: u64,
    pub dynamic: u64,
    pub bss_start: u64,
    pub bss_end: u64,
    pub eh_frame_hdr_start: u64,
    pub eh_frame_hdr_end: u64,
    pub module_object: u64,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub info: u8,
    pub shndx: u16,
}

impl Symbol {
    pub fn is_defined(&self) -> bool {
        self.shndx != 0
    }
    pub fn is_function(&self) -> bool {
        self.info & 0xF == 2
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rela {
    pub offset: u64,
    pub kind: u32,
    pub symbol: u32,
    pub addend: i64,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub version: u32,
    pub flags: u32,
    /// ModuleId (build ID): 32 bytes, the first 0x14 are significant in SDK builds.
    pub build_id: [u8; 32],
    pub text: Segment,
    pub ro: Segment,
    pub data: Segment,
    pub bss_size: u64,
    /// text, then ro, then data at their addresses; bss is not stored.
    pub image: Vec<u8>,
    pub mod0: Mod0,
    pub dynamic: Vec<(i64, u64)>,
    pub symbols: Vec<Symbol>,
    pub relocations: Vec<Rela>,
    /// Number of words in the DT_RELR table (packed relative relocations).
    pub relr_words: u64,
    pub needed: Vec<String>,
    pub soname: Option<String>,
}

impl Module {
    /// Parses an NSO0 file: decompresses and checks every segment, then
    /// reads MOD0, the dynamic table, symbols and relocations.
    pub fn parse_nso(file: &[u8]) -> Result<Module> {
        if file.len() < HEADER_SIZE || &file[0..4] != NSO_MAGIC {
            return bad("not an NSO0 file");
        }
        let flags = u32_at(file, 0x0C)?;
        let mut segs = [Segment { addr: 0, size: 0 }; 3];
        let mut image = Vec::new();
        for (i, seg) in segs.iter_mut().enumerate() {
            let h = 0x10 + 0x10 * i as u64;
            let file_off = u64::from(u32_at(file, h)?);
            let addr = u64::from(u32_at(file, h + 4)?);
            let size = u32_at(file, h + 8)?;
            let stored = u64::from(u32_at(file, 0x60 + 4 * i as u64)?);
            if size > MAX_SEGMENT {
                return bad(format!("segment {i} is too large ({size} bytes)"));
            }
            let Some(raw) = file.get(file_off as usize..(file_off + stored) as usize) else {
                return bad(format!("segment {i} lies outside the file"));
            };
            let content = if flags & (1 << i) != 0 {
                let mut out = vec![0u8; size as usize];
                let n = lz4_flex::block::decompress_into(raw, &mut out).map_err(|e| Error(format!("segment {i}: LZ4: {e}")))?;
                if n != size as usize {
                    return bad(format!("segment {i}: decompressed to {n} bytes, expected {size}"));
                }
                out
            } else {
                if stored != u64::from(size) {
                    return bad(format!("segment {i}: stored size differs from its size"));
                }
                raw.to_vec()
            };
            if flags & (1 << (i + 3)) != 0 {
                let want: [u8; 32] = bytes(file, 0xA0 + 0x20 * i as u64)?;
                if <[u8; 32]>::from(Sha256::digest(&content)) != want {
                    return bad(format!("segment {i}: SHA-256 mismatch"));
                }
            }
            *seg = Segment { addr, size: u64::from(size) };
            let end = (addr + seg.size) as usize;
            if image.len() < end {
                image.resize(end, 0);
            }
            image[addr as usize..end].copy_from_slice(&content);
        }
        let [text, ro, data] = segs;
        if !(text.addr == 0 && text.end() <= ro.addr && ro.end() <= data.addr) {
            return bad("segments are not ordered text < ro < data from address 0");
        }
        let mod0 = parse_mod0(&image)?;
        let mut m = Module {
            version: u32_at(file, 0x04)?,
            flags,
            build_id: bytes(file, 0x40)?,
            text,
            ro,
            data,
            bss_size: u64::from(u32_at(file, 0x3C)?),
            image,
            mod0,
            dynamic: Vec::new(),
            symbols: Vec::new(),
            relocations: Vec::new(),
            relr_words: 0,
            needed: Vec::new(),
            soname: None,
        };
        m.parse_dynamic()?;
        Ok(m)
    }

    /// Value of the first dynamic entry with this tag.
    pub fn dyn_value(&self, tag: i64) -> Option<u64> {
        self.dynamic.iter().find(|(t, _)| *t == tag).map(|(_, v)| *v)
    }

    /// End of the module in memory, bss included.
    pub fn mem_end(&self) -> u64 {
        (self.data.end() + self.bss_size).max(self.mod0.bss_end)
    }

    fn parse_dynamic(&mut self) -> Result<()> {
        let mut at = self.mod0.dynamic;
        loop {
            let tag = u64_at(&self.image, at)? as i64;
            let val = u64_at(&self.image, at + 8)?;
            at += 16;
            if tag == dt::NULL {
                break;
            }
            self.dynamic.push((tag, val));
        }
        let strtab = self.dyn_value(dt::STRTAB).unwrap_or(0);
        let strsz = self.dyn_value(dt::STRSZ).unwrap_or(0);
        let Some(strings) = self.image.get(strtab as usize..strtab.saturating_add(strsz) as usize) else {
            return bad("DT_STRTAB lies outside the image");
        };
        let string = |off: u64| -> String {
            let s = strings.get(off as usize..).unwrap_or(&[]);
            let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
            String::from_utf8_lossy(&s[..end]).into_owned()
        };
        for &(tag, val) in &self.dynamic {
            match tag {
                dt::NEEDED => self.needed.push(string(val)),
                dt::SONAME => self.soname = Some(string(val)),
                _ => {}
            }
        }
        if let Some(symtab) = self.dyn_value(dt::SYMTAB) {
            let syment = self.dyn_value(dt::SYMENT).unwrap_or(24);
            if syment != 24 {
                return bad(format!("unexpected DT_SYMENT {syment}"));
            }
            for i in 0..self.symbol_count()? {
                let s = symtab + 24 * i;
                self.symbols.push(Symbol {
                    name: string(u64::from(u32_at(&self.image, s)?)),
                    info: self.image[s as usize + 4],
                    shndx: u16_at(&self.image, s + 6)?,
                    value: u64_at(&self.image, s + 8)?,
                    size: u64_at(&self.image, s + 16)?,
                });
            }
        }
        for (table, size) in [(dt::RELA, dt::RELASZ), (dt::JMPREL, dt::PLTRELSZ)] {
            let (Some(start), Some(len)) = (self.dyn_value(table), self.dyn_value(size)) else { continue };
            for r in (start..start + len).step_by(24) {
                let info = u64_at(&self.image, r + 8)?;
                self.relocations.push(Rela {
                    offset: u64_at(&self.image, r)?,
                    kind: info as u32,
                    symbol: (info >> 32) as u32,
                    addend: u64_at(&self.image, r + 16)? as i64,
                });
            }
        }
        self.relr_words = self.dyn_value(dt::RELRSZ).unwrap_or(0) / 8;
        Ok(())
    }

    /// Symbol count from DT_HASH (nchain) or DT_GNU_HASH (highest chain end).
    fn symbol_count(&self) -> Result<u64> {
        if let Some(hash) = self.dyn_value(dt::HASH) {
            return Ok(u64::from(u32_at(&self.image, hash + 4)?));
        }
        let Some(gnu) = self.dyn_value(dt::GNU_HASH) else {
            return bad("DT_SYMTAB without DT_HASH or DT_GNU_HASH: symbol count unknown");
        };
        let nbuckets = u64::from(u32_at(&self.image, gnu)?);
        let symoffset = u64::from(u32_at(&self.image, gnu + 4)?);
        let bloom_size = u64::from(u32_at(&self.image, gnu + 8)?);
        let buckets = gnu + 16 + 8 * bloom_size;
        let chains = buckets + 4 * nbuckets;
        let mut last = 0u64;
        for b in 0..nbuckets {
            last = last.max(u64::from(u32_at(&self.image, buckets + 4 * b)?));
        }
        if last < symoffset {
            return Ok(symoffset);
        }
        while u32_at(&self.image, chains + 4 * (last - symoffset))? & 1 == 0 {
            last += 1;
        }
        Ok(last + 1)
    }

    /// Writes the module as an AArch64 ELF64 shared object at address 0:
    /// one PT_LOAD per segment (bss included in the data segment) and a
    /// PT_DYNAMIC, so the dynamic table drives symbols and relocations.
    pub fn to_elf(&self) -> Vec<u8> {
        const IMAGE_OFF: u64 = 0x1000;
        let dyn_size = 16 * (self.dynamic.len() as u64 + 1);
        // (type, flags, addr, file size, memory size)
        let phdrs = [
            (1u32, 5u32, self.text.addr, self.text.size, self.text.size),
            (1, 4, self.ro.addr, self.ro.size, self.ro.size),
            (1, 6, self.data.addr, self.data.size, self.mem_end() - self.data.addr),
            (2, 6, self.mod0.dynamic, dyn_size, dyn_size),
        ];
        let mut elf = Vec::with_capacity(IMAGE_OFF as usize + self.image.len());
        elf.extend_from_slice(b"\x7fELF\x02\x01\x01");
        elf.resize(16, 0);
        elf.extend_from_slice(&3u16.to_le_bytes()); // ET_DYN
        elf.extend_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        elf.extend_from_slice(&1u32.to_le_bytes());
        elf.extend_from_slice(&0u64.to_le_bytes()); // entry: module start
        elf.extend_from_slice(&0x40u64.to_le_bytes()); // phoff
        elf.extend_from_slice(&0u64.to_le_bytes()); // shoff: no sections
        elf.extend_from_slice(&0u32.to_le_bytes());
        for v in [64u16, 56, phdrs.len() as u16, 64, 0, 0] {
            elf.extend_from_slice(&v.to_le_bytes());
        }
        for (ty, flags, addr, filesz, memsz) in phdrs {
            elf.extend_from_slice(&ty.to_le_bytes());
            elf.extend_from_slice(&flags.to_le_bytes());
            for v in [IMAGE_OFF + addr, addr, addr, filesz, memsz, if ty == 1 { 0x1000 } else { 8 }] {
                elf.extend_from_slice(&v.to_le_bytes());
            }
        }
        elf.resize(IMAGE_OFF as usize, 0);
        elf.extend_from_slice(&self.image);
        elf
    }
}

fn parse_mod0(image: &[u8]) -> Result<Mod0> {
    let addr = u64::from(u32_at(image, 4)?);
    if bytes::<4>(image, addr)? != *MOD0_MAGIC {
        return bad("no MOD0 header where the module start points");
    }
    let rel = |off: u64| -> Result<u64> { Ok(addr.wrapping_add(i64::from(i32_at(image, addr + off)?) as u64)) };
    Ok(Mod0 {
        addr,
        dynamic: rel(4)?,
        bss_start: rel(8)?,
        bss_end: rel(12)?,
        eh_frame_hdr_start: rel(16)?,
        eh_frame_hdr_end: rel(20)?,
        module_object: rel(24)?,
    })
}
