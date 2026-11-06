//! Function ranges from the unwind tables: `.eh_frame_hdr` lists every FDE
//! sorted by start address, and each FDE gives its function's length.
//!
//! Only the encodings that compilers emit for these tables are accepted;
//! anything else is reported, never guessed.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    /// Module-relative start address.
    pub start: u64,
    pub len: u64,
    /// Module-relative address of the function's exception table (LSDA).
    pub lsda: Option<u64>,
}

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

struct Reader<'a> {
    image: &'a [u8],
    at: u64,
}

impl Reader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let start = usize::try_from(self.at).map_err(|_| self.oob())?;
        let b = self.image.get(start..start + N).ok_or_else(|| self.oob())?;
        self.at += N as u64;
        Ok(b.try_into().expect("slice has length N"))
    }
    fn oob(&self) -> Error {
        Error(format!("unwind table read at {:#x} is out of bounds", self.at))
    }
    fn u8(&mut self) -> Result<u8> {
        self.bytes::<1>().map(|b| b[0])
    }
    fn uleb(&mut self) -> Result<u64> {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                v |= u64::from(b & 0x7F) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
    }
    fn sleb(&mut self) -> Result<i64> {
        let (mut v, mut shift) = (0i64, 0);
        loop {
            let b = self.u8()?;
            if shift < 64 {
                v |= i64::from(b & 0x7F) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                if shift < 64 && b & 0x40 != 0 {
                    v |= -1 << shift;
                }
                return Ok(v);
            }
        }
    }
    /// A DW_EH_PE-encoded pointer. `datarel` is the base for 0x30 encodings.
    fn encoded(&mut self, enc: u8, datarel: u64) -> Result<u64> {
        let here = self.at;
        let raw = match enc & 0x0F {
            0x00 | 0x04 => u64::from_le_bytes(self.bytes()?),
            0x01 => self.uleb()?,
            0x02 => u64::from(u16::from_le_bytes(self.bytes()?)),
            0x03 => u64::from(u32::from_le_bytes(self.bytes()?)),
            0x09 => self.sleb()? as u64,
            0x0A => i64::from(i16::from_le_bytes(self.bytes()?)) as u64,
            0x0B => i64::from(i32::from_le_bytes(self.bytes()?)) as u64,
            0x0C => u64::from_le_bytes(self.bytes()?),
            _ => return Err(Error(format!("pointer encoding {enc:#04x} at {here:#x} is not supported"))),
        };
        if raw == 0 {
            return Ok(0); // a null pointer stays null, as in the unwinder
        }
        Ok(match enc & 0x70 {
            0x00 => raw,
            0x10 => here.wrapping_add(raw),
            0x30 => datarel.wrapping_add(raw),
            _ => return Err(Error(format!("pointer encoding {enc:#04x} at {here:#x} is not supported"))),
        })
    }
}

/// Reads the `.eh_frame_hdr` at `hdr` in `image` (both module-relative) and
/// returns one range per FDE, sorted by start.
pub fn function_ranges(image: &[u8], hdr: u64) -> Result<Vec<Range>> {
    let mut r = Reader { image, at: hdr };
    let [version, ptr_enc, count_enc, table_enc] = r.bytes::<4>()?;
    if version != 1 {
        return Err(Error(format!("eh_frame_hdr version {version}")));
    }
    if ptr_enc == 0xFF || count_enc == 0xFF || table_enc == 0xFF {
        return Err(Error("eh_frame_hdr has no search table".into()));
    }
    r.encoded(ptr_enc, hdr)?;
    let count = r.encoded(count_enc, hdr)?;
    let mut ranges = Vec::with_capacity(count.min(1 << 24) as usize);
    let mut prev = None;
    for _ in 0..count {
        let start = r.encoded(table_enc, hdr)?;
        let fde = r.encoded(table_enc, hdr)?;
        if prev.is_some_and(|p| p > start) {
            return Err(Error(format!("eh_frame_hdr table is not sorted at {start:#x}")));
        }
        prev = Some(start);
        let range = read_fde(image, fde)?;
        if range.start != start {
            return Err(Error(format!("FDE at {fde:#x} starts at {:#x}, the table says {start:#x}", range.start)));
        }
        ranges.push(range);
    }
    Ok(ranges)
}

fn read_fde(image: &[u8], fde: u64) -> Result<Range> {
    let mut r = Reader { image, at: fde };
    let len = u32::from_le_bytes(r.bytes()?);
    if len == 0 || len == u32::MAX {
        return Err(Error(format!("FDE at {fde:#x} has length {len:#x}")));
    }
    let id_at = r.at;
    let cie_delta = u32::from_le_bytes(r.bytes()?);
    if cie_delta == 0 {
        return Err(Error(format!("{fde:#x} is a CIE, not an FDE")));
    }
    let cie = read_cie(image, id_at.wrapping_sub(u64::from(cie_delta)))?;
    let start = r.encoded(cie.fde_enc, 0)?;
    let len = r.encoded(cie.fde_enc & 0x0F, 0)?;
    let mut lsda = None;
    if cie.augmented {
        r.uleb()?;
        if let Some(enc) = cie.lsda_enc {
            lsda = Some(r.encoded(enc, 0)?).filter(|&p| p != 0);
        }
    }
    Ok(Range { start, len, lsda })
}

struct Cie {
    fde_enc: u8,
    lsda_enc: Option<u8>,
    /// Has augmentation data ("z"), so FDEs carry it too.
    augmented: bool,
}

fn read_cie(image: &[u8], cie: u64) -> Result<Cie> {
    let mut r = Reader { image, at: cie };
    let _len = u32::from_le_bytes(r.bytes()?);
    if u32::from_le_bytes(r.bytes()?) != 0 {
        return Err(Error(format!("no CIE at {cie:#x}")));
    }
    let version = r.u8()?;
    let mut aug = Vec::new();
    loop {
        match r.u8()? {
            0 => break,
            c => aug.push(c),
        }
    }
    r.uleb()?; // code alignment
    r.sleb()?; // data alignment
    if version == 1 {
        r.u8()?;
    } else {
        r.uleb()?;
    }
    let mut out = Cie { fde_enc: 0x00, lsda_enc: None, augmented: aug.first() == Some(&b'z') };
    if !out.augmented {
        return Ok(out); // no augmentation data: absolute pointers
    }
    r.uleb()?;
    for c in &aug[1..] {
        match c {
            b'R' => out.fde_enc = r.u8()?,
            b'L' => out.lsda_enc = Some(r.u8()?).filter(|&e| e != 0xFF),
            b'P' => {
                let enc = r.u8()?;
                r.encoded(enc, 0)?;
            }
            b'S' | b'B' => {}
            _ => return Err(Error(format!("CIE at {cie:#x}: unknown augmentation {:?}", String::from_utf8_lossy(&aug)))),
        }
    }
    Ok(out)
}

/// Landing pads of the function at `start` from its LSDA (the call-site
/// table of `.gcc_except_table`), sorted and deduplicated. Module-relative.
pub fn landing_pads(image: &[u8], start: u64, lsda: u64) -> Result<Vec<u64>> {
    let mut r = Reader { image, at: lsda };
    let lp_enc = r.u8()?;
    let lp_start = if lp_enc == 0xFF { start } else { r.encoded(lp_enc, 0)? };
    if r.u8()? != 0xFF {
        r.uleb()?; // type table offset
    }
    let cs_enc = r.u8()?;
    if cs_enc & 0x70 != 0 {
        return Err(Error(format!("LSDA at {lsda:#x}: call-site encoding {cs_enc:#04x} is not supported")));
    }
    let len = r.uleb()?;
    let end = r.at.checked_add(len).ok_or_else(|| r.oob())?;
    let mut pads = Vec::new();
    while r.at < end {
        r.encoded(cs_enc, 0)?; // call-site start
        r.encoded(cs_enc, 0)?; // call-site length
        let pad = r.encoded(cs_enc, 0)?;
        r.uleb()?; // action
        if pad != 0 {
            pads.push(lp_start.wrapping_add(pad));
        }
    }
    if r.at != end {
        return Err(Error(format!("LSDA at {lsda:#x}: call-site table overruns its length")));
    }
    pads.sort_unstable();
    pads.dedup();
    Ok(pads)
}
