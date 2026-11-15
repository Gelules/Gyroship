//! The build stamp of the shader cache (src/shader/cache.rs): a hash of the
//! compiler's version and of every file its entries depend on (the translator,
//! naga, this manifest). Entries made under another stamp are never read.

use std::path::{Path, PathBuf};

const SOURCES: [&str; 4] = ["Cargo.toml", "src/shader", "src/sass", "../../vendor/naga"];

fn files(path: &Path, out: &mut Vec<PathBuf>) {
    match std::fs::read_dir(path) {
        Ok(dir) => dir.flatten().for_each(|e| files(&e.path(), out)),
        Err(_) => out.push(path.to_path_buf()),
    }
}

fn main() {
    let mut list = Vec::new();
    for s in SOURCES {
        println!("cargo::rerun-if-changed={s}");
        files(Path::new(s), &mut list);
    }
    list.sort();
    let rustc = std::process::Command::new(std::env::var_os("RUSTC").expect("RUSTC")).arg("-V").output().expect("rustc -V").stdout;
    let mut text = String::from_utf8_lossy(&rustc).into_owned();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut add = |bytes: &[u8]| bytes.iter().for_each(|&b| hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    add(text.as_bytes());
    for f in &list {
        let name = f.to_string_lossy().replace('\\', "/");
        add(name.as_bytes());
        add(&[0]);
        add(&std::fs::read(f).unwrap_or_else(|e| panic!("{name}: {e}")));
        text += &name;
        text.push('\n');
    }
    println!("cargo::rustc-env=GYROSHIP_SHADER_BUILD={hash:016x}");
    std::fs::write(Path::new(&std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("shader-sources.txt"), text).expect("shader-sources.txt");
}
