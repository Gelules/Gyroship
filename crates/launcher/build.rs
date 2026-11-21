fn main() {
    // tauri-build puts the icon into the executable, yet does not ask to run again when it changes.
    println!("cargo:rerun-if-changed=icons/icon.ico");
    tauri_build::build()
}
