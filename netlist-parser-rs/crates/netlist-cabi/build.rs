//! Generate `netlist_parser.h` from the `#[no_mangle]` C ABI via cbindgen.

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = std::path::Path::new(&crate_dir).join("include/netlist_parser.h");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_root_or_default(&crate_dir))
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&out);
        }
        // Don't fail the build if header generation hiccups (e.g. offline); the
        // committed header stays usable.
        Err(e) => println!("cargo:warning=cbindgen failed: {e}"),
    }
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
