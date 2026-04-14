use spirv_builder::{SpirvBuilder, SpirvMetadata};
use std::error::Error;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let shader_crate = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // bin/
        .join("spirv_testing");

    println!("Building shader: {}", shader_crate.display());

    let mut builder = SpirvBuilder::new(&shader_crate, "spirv-unknown-vulkan1.1");
    builder.target_dir_path = Some(PathBuf::from("spirv-builder"));
    builder.build_script.forward_rustc_warnings = Some(true);
    // Full metadata keeps OpName/OpLine so the disassembly is readable.
    builder.spirv_metadata = SpirvMetadata::Full;

    let result = builder.build()?;

    let spv_path = result.module.unwrap_single();
    println!("SPIRV: {}", spv_path.display());
    println!("Entry points: {:?}", result.entry_points);

    // Disassemble via rspirv (already a transitive dep of spirv-builder).
    disassemble(spv_path)?;

    Ok(())
}

fn disassemble(spv_path: &Path) -> Result<(), Box<dyn Error>> {
    use rspirv::binary::Disassemble;

    let data = std::fs::read(spv_path)?;
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_bytes(&data, &mut loader)?;
    let module = loader.module();

    std::fs::write("./out.spv.txt", module.disassemble())?;

    Ok(())
}
