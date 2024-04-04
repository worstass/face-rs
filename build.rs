use anyhow::{Context, Result};
use std::{io::Write, path::Path, process::Command};
// https://drive.google.com/uc?id=17TpxpyHuUc1ZTm3RIbfvhnBcZqhyKszV
// arcface-mnet.zip 17TpxpyHuUc1ZTm3RIbfvhnBcZqhyKszV
// arcface_r100_v1.zip 11xFaEHIQLNze3-2RUV1cQfT-q6PKKfYp
// arcface_resnet34.zip 1ECp5XrLgfEAnwyTYFEhJgIsOAw6KaHa7 
// arcface_resnet50.zip 1a9nib4I9OIVORwsqLB0gz0WuLC32E8gf
// arcface-r50-msfdrop75.zip  1gNuvRNHCNgvFtz7SjhW82v2-znlAYaRO
// arcface-r100-msfdrop75.zip 1lAnFcBXoMKqE-SkZKTmi6MsYAmzG0tFw
// arcface_mobilefacenet_casia_masked.zip 1ltcJChTdP1yQWF9e1ESpTNYAVwxLSNLP
// genderage_v1.zip 1J9hqSWqZz6YvMMNrDrmrzEW9anhvdKuC
// 2d106det.zip 18cL35hF2exZ8u4pfLKWjJGxF0ySuYM2o
fn main() -> Result<()> {
    // let output = Command::new("python3")
    //     .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_model.py"))
    //     .arg(&format!("--build-dir={}", env!("CARGO_MANIFEST_DIR")))
    //     .output()
    //     .with_context(|| anyhow::anyhow!("failed to run python3"))?;
    // if !output.status.success() {
    //     std::io::stdout()
    //         .write_all(&output.stderr)
    //         .context("Failed to write error")?;
    //     panic!("Failed to execute build script");
    // }
    // assert!(
    //     Path::new(&format!("{}/deploy_lib.o", env!("CARGO_MANIFEST_DIR"))).exists(),
    //     "Could not prepare demo: {}",
    //     String::from_utf8(output.stderr)
    //         .unwrap()
    //         .trim()
    //         .split("\n")
    //         .last()
    //         .unwrap_or("")
    // );
    // println!(
    //     "cargo:rustc-link-search=native={}",
    //     env!("CARGO_MANIFEST_DIR")
    // );

    Ok(())
}
