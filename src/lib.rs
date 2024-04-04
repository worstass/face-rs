use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Cursor},
    path::Path,
};

use ::ndarray::{Array, ArrayD, Axis};
use image::{
    io::Reader, DynamicImage, GenericImageView, ImageFormat, ImageResult,
    imageops::FilterType
};

use anyhow::Result;
use anyhow::{Context as _, Ok};
use tvm_rt::graph_rt::GraphRt;
use tvm_rt::*;
use tvm::{DataType, Device, Module, NDArray};
use zip::ZipArchive;

fn readimage(data: &[u8]) -> Result<DynamicImage> {
    let mut reader = Reader::new(Cursor::new(data))
      .with_guessed_format()?;

    // assert_eq!(reader.format(), Some(ImageFormat::Pnm));

    let image = reader.decode()?;
    Ok(image)
}


fn load_tvm_model(path: Path) -> Result<GraphRt> {
    let archive = File::open(path)?;
    let mut archive = ZipArchive::new(archive)?;
    let mut file = archive.by_name("mod.so")?;
    let path = "mod.so";
    let mut output = File::create(path)?;
    std::io::copy(&mut file, &mut output);
        // load the built module
    let lib = Module::load(&Path::new(path))?;

    let mut graph = GraphRt::create_from_parts(&graph, lib, ctx)?;
    let params: Vec<u8> = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_param.params"))?;
    println!("param bytes: {}", params.len());

    graph.load_params(&params)?;
    graph.set_input("data", input)?;
    Ok(graph)
    // graph.run()?;

    // // prepare to get the output
    // let output_shape = &[1, 1000];
    // let output = NDArray::empty(output_shape, ctx, DataType::float(32, 1));
    // graph.get_output_into(0, output.clone())?;

    // // flatten the output as Vec<f32>
    // let output = output.to_vec::<f32>()?;

}

fn run() -> Result<()> {
    let ctx = Device::cpu(0);
    println!("{}", concat!(env!("CARGO_MANIFEST_DIR"), "/cat.png"));

    let img = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/cat.png"))
        .context("Failed to open cat.png")?;

    println!("original image dimensions: {:?}", img.dimensions());
    // for bigger size images, one needs to first resize to 256x256
    // with `img.resize_exact` method and then `image.crop` to 224x224
    let img = img.resize(224, 224, FilterType::Nearest).to_rgb8();
    println!("resized image dimensions: {:?}", img.dimensions());
    let mut pixels = vec![];
    for pixel in img.pixels() {
        let tmp = pixel.data;
        // normalize the RGB channels using mean, std of imagenet1k
        let tmp = [
            (tmp[0] as f32 - 123.0) / 58.395, // R
            (tmp[1] as f32 - 117.0) / 57.12,  // G
            (tmp[2] as f32 - 104.0) / 57.375, // B
        ];
        for e in &tmp {
            pixels.push(*e);
        }
    }

    let arr = Array::from_shape_vec((224, 224, 3), pixels)?;
    let arr: ArrayD<f32> = arr.permuted_axes([2, 0, 1]).into_dyn();
    // make arr shape as [1, 3, 224, 224] acceptable to resnet
    let arr = arr.insert_axis(Axis(0));
    // create input tensor from rust's ndarray
    let input = NDArray::from_rust_ndarray(&arr, ctx, DataType::float(32, 1))?;
    println!(
        "input shape is {:?}, len: {}, size: {}",
        input.shape(),
        input.len(),
        input.size(),
    );

    let graph = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_graph.json"))
        .context("Failed to open graph")?;

    // load the built module
    let lib = Module::load(&Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/deploy_lib.so"
    )))?;

    let mut graph = GraphRt::create_from_parts(&graph, lib, ctx)?;
    let params = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/deploy_param.params"))?;
    println!("param bytes: {}", params.len());

    graph.load_params(&params)?;
    graph.set_input("data", input)?;
    graph.run()?;

    // prepare to get the output
    let output_shape = &[1, 1000];
    let output = NDArray::empty(output_shape, ctx, DataType::float(32, 1));
    graph.get_output_into(0, output.clone())?;

    // flatten the output as Vec<f32>
    let output = output.to_vec::<f32>()?;

    // find the maximum entry in the output and its index
    let (argmax, max_prob) = output
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // create a hash map of (class id, class name)
    let file = File::open("synset.txt").context("failed to open synset")?;
    let synset: Vec<String> = BufReader::new(file)
        .lines()
        .into_iter()
        .map(|x| x.expect("readline failed"))
        .collect();

    let label = &synset[argmax];
    println!(
        "input image belongs to the class `{}` with probability {}",
        label, max_prob
    );

    Ok(())
}