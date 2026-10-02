#[allow(dead_code)]
#[path = "../hash.rs"]
mod hash;
#[allow(dead_code)]
#[path = "../input.rs"]
mod input;
#[allow(dead_code)]
#[path = "../poseidon.rs"]
mod poseidon;

use clap::{App, Arg};
use hash::{chain_hash, scalar_to_decimal_string, step_digest};
use input::{resolution_spec, DctqInput};
use rayon::prelude::*;
use serde::Serialize;
use std::{fs::File, io::Write, path::PathBuf};

const INPUT_DIGEST_FORMAT: &str = "speg-poseidon-hh-input-digest-v1";

#[derive(Serialize)]
struct InputDigestArtifact<'a> {
    format: &'static str,
    resolution: &'a str,
    digest: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let matches = App::new("SPEG_Poseidon_hh_input_digest")
        .version("v1.0.0")
        .about("Generate the public input-chain digest that a verifier may authorize")
        .arg(
            Arg::with_name("input")
                .required(true)
                .short("i")
                .long("input")
                .value_name("FILE")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .required(true)
                .short("o")
                .long("output")
                .value_name("FILE")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("resolution")
                .required(true)
                .short("r")
                .long("resolution")
                .value_name("RESOLUTION")
                .takes_value(true)
                .possible_values(&["SD", "HD", "FHD", "QHD", "4K"]),
        )
        .get_matches();

    let resolution = matches.value_of("resolution").unwrap();
    let spec = resolution_spec(resolution).expect("clap validates the resolution");
    let input_path = PathBuf::from(matches.value_of("input").unwrap());
    let output_path = PathBuf::from(matches.value_of("output").unwrap());
    let input = DctqInput::load(&input_path, spec).map_err(|error| error.to_string())?;
    let steps = input.into_steps();
    let step_digests = steps.par_iter().map(step_digest).collect::<Vec<_>>();
    let digest = chain_hash(&step_digests);
    let artifact = InputDigestArtifact {
        format: INPUT_DIGEST_FORMAT,
        resolution,
        digest: scalar_to_decimal_string(&digest),
    };
    let json = serde_json::to_vec_pretty(&artifact).map_err(|error| error.to_string())?;
    let mut output = File::create(&output_path).map_err(|error| error.to_string())?;
    output.write_all(&json).map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    println!("input digest written to {}", output_path.display());
    Ok(())
}
