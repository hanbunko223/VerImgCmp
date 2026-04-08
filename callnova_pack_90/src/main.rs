pub mod frontend {
    pub use nova_snark::frontend::*;
}

mod artifact;
mod circuit;
mod dctq;
mod hash;
mod input;
mod poseidon;
mod prover;

use crate::{
    artifact::RecursiveProofArtifact,
    hash::scalar_to_decimal_string,
    prover::{NativeRecursiveSNARK, ProverError, prove},
};
use clap::{App, Arg};
use nova_snark::timing::snapshot_recursive_timing;
use rayon::ThreadPoolBuilder;
use serde_json::to_string_pretty;
use std::{
    env,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

fn print_neutron_timing_summary(
    recursive_creation_s: f64,
    frontend_prepare_s: f64,
    verify_s: f64,
    final_outputs: &[String],
) {
    let timing = snapshot_recursive_timing();
    let prove_step_s = timing.neutron_prove_step_total;
    let others_s = (recursive_creation_s - frontend_prepare_s - prove_step_s).max(0.0);
    let extract_other_s = (timing.neutron_extract_instance_witness - timing.commit_w).max(0.0);

    println!();
    println!("NeutronNova recursive proving summary:");
    println!("+-------------------------------+-------------+");
    println!("| metric                        | time (s)    |");
    println!("+-------------------------------+-------------+");
    println!(
        "| total recursive creation      | {:>11.6} |",
        recursive_creation_s
    );
    println!(
        "| frontend_prepare              | {:>11.6} |",
        frontend_prepare_s
    );
    println!("| prove_step                    | {:>11.6} |", prove_step_s);
    println!("| others                        | {:>11.6} |", others_s);
    println!("+-------------------------------+-------------+");
    println!();
    println!("prove_step breakdown (sum over all folded steps):");
    println!("+-------------------------------+-------------+");
    println!("| metric                        | time (s)    |");
    println!("+-------------------------------+-------------+");
    println!(
        "| commit_E                      | {:>11.6} |",
        timing.neutron_commit_e
    );
    println!(
        "| multiply_vec_z1               | {:>11.6} |",
        timing.neutron_multiply_vec_z1
    );
    println!(
        "| multiply_vec_z2               | {:>11.6} |",
        timing.neutron_multiply_vec_z2
    );
    println!(
        "| prove_helper                  | {:>11.6} |",
        timing.neutron_prove_helper
    );
    println!(
        "| poly_finalize                 | {:>11.6} |",
        timing.neutron_poly_finalize
    );
    println!(
        "| fold_instance                 | {:>11.6} |",
        timing.neutron_fold_instance
    );
    println!(
        "| fold_witness                  | {:>11.6} |",
        timing.neutron_fold_witness
    );
    println!(
        "| augmented_synthesize          | {:>11.6} |",
        timing.neutron_augmented_synthesize
    );
    println!("| commit_W                      | {:>11.6} |", timing.commit_w);
    println!("| extract_other                 | {:>11.6} |", extract_other_s);
    println!("+-------------------------------+-------------+");
    println!("recursive verify: {:.6}s", verify_s);
    println!("Final outputs: {}", final_outputs.join(", "));
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn resolve_rayon_thread_count(matches: &clap::ArgMatches) -> Result<usize, ProverError> {
    let cli_value = matches.value_of("rayon_threads");
    let env_value = env::var("RAYON_NUM_THREADS").ok();
    let raw_value = cli_value.or(env_value.as_deref());
    let default_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);

    match raw_value {
        Some(value) => {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| ProverError::Configuration(format!(
                    "invalid Rayon thread count: {value}"
                )))?;
            if parsed == 0 {
                return Err(ProverError::Configuration(
                    "Rayon thread count must be greater than 0".to_string(),
                ));
            }
            Ok(parsed)
        }
        None => Ok(default_threads),
    }
}

fn initialize_rayon(thread_count: usize) -> Result<usize, ProverError> {
    ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .map_err(|error| {
            ProverError::Configuration(format!(
                "failed to initialize Rayon thread pool: {error}"
            ))
        })?;
    Ok(rayon::current_num_threads())
}

fn run() -> Result<(), ProverError> {
    let matches = App::new("VIMz")
        .version("v1.3.0")
        .author("Zero-Savvy")
        .about("Verifiable Image Manipulation from Folded zkSNARKs")
        .arg(
            Arg::with_name("input")
                .required(true)
                .short("i")
                .long("input")
                .value_name("FILE")
                .help("The JSON file containing the original image rows to verify.")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .required(true)
                .short("o")
                .long("output")
                .value_name("FILE")
                .help("This file will contain the final proof artifact.")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("function")
                .required(true)
                .short("f")
                .long("function")
                .value_name("FUNCTION")
                .help("The transformation function.")
                .takes_value(true)
                .possible_values(&["dctq"]),
        )
        .arg(
            Arg::with_name("resolution")
                .required(true)
                .short("r")
                .long("resolution")
                .value_name("RESOLUTION")
                .help("The resolution of the image.")
                .takes_value(true)
                .possible_values(&["HD"]),
        )
        .arg(
            Arg::with_name("rayon_threads")
                .long("rayon-threads")
                .value_name("N")
                .help("Override the Rayon thread count. Falls back to RAYON_NUM_THREADS, then the system CPU count.")
                .takes_value(true),
        )
        .get_matches();

    let requested_rayon_threads = resolve_rayon_thread_count(&matches)?;
    let active_rayon_threads = initialize_rayon(requested_rayon_threads)?;

    let input_path = PathBuf::from(matches.value_of("input").unwrap());
    let output_path = PathBuf::from(matches.value_of("output").unwrap());
    let selected_function = matches.value_of("function").unwrap();
    let resolution = matches.value_of("resolution").unwrap();

    println!(" ________________________________________________________");
    println!("                                                         ");
    println!(" ██     ██  ██  ███    ███  ████████   Verifiable  Image");
    println!(" ██     ██  ██  ████  ████      ███    Manipulation from");
    println!("  ██   ██   ██  ██ ████ ██     ██      Folded   zkSNARKs");
    println!("   ██ ██    ██  ██  ██  ██   ███                         ");
    println!("    ███     ██  ██      ██  ████████████ v1.3.0 ████████");
    println!(" ________________________________________________________");
    println!("| Input file: {}", input_path.display());
    println!("| Output file: {}", output_path.display());
    println!("| Selected function: {}", selected_function);
    println!("| Image resolution: {}", resolution);
    println!("| Rayon threads: {}", active_rayon_threads);
    println!(" ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾");

    let proving = prove(&input_path, selected_function, resolution)?;
    let final_output_strings = proving
        .final_outputs
        .iter()
        .map(scalar_to_decimal_string)
        .collect::<Vec<_>>();
    let start_public_input_strings = proving
        .start_public_input
        .iter()
        .map(scalar_to_decimal_string)
        .collect::<Vec<_>>();

    let artifact = RecursiveProofArtifact::<NativeRecursiveSNARK> {
        backend: "neutron-pack-90".to_string(),
        proof_kind: "recursive".to_string(),
        function: selected_function.to_string(),
        resolution: resolution.to_string(),
        num_steps: proving.num_steps,
        start_public_input: start_public_input_strings,
        final_outputs: final_output_strings.clone(),
        proof: proving.proof,
    };

    let json_string = to_string_pretty(&artifact).expect("failed to serialize recursive proof");
    let mut output_file =
        File::create(&output_path).expect("unable to create the proof output file");
    output_file
        .write_all(json_string.as_bytes())
        .expect("unable to write proof output");
    println!("Recursive proof artifact written to {}", output_path.display());

    let mut file = File::open(&output_path).expect("unable to open the proof output");
    let mut json_string = String::new();
    file.read_to_string(&mut json_string)
        .expect("unable to read the proof output");

    let artifact_roundtrip: RecursiveProofArtifact<NativeRecursiveSNARK> =
        serde_json::from_str(&json_string).expect("failed to deserialize recursive proof output");
    let roundtrip_outputs = artifact_roundtrip
        .proof
        .verify(
            &proving.pp,
            proving.num_steps,
            &proving.start_public_input,
        )
        .expect("round-trip RecursiveSNARK verification failed");
    let roundtrip_output_strings = roundtrip_outputs
        .iter()
        .map(scalar_to_decimal_string)
        .collect::<Vec<_>>();
    assert_eq!(
        roundtrip_output_strings, artifact_roundtrip.final_outputs,
        "round-trip proof outputs do not match serialized final outputs"
    );

    print_neutron_timing_summary(
        proving.recursive_creation_s,
        proving.frontend_prepare_s,
        proving.verify_s,
        &final_output_strings,
    );

    Ok(())
}
