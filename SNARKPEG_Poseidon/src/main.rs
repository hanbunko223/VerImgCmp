pub mod frontend {
    pub use nova_snark::frontend::*;
}

mod artifact;
mod bundle;
mod circuit;
mod coefficient;
mod dctq;
mod hash;
mod input;
mod poseidon;
mod prover;
mod statement;

use crate::{
    artifact::RecursiveProofArtifact,
    coefficient::{digest_hex, PublicCoefficients, COEFFICIENT_FORMAT},
    hash::{scalar_to_decimal_string, Scalar},
    input::{resolution_spec, PIXELS_PER_STEP},
    prover::{
        prove, prove_spartan_compressed, NativePublicParams, NativeRecursiveSNARK, ProverError,
        SpartanCompressedSNARK,
    },
    statement::{
        derive_coefficient_challenge, PublicInputDigest, CHALLENGE_SCHEME, INPUT_DIGEST_FORMAT,
    },
};
use clap::{App, Arg};
use ff::Field;
use nova_snark::frontend::ConstraintSystem;
use nova_snark::timing::snapshot_recursive_timing;
use nova_snark::traits::snark::default_ck_hint;
use rayon::ThreadPoolBuilder;
use serde_json::{json, Value};
use std::time::Instant;
use std::{
    env,
    fs::File,
    io::{Read, Write},
    mem::MaybeUninit,
    path::PathBuf,
};

struct VerifierStatement {
    input_digest: Scalar,
    coefficient_sha256: [u8; 32],
    challenge: Scalar,
    coefficient_evaluation: Scalar,
    start_public_input: Vec<Scalar>,
    expected_final_outputs: Vec<Scalar>,
}

fn load_verifier_statement(
    input_digest_path: &std::path::Path,
    coefficient_path: &std::path::Path,
    spec: &input::ResolutionSpec,
) -> Result<VerifierStatement, ProverError> {
    let input_digest = PublicInputDigest::load(input_digest_path, spec)?.value;
    let coefficients = PublicCoefficients::load(coefficient_path, spec)?;
    let coefficient_sha256 = coefficients.canonical_digest(spec);
    let challenge = derive_coefficient_challenge(input_digest, &coefficient_sha256, spec)?;
    let coefficient_evaluation = coefficients.active_evaluation(challenge);
    Ok(VerifierStatement {
        input_digest,
        coefficient_sha256,
        challenge,
        coefficient_evaluation,
        start_public_input: vec![Scalar::ZERO, Scalar::ZERO, challenge, Scalar::ZERO],
        expected_final_outputs: vec![
            input_digest,
            coefficient_evaluation,
            challenge,
            Scalar::from(spec.step_count as u64),
        ],
    })
}

#[cfg(unix)]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }

    let usage = unsafe { usage.assume_init() };
    let rss = u64::try_from(usage.ru_maxrss).ok()?;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(rss)
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        Some(rss.saturating_mul(1024))
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

const ARTIFACT: &str = "poseidon-97-centered-dct128-recip1024-proof-v1";
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let begin = Instant::now();
    let mut app = App::new("poseidon_97").arg(
        Arg::with_name("command")
            .index(1)
            .required(true)
            .possible_values(&[
                "digest",
                "coefficients",
                "inspect",
                "prove",
                "verify",
                "bench",
            ]),
    );
    for name in [
        "input",
        "output",
        "coefficients",
        "input-digest",
        "proof",
        "metrics",
        "resolution",
        "threads",
        "segment-steps",
    ] {
        app = app.arg(Arg::with_name(name).long(name).takes_value(true));
    }
    app = app.arg(Arg::with_name("spartan-compress").long("spartan-compress"));
    let m = app.get_matches();
    let command = m.value_of("command").unwrap();
    let threads = m.value_of("threads").unwrap_or("8").parse::<usize>()?;
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let path = |name: &str| -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(PathBuf::from(
            m.value_of(name)
                .ok_or_else(|| format!("--{name} required"))?,
        ))
    };
    let spec = resolution_spec(m.value_of("resolution").unwrap_or("HD"))
        .ok_or("unsupported resolution")?;
    if command == "inspect" {
        use nova_snark::frontend::{r1cs::NovaShape, shape_cs::ShapeCS};
        use nova_snark::traits::circuit::StepCircuit;
        let mut cs: ShapeCS<prover::PrimaryEngine> = ShapeCS::new();
        let z = (0..4)
            .map(|i| {
                nova_snark::frontend::num::AllocatedNum::alloc_infallible(
                    &mut cs.namespace(|| format!("z{i}")),
                    || Scalar::ZERO,
                )
            })
            .collect::<Vec<_>>();
        circuit::DctqStepCircuit::new(circuit::PreparedStep::zero()).synthesize(&mut cs, &z)?;
        let shape = cs.r1cs_shape()?;
        println!(
            "{}",
            json!({"constraints":shape.num_cons(),"variables":shape.num_vars(),"padded_dimension":shape.num_cons().max(shape.num_vars()).next_power_of_two(),"active_per_step":coefficient::ACTIVE_COEFFICIENTS_PER_STEP})
        );
        return Ok(());
    }
    if command == "digest" || command == "coefficients" {
        use rayon::prelude::*;
        let steps = input::DctqInput::load(&path("input")?, spec)?.into_steps();
        let v = if command == "digest" {
            let ds = steps.par_iter().map(hash::step_digest).collect::<Vec<_>>();
            json!({"format":INPUT_DIGEST_FORMAT,"resolution":spec.name,"digest":scalar_to_decimal_string(&hash::chain_hash(&ds))})
        } else {
            let values = steps
                .par_iter()
                .map(|step| {
                    dctq::compute_dctq_flattened(step)
                        .unwrap()
                        .iter()
                        .map(|v| dctq::scalar_to_i128(*v) as i64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let rows = values
                .iter()
                .flat_map(|v| {
                    v.chunks(480)
                        .map(|row| row.chunks(3).map(|x| x.to_vec()).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();
            json!({"format":COEFFICIENT_FORMAT,"resolution":spec.name,"coefficients":rows})
        };
        std::fs::write(path("output")?, serde_json::to_vec(&v)?)?;
        return Ok(());
    }
    let digest = path("input-digest")?;
    let coefficients = path("coefficients")?;
    let st_start = Instant::now();
    let st = load_verifier_statement(&digest, &coefficients, spec)?;
    let statement_s = st_start.elapsed().as_secs_f64();
    let metadata = json!({"artifact_version":ARTIFACT,"resolution":spec.name,"steps":spec.step_count,"coefficient_format":COEFFICIENT_FORMAT,"challenge_scheme":CHALLENGE_SCHEME,
 "input_digest":scalar_to_decimal_string(&st.input_digest),"coefficient_sha256":digest_hex(&st.coefficient_sha256),"challenge":scalar_to_decimal_string(&st.challenge)});
    let mut metrics = json!({"proof_count":1,"command":command,"threads":threads,"resolution":spec.name,"steps":spec.step_count,"statement_s":statement_s});
    let segment_steps = m
        .value_of("segment-steps")
        .unwrap_or("360")
        .parse::<usize>()?;
    if segment_steps == 0 || segment_steps > 360 {
        return Err("invalid segment size".into());
    }
    if segment_steps < spec.step_count {
        bundle::execute(&m, &st, metadata, &mut metrics, spec, segment_steps)?;
        metrics["application_s"] = json!(begin.elapsed().as_secs_f64());
        metrics["peak_rss_bytes"] = json!(peak_rss_bytes());
        if m.is_present("metrics") {
            std::fs::write(path("metrics")?, serde_json::to_vec_pretty(&metrics)?)?;
        }
        println!("{}", serde_json::to_string_pretty(&metrics)?);
        return Ok(());
    }
    if command == "verify" {
        let artifact: Value = serde_json::from_slice(&std::fs::read(path("proof")?)?)?;
        if artifact["metadata"] != metadata {
            return Err("artifact statement or version mismatch".into());
        }
        let start = Instant::now();
        let template = circuit::DctqStepCircuit::new(circuit::PreparedStep::zero());
        let pp = NativePublicParams::setup(&template, &*default_ck_hint(), &*default_ck_hint())?;
        metrics["setup_s"] = json!(start.elapsed().as_secs_f64());
        let output = match artifact["kind"].as_str() {
            Some("recursive") => {
                let proof: NativeRecursiveSNARK =
                    serde_json::from_value(artifact["proof"].clone())?;
                let now = Instant::now();
                let v = proof.verify(&pp, spec.step_count, &st.start_public_input)?;
                metrics["verify_s"] = json!(now.elapsed().as_secs_f64());
                v
            }
            Some("spartan") => {
                let proof: SpartanCompressedSNARK =
                    serde_json::from_value(artifact["proof"].clone())?;
                let now = Instant::now();
                let (_, vk) = SpartanCompressedSNARK::setup(&pp)?;
                metrics["compression_setup_s"] = json!(now.elapsed().as_secs_f64());
                let now = Instant::now();
                let v = proof.verify(&vk, spec.step_count, &st.start_public_input)?;
                metrics["verify_s"] = json!(now.elapsed().as_secs_f64());
                v
            }
            _ => return Err("unsupported proof kind".into()),
        };
        if output != st.expected_final_outputs {
            return Err("proof output differs from public statement".into());
        }
    } else {
        let proving = prove(&path("input")?, &coefficients, &digest, "dctq", spec.name)?;
        let output = path("output")?;
        let start = Instant::now();
        let raw = serde_json::to_vec(&proving.proof)?;
        metrics["recursive_proof_bytes"] = json!(raw.len());
        let bytes = serde_json::to_vec(
            &json!({"metadata":metadata,"kind":"recursive","proof":serde_json::from_slice::<Value>(&raw)?}),
        )?;
        std::fs::write(&output, &bytes)?;
        metrics["recursive_artifact_bytes"] = json!(bytes.len());
        metrics["serialization_s"] = json!(start.elapsed().as_secs_f64());
        metrics["setup_s"] = json!(proving.setup_s);
        metrics["preparation_s"] = json!(proving.frontend_prepare_s);
        metrics["recursive_prover_s"] = json!(proving.recursive_creation_s);
        metrics["backend_s"] = json!(proving.recursive_creation_s - proving.frontend_prepare_s);
        metrics["recursive_verify_s"] = json!(proving.verify_s);
        let timing = snapshot_recursive_timing();
        metrics["prove_step_s"] = json!(timing.neutron_prove_step_total);
        metrics["internal"] = json!({"augmented_synthesize_s":timing.neutron_augmented_synthesize,"commit_w_s":timing.commit_w,"commit_e_s":timing.neutron_commit_e,"multiply_vec_z1_s":timing.neutron_multiply_vec_z1,"multiply_vec_z2_s":timing.neutron_multiply_vec_z2,"prove_helper_s":timing.neutron_prove_helper,"fold_witness_s":timing.neutron_fold_witness});
        if m.is_present("spartan-compress") {
            let compressed = prove_spartan_compressed(
                &proving.pp,
                &proving.proof,
                spec.step_count,
                &st.start_public_input,
            )?;
            if compressed.final_outputs != st.expected_final_outputs {
                return Err("compressed output mismatch".into());
            }
            let start = Instant::now();
            let bytes = serde_json::to_vec(
                &json!({"metadata":metadata,"kind":"spartan","proof":serde_json::from_slice::<Value>(&compressed.proof_json)?}),
            )?;
            let dest = output.with_extension("spartan.json");
            std::fs::write(dest, &bytes)?;
            metrics["compressed_artifact_bytes"] = json!(bytes.len());
            metrics["compressed_proof_bytes"] = json!(compressed.proof_json_bytes);
            metrics["compressed_serialization_s"] =
                json!(compressed.serialization_s + start.elapsed().as_secs_f64());
            metrics["compression_setup_s"] = json!(compressed.setup_s);
            metrics["compression_s"] = json!(compressed.compression_s);
            metrics["compressed_verify_s"] = json!(compressed.verify_s);
        }
    }
    metrics["application_s"] = json!(begin.elapsed().as_secs_f64());
    metrics["peak_rss_bytes"] = json!(peak_rss_bytes());
    if m.is_present("metrics") {
        std::fs::write(path("metrics")?, serde_json::to_vec_pretty(&metrics)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    Ok(())
}
