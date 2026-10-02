//! Connected Nova bundles. The verifier reconstructs coefficient accumulators,
//! challenges and counters, and verifies every hash boundary and final anchor.
use crate::circuit::{DctqStepCircuit, PreparedStep};
use crate::prover::{NativePublicParams, NativeRecursiveSNARK, SpartanCompressedSNARK};
use crate::*;
use nova_snark::traits::snark::default_ck_hint;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Segment {
    index: usize,
    steps: usize,
    initial_hash: String,
    final_hash: String,
    proof: Value,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    metadata: Value,
    kind: String,
    segments: Vec<Segment>,
}
fn schedule(total: usize, size: usize) -> Vec<usize> {
    (0..total)
        .step_by(size)
        .map(|i| size.min(total - i))
        .collect()
}
fn eval_step(acc: Scalar, step: &[i64], r: Scalar) -> Scalar {
    step.iter()
        .enumerate()
        .filter(|(i, _)| dctq::active(i % 3, (i / 480) % 8, (i / 3) % 8))
        .fold(acc, |a, (_, v)| {
            a * r + coefficient::signed_coefficient_to_scalar(*v)
        })
}
pub fn execute(
    m: &clap::ArgMatches,
    st: &VerifierStatement,
    metadata: Value,
    metrics: &mut Value,
    spec: &input::ResolutionSpec,
    size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let get = |key: &str| -> Result<&Path, Box<dyn std::error::Error>> {
        Ok(Path::new(
            m.value_of(key).ok_or_else(|| format!("missing --{key}"))?,
        ))
    };
    let counts = schedule(spec.step_count, size);
    let meta = json!({"statement":metadata,"schedule":counts,"bundle_version":"connected-hash-horner-counter-v1"});
    let now = Instant::now();
    let pp = NativePublicParams::setup(
        &DctqStepCircuit::new(PreparedStep::zero()),
        &*default_ck_hint(),
        &*default_ck_hint(),
    )?;
    metrics["setup_s"] = json!(now.elapsed().as_secs_f64());
    metrics["proof_count"] = json!(counts.len());
    metrics["schedule"] = json!(counts);
    let public = PublicCoefficients::load(get("coefficients")?, spec)?.into_steps();
    let proving = m.value_of("command") != Some("verify");
    let mut scalar_h = Scalar::ZERO;
    let mut scalar_a = Scalar::ZERO;
    let mut offset = 0;
    if !proving {
        let b: Bundle = serde_json::from_slice(&std::fs::read(get("proof")?)?)?;
        if b.metadata != meta || b.segments.len() != counts.len() {
            return Err("bundle statement/schedule mismatch".into());
        }
        let now = Instant::now();
        let vk = if b.kind == "spartan" {
            Some(SpartanCompressedSNARK::setup(&pp)?.1)
        } else if b.kind == "recursive" {
            None
        } else {
            return Err("unknown bundle kind".into());
        };
        metrics["compression_setup_s"] = json!(now.elapsed().as_secs_f64());
        let mut verify_s = 0.;
        for (i, (seg, count)) in b.segments.iter().zip(&counts).enumerate() {
            if seg.index != i
                || seg.steps != *count
                || seg.initial_hash != scalar_to_decimal_string(&scalar_h)
            {
                return Err("missing/reordered/broken hash segment".into());
            }
            let z0 = vec![
                scalar_h,
                scalar_a,
                st.challenge,
                Scalar::from(offset as u64),
            ];
            for step in &public[offset..offset + count] {
                scalar_a = eval_step(scalar_a, step, st.challenge);
            }
            let final_h = statement::parse_canonical_scalar(&seg.final_hash)?;
            let expected = vec![
                final_h,
                scalar_a,
                st.challenge,
                Scalar::from((offset + count) as u64),
            ];
            let now = Instant::now();
            let result = if let Some(vk) = &vk {
                let p: SpartanCompressedSNARK = serde_json::from_value(seg.proof.clone())?;
                p.verify(vk, *count, &z0)?
            } else {
                let p: NativeRecursiveSNARK = serde_json::from_value(seg.proof.clone())?;
                p.verify(&pp, *count, &z0)?
            };
            verify_s += now.elapsed().as_secs_f64();
            if result != expected {
                return Err("segment output mismatch".into());
            }
            scalar_h = final_h;
            offset += count;
        }
        if scalar_h != st.input_digest
            || offset != spec.step_count
            || scalar_a != st.coefficient_evaluation
        {
            return Err("final bundle anchor mismatch".into());
        }
        metrics["verify_s"] = json!(verify_s);
        return Ok(());
    }
    let now = Instant::now();
    let steps = input::DctqInput::load(get("input")?, spec)?.into_steps();
    let prepared = steps
        .into_par_iter()
        .zip(public.par_iter())
        .map(|(step, expected)| -> Result<PreparedStep, String> {
            let native = dctq::compute_dctq_flattened(&step).map_err(|e| e.to_string())?;
            if native
                .iter()
                .zip(expected)
                .any(|(x, y)| dctq::scalar_to_i128(*x) != i128::from(*y))
            {
                return Err("native coefficient mismatch".into());
            }
            Ok(PreparedStep::from_step(step))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if hash::chain_hash(&prepared.iter().map(|p| p.step_digest).collect::<Vec<_>>())
        != st.input_digest
    {
        return Err("input anchor mismatch".into());
    }
    let prep_s = now.elapsed().as_secs_f64();
    let compress = m.is_present("spartan-compress");
    let now = Instant::now();
    let keys = if compress {
        Some(SpartanCompressedSNARK::setup(&pp)?)
    } else {
        None
    };
    metrics["compression_setup_s"] = json!(now.elapsed().as_secs_f64());
    let (
        mut backend_s,
        mut recursive_verify_s,
        mut compression_s,
        mut compressed_verify_s,
        mut serialization_s,
    ) = (0., 0., 0., 0., 0.);
    let (mut recursive_bytes, mut compressed_bytes) = (0usize, 0usize);
    let (mut recursive, mut compressed) = (Vec::new(), Vec::new());
    for (i, count) in counts.iter().enumerate() {
        let z0 = vec![
            scalar_h,
            scalar_a,
            st.challenge,
            Scalar::from(offset as u64),
        ];
        let initial_hash = scalar_to_decimal_string(&scalar_h);
        let now = Instant::now();
        let circuits = prepared[offset..offset + count]
            .iter()
            .cloned()
            .map(DctqStepCircuit::new)
            .collect::<Vec<_>>();
        let mut proof = NativeRecursiveSNARK::new(&pp, &circuits[0], &z0)?;
        for c in &circuits {
            proof.prove_step(&pp, c)?;
        }
        backend_s += now.elapsed().as_secs_f64();
        for (p, k) in prepared[offset..offset + count]
            .iter()
            .zip(&public[offset..offset + count])
        {
            scalar_h = poseidon::poseidon_hash_2([scalar_h, p.step_digest]);
            scalar_a = eval_step(scalar_a, k, st.challenge);
        }
        offset += count;
        let expected = vec![
            scalar_h,
            scalar_a,
            st.challenge,
            Scalar::from(offset as u64),
        ];
        let final_hash = scalar_to_decimal_string(&scalar_h);
        let now = Instant::now();
        if proof.verify(&pp, *count, &z0)? != expected {
            return Err("recursive segment mismatch".into());
        }
        recursive_verify_s += now.elapsed().as_secs_f64();
        let now = Instant::now();
        let bytes = serde_json::to_vec(&proof)?;
        recursive_bytes += bytes.len();
        recursive.push(Segment {
            index: i,
            steps: *count,
            initial_hash: initial_hash.clone(),
            final_hash: final_hash.clone(),
            proof: serde_json::from_slice(&bytes)?,
        });
        serialization_s += now.elapsed().as_secs_f64();
        if let Some((pk, vk)) = &keys {
            let now = Instant::now();
            let cp = SpartanCompressedSNARK::prove(&pp, pk, &proof)?;
            compression_s += now.elapsed().as_secs_f64();
            let now = Instant::now();
            if cp.verify(vk, *count, &z0)? != expected {
                return Err("compressed segment mismatch".into());
            }
            compressed_verify_s += now.elapsed().as_secs_f64();
            let now = Instant::now();
            let bytes = serde_json::to_vec(&cp)?;
            compressed_bytes += bytes.len();
            compressed.push(Segment {
                index: i,
                steps: *count,
                initial_hash,
                final_hash,
                proof: serde_json::from_slice(&bytes)?,
            });
            serialization_s += now.elapsed().as_secs_f64();
        }
        println!("segment {}/{} complete", i + 1, counts.len());
    }
    let now = Instant::now();
    let bytes = serde_json::to_vec(&Bundle {
        metadata: meta.clone(),
        kind: "recursive".into(),
        segments: recursive,
    })?;
    std::fs::write(get("output")?, &bytes)?;
    metrics["recursive_artifact_bytes"] = json!(bytes.len());
    if compress {
        let bytes = serde_json::to_vec(&Bundle {
            metadata: meta,
            kind: "spartan".into(),
            segments: compressed,
        })?;
        std::fs::write(get("output")?.with_extension("spartan.json"), &bytes)?;
        metrics["compressed_artifact_bytes"] = json!(bytes.len());
    }
    serialization_s += now.elapsed().as_secs_f64();
    for (key, value) in [
        ("preparation_s", prep_s),
        ("backend_s", backend_s),
        ("recursive_prover_s", prep_s + backend_s),
        ("recursive_verify_s", recursive_verify_s),
        ("compression_s", compression_s),
        ("compressed_verify_s", compressed_verify_s),
        ("serialization_s", serialization_s),
    ] {
        metrics[key] = json!(value)
    }
    metrics["recursive_proof_bytes"] = json!(recursive_bytes);
    metrics["compressed_proof_bytes"] = json!(compressed_bytes);
    Ok(())
}
