use crate::{
    circuit::{DctqStepCircuit, PreparedStep},
    coefficient::{CoefficientError, PublicCoefficients, COEFFICIENT_FORMAT},
    dctq::{compute_dctq_flattened, dctq_matrices, scalar_to_i128, DctqError},
    hash::{chain_hash, Scalar},
    input::{resolution_spec, DctqInput, InputError, ResolutionSpec},
    statement::{
        derive_coefficient_challenge, PublicInputDigest, StatementError, CHALLENGE_SCHEME,
        INPUT_DIGEST_FORMAT,
    },
};
use ff::Field;
use nova_snark::{
    neutron::{decider::CompressedSNARK as NeutronCompressedSNARK, PublicParams, RecursiveSNARK},
    provider::ipa_pc::EvaluationEngine,
    provider::{PallasEngine, VestaEngine},
    spartan::snark::RelaxedR1CSSNARK as SpartanRelaxedR1CSSNARK,
    timing::reset_recursive_timing,
    traits::snark::default_ck_hint,
};
use rayon::prelude::*;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use thiserror::Error;

pub type PrimaryEngine = PallasEngine;
pub type SecondaryEngine = VestaEngine;
pub type NativePublicParams = PublicParams<PrimaryEngine, SecondaryEngine, DctqStepCircuit>;
pub type NativeRecursiveSNARK = RecursiveSNARK<PrimaryEngine, SecondaryEngine, DctqStepCircuit>;
type SpartanEvaluationEngine<E> = EvaluationEngine<E>;
type SpartanSNARK<E> = SpartanRelaxedR1CSSNARK<E, SpartanEvaluationEngine<E>>;
pub type SpartanCompressedSNARK = NeutronCompressedSNARK<
    PrimaryEngine,
    SecondaryEngine,
    DctqStepCircuit,
    SpartanEvaluationEngine<PrimaryEngine>,
    SpartanSNARK<PrimaryEngine>,
>;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error(transparent)]
    Input(#[from] InputError),
    #[error(transparent)]
    Dctq(#[from] DctqError),
    #[error(transparent)]
    Coefficient(#[from] CoefficientError),
    #[error(transparent)]
    Statement(#[from] StatementError),
    #[error(transparent)]
    Nova(#[from] nova_snark::errors::NovaError),
    #[error("{0}")]
    Configuration(String),
    #[error("SPEG_Poseidon_hh currently supports only `dctq`")]
    UnsupportedFunction,
    #[error("this native SNARKPEG_Poseidon branch does not support resolution {0}")]
    UnsupportedResolution(String),
    #[error("expected {expected} steps, got {actual}")]
    InvalidStepCount { expected: usize, actual: usize },
    #[error("the private image hashes to {actual}, not the externally supplied digest {expected}")]
    InputDigestMismatch { expected: String, actual: String },
    #[error("native proof final output mismatch")]
    FinalOutputMismatch,
    #[error(
        "public coefficient mismatch at step {step_index}, coefficient {coefficient_index}: expected {expected}, computed {actual}"
    )]
    PublicCoefficientMismatch {
        step_index: usize,
        coefficient_index: usize,
        expected: i64,
        actual: i128,
    },
}

pub struct ProvingResult {
    pub pp: NativePublicParams,
    pub proof: NativeRecursiveSNARK,
    pub start_public_input: Vec<Scalar>,
    pub final_outputs: Vec<Scalar>,
    pub input_digest_path: PathBuf,
    pub input_digest_format: &'static str,
    pub public_input_digest: Scalar,
    pub coefficient_path: PathBuf,
    pub coefficient_format: &'static str,
    pub public_coefficients_sha256: [u8; 32],
    pub challenge_scheme: &'static str,
    pub coefficient_challenge: Scalar,
    pub public_coefficient_evaluation: Scalar,
    pub num_steps: usize,
    pub frontend_prepare_s: f64,
    pub setup_s: f64,
    pub recursive_creation_s: f64,
    pub verify_s: f64,
}

pub struct SpartanCompressionResult {
    pub proof_json_bytes: usize,
    pub proof_json: Vec<u8>,
    pub serialization_s: f64,
    pub setup_s: f64,
    pub compression_s: f64,
    pub verify_s: f64,
    pub final_outputs: Vec<Scalar>,
}

fn validate_mode(
    selected_function: &str,
    resolution: &str,
) -> Result<&'static ResolutionSpec, ProverError> {
    if selected_function != "dctq" {
        return Err(ProverError::UnsupportedFunction);
    }
    resolution_spec(resolution)
        .ok_or_else(|| ProverError::UnsupportedResolution(resolution.to_string()))
}

fn prepare_dctq_circuits(
    input_path: &Path,
    coefficient_path: &Path,
    input_digest_path: &Path,
    spec: &ResolutionSpec,
) -> Result<
    (
        Vec<DctqStepCircuit>,
        Vec<Scalar>,
        Vec<Scalar>,
        Scalar,
        [u8; 32],
        Scalar,
        f64,
    ),
    ProverError,
> {
    let frontend_start = Instant::now();
    let input = DctqInput::load(input_path, spec)?;
    let steps = input.into_steps();
    if steps.len() != spec.step_count {
        return Err(ProverError::InvalidStepCount {
            expected: spec.step_count,
            actual: steps.len(),
        });
    }
    let public_coefficients = PublicCoefficients::load(coefficient_path, spec)?;
    let public_coefficients_sha256 = public_coefficients.canonical_digest(spec);
    let public_input_digest = PublicInputDigest::load(input_digest_path, spec)?.value;
    let coefficient_challenge =
        derive_coefficient_challenge(public_input_digest, &public_coefficients_sha256, spec)?;
    let public_coefficient_evaluation =
        public_coefficients.active_evaluation(coefficient_challenge);
    let public_steps = public_coefficients.into_steps();
    // Each step's preparation (pixel packing + Poseidon row hashing) is
    // independent of every other step, so this parallelizes cleanly --
    // unlike the recursive fold itself, which has to run one step at a
    // time since each fold depends on the previous one's output.
    let prepared_steps = steps
        .into_par_iter()
        .zip(public_steps.par_iter())
        .enumerate()
        .map(|(step_index, (step, expected_coefficients))| {
            let computed_coefficients = compute_dctq_flattened(&step)?;
            compare_public_coefficients(step_index, &computed_coefficients, expected_coefficients)?;
            Ok(PreparedStep::from_step(step))
        })
        .collect::<Result<Vec<_>, ProverError>>()?;
    let step_digests = prepared_steps
        .iter()
        .map(|step| step.step_digest)
        .collect::<Vec<_>>();
    let expected_input_digest = chain_hash(&step_digests);
    if expected_input_digest != public_input_digest {
        return Err(ProverError::InputDigestMismatch {
            expected: crate::hash::scalar_to_decimal_string(&public_input_digest),
            actual: crate::hash::scalar_to_decimal_string(&expected_input_digest),
        });
    }
    let start_public_input = vec![
        Scalar::ZERO,
        Scalar::ZERO,
        coefficient_challenge,
        Scalar::ZERO,
    ];
    let expected_final = vec![
        public_input_digest,
        public_coefficient_evaluation,
        coefficient_challenge,
        Scalar::from(spec.step_count as u64),
    ];
    let circuits = prepared_steps
        .into_iter()
        .map(DctqStepCircuit::new)
        .collect::<Vec<_>>();
    Ok((
        circuits,
        start_public_input,
        expected_final,
        public_input_digest,
        public_coefficients_sha256,
        public_coefficient_evaluation,
        frontend_start.elapsed().as_secs_f64(),
    ))
}

fn compare_public_coefficients(
    step_index: usize,
    computed: &[Scalar],
    expected: &[i64],
) -> Result<(), ProverError> {
    debug_assert_eq!(computed.len(), expected.len());
    for (coefficient_index, (computed, expected)) in
        computed.iter().zip(expected.iter()).enumerate()
    {
        let actual = scalar_to_i128(*computed);
        if actual != i128::from(*expected) {
            return Err(ProverError::PublicCoefficientMismatch {
                step_index,
                coefficient_index,
                expected: *expected,
                actual,
            });
        }
    }
    Ok(())
}

pub fn prove(
    input_path: &Path,
    coefficient_path: &Path,
    input_digest_path: &Path,
    selected_function: &str,
    resolution: &str,
) -> Result<ProvingResult, ProverError> {
    let spec = validate_mode(selected_function, resolution)?;

    let _ = dctq_matrices()?;

    let pp_start = Instant::now();
    let template_circuit = DctqStepCircuit::new(PreparedStep::zero());
    let pp =
        NativePublicParams::setup(&template_circuit, &*default_ck_hint(), &*default_ck_hint())?;
    let setup_s = pp_start.elapsed().as_secs_f64();

    reset_recursive_timing();
    let recursive_start = Instant::now();

    let (
        circuits,
        start_public_input,
        expected_final,
        public_input_digest,
        public_coefficients_sha256,
        public_coefficient_evaluation,
        frontend_prepare_s,
    ) = prepare_dctq_circuits(input_path, coefficient_path, input_digest_path, spec)?;
    println!("frontend preparation took {:.3}s", frontend_prepare_s);

    let coefficient_challenge = start_public_input[2];

    println!("Creating a RecursiveSNARK...");
    let mut recursive_snark = NativeRecursiveSNARK::new(&pp, &circuits[0], &start_public_input)?;
    let overall_steps_start = Instant::now();
    recursive_snark.prove_step(&pp, &circuits[0])?;
    print_step_progress(
        1,
        circuits.len(),
        overall_steps_start.elapsed(),
        overall_steps_start,
    );

    for (index, circuit) in circuits.iter().enumerate().skip(1) {
        let step_start = Instant::now();
        recursive_snark.prove_step(&pp, circuit)?;
        print_step_progress(
            index + 1,
            circuits.len(),
            step_start.elapsed(),
            overall_steps_start,
        );
    }

    let recursive_creation_s = recursive_start.elapsed().as_secs_f64();

    println!("Verifying a RecursiveSNARK...");
    let verify_start = Instant::now();
    let final_outputs = recursive_snark.verify(&pp, circuits.len(), &start_public_input)?;
    let verify_s = verify_start.elapsed().as_secs_f64();

    if final_outputs != expected_final {
        return Err(ProverError::FinalOutputMismatch);
    }

    Ok(ProvingResult {
        pp,
        proof: recursive_snark,
        start_public_input,
        final_outputs,
        input_digest_path: input_digest_path.to_path_buf(),
        input_digest_format: INPUT_DIGEST_FORMAT,
        public_input_digest,
        coefficient_path: coefficient_path.to_path_buf(),
        coefficient_format: COEFFICIENT_FORMAT,
        public_coefficients_sha256,
        challenge_scheme: CHALLENGE_SCHEME,
        coefficient_challenge,
        public_coefficient_evaluation,
        num_steps: circuits.len(),
        frontend_prepare_s,
        setup_s,
        recursive_creation_s,
        verify_s,
    })
}

pub fn prove_spartan_compressed(
    pp: &NativePublicParams,
    recursive_snark: &NativeRecursiveSNARK,
    num_steps: usize,
    start_public_input: &[Scalar],
) -> Result<SpartanCompressionResult, ProverError> {
    let setup_start = Instant::now();
    let (pk, vk) = SpartanCompressedSNARK::setup(pp)?;
    let setup_s = setup_start.elapsed().as_secs_f64();

    let compression_start = Instant::now();
    let compressed_snark = SpartanCompressedSNARK::prove(pp, &pk, recursive_snark)?;
    let compression_s = compression_start.elapsed().as_secs_f64();

    let serialization_start = Instant::now();
    let proof_json = serde_json::to_vec(&compressed_snark).map_err(|error| {
        ProverError::Configuration(format!(
            "failed to serialize Spartan compressed proof: {error}"
        ))
    })?;
    let serialization_s = serialization_start.elapsed().as_secs_f64();
    let proof_json_bytes = proof_json.len();

    let verify_start = Instant::now();
    let final_outputs = compressed_snark.verify(&vk, num_steps, start_public_input)?;
    let verify_s = verify_start.elapsed().as_secs_f64();

    Ok(SpartanCompressionResult {
        proof_json_bytes,
        proof_json,
        serialization_s,
        setup_s,
        compression_s,
        verify_s,
        final_outputs,
    })
}

fn format_eta(elapsed_s: f64, completed_steps: usize, total_steps: usize) -> String {
    if completed_steps == 0 || completed_steps >= total_steps {
        return "0s".to_string();
    }

    let avg_per_step = elapsed_s / completed_steps as f64;
    let remaining = avg_per_step * (total_steps - completed_steps) as f64;
    if remaining >= 60.0 {
        format!("{:.1}m", remaining / 60.0)
    } else {
        format!("{remaining:.1}s")
    }
}

fn print_step_progress(
    step_number: usize,
    total_steps: usize,
    recursive_elapsed: std::time::Duration,
    overall_start: Instant,
) {
    let overall_elapsed = overall_start.elapsed().as_secs_f64();
    let eta = format_eta(overall_elapsed, step_number, total_steps);
    println!(
        "step {step_number}/{total_steps}: recursive={:.3}s, elapsed={overall_elapsed:.3}s, eta={eta}",
        recursive_elapsed.as_secs_f64(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PreparedStep,
        dctq::{active, compute_dctq_flattened},
        hash::{pack_pixel, scalar_to_decimal_string, step_digest},
        input::{DctqStep, DCTQ_HD_WIDTH, DCTQ_STEP_ROWS},
    };
    use nova_snark::{
        frontend::{
            num::AllocatedNum,
            r1cs::{NovaShape, NovaWitness},
            shape_cs::ShapeCS,
            solver::SatisfyingAssignment,
            ConstraintSystem,
        },
        r1cs::R1CSShape,
        traits::circuit::StepCircuit,
    };
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn synthetic_step(seed: u8) -> DctqStep {
        let mut step = [[[0u8; 3]; DCTQ_HD_WIDTH]; DCTQ_STEP_ROWS];
        step[0][0] = [seed, seed.wrapping_add(1), seed.wrapping_add(2)];
        step[1][1] = [
            seed.wrapping_add(3),
            seed.wrapping_add(4),
            seed.wrapping_add(5),
        ];
        step
    }

    fn active_evaluation(circuits: &[DctqStepCircuit], challenge: Scalar) -> Scalar {
        circuits
            .iter()
            .flat_map(|circuit| compute_dctq_flattened(&circuit.prepared.step).unwrap())
            .enumerate()
            .filter(|(global_index, _)| {
                let step_index = global_index % crate::coefficient::COEFFICIENTS_PER_STEP;
                let pixel_index = step_index / 3;
                let row = pixel_index / DCTQ_HD_WIDTH;
                let col = pixel_index % DCTQ_HD_WIDTH;
                active(step_index % 3, row % 8, col % 8)
            })
            .fold(Scalar::ZERO, |accumulator, (_, coefficient)| {
                accumulator * challenge + coefficient
            })
    }

    fn expected_outputs(circuits: &[DctqStepCircuit], challenge: Scalar) -> Vec<Scalar> {
        let input_digest = chain_hash(
            &circuits
                .iter()
                .map(|circuit| circuit.prepared.step_digest)
                .collect::<Vec<_>>(),
        );
        vec![
            input_digest,
            active_evaluation(circuits, challenge),
            challenge,
            Scalar::from(circuits.len() as u64),
        ]
    }

    fn one_step_test_spec() -> ResolutionSpec {
        ResolutionSpec {
            name: "TEST",
            width: DCTQ_HD_WIDTH,
            height: DCTQ_STEP_ROWS,
            logical_rows: 2,
            padded_logical_rows: 2,
            packed_rows: DCTQ_STEP_ROWS,
            step_count: 1,
            padded_pixels: 0,
        }
    }

    #[test]
    fn prepared_step_matches_host_hash() {
        let step = synthetic_step(7);
        let prepared = PreparedStep::from_step(step);
        assert_eq!(prepared.step_digest, step_digest(&step));
    }

    #[test]
    fn host_preflight_rejects_one_mutated_public_coefficient() {
        let step = synthetic_step(11);
        let computed = compute_dctq_flattened(&step).unwrap();
        let mut expected = computed
            .iter()
            .copied()
            .map(scalar_to_i128)
            .map(|value| i64::try_from(value).unwrap())
            .collect::<Vec<_>>();
        expected[731] += 1;

        assert!(matches!(
            compare_public_coefficients(4, &computed, &expected),
            Err(ProverError::PublicCoefficientMismatch {
                step_index: 4,
                coefficient_index: 731,
                ..
            })
        ));
    }

    #[test]
    fn preparation_accepts_an_exact_versioned_public_coefficient_file() {
        let step = synthetic_step(13);
        let signed = compute_dctq_flattened(&step)
            .unwrap()
            .into_iter()
            .map(scalar_to_i128)
            .map(|value| i64::try_from(value).unwrap())
            .collect::<Vec<_>>();
        let coefficient_rows = signed
            .chunks_exact(DCTQ_HD_WIDTH * 3)
            .map(|row| {
                row.chunks_exact(3)
                    .map(|channels| channels.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let input_rows = step
            .iter()
            .map(|row| row.iter().map(|pixel| pixel.to_vec()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let input_path = std::env::temp_dir().join(format!(
            "snarkpeg-input-{}-{nonce}.json",
            std::process::id()
        ));
        let coefficient_path = std::env::temp_dir().join(format!(
            "snarkpeg-coefficients-{}-{nonce}.json",
            std::process::id()
        ));
        let input_digest_path = std::env::temp_dir().join(format!(
            "speg-hh-input-digest-{}-{nonce}.json",
            std::process::id()
        ));
        fs::write(
            &input_path,
            serde_json::to_vec(&serde_json::json!({ "original": input_rows })).unwrap(),
        )
        .unwrap();
        fs::write(
            &coefficient_path,
            serde_json::to_vec(&serde_json::json!({
                "format": crate::coefficient::COEFFICIENT_FORMAT,
                "resolution": "TEST",
                "coefficients": coefficient_rows,
            }))
            .unwrap(),
        )
        .unwrap();
        let expected_input_digest = chain_hash(&[PreparedStep::from_step(step).step_digest]);
        fs::write(
            &input_digest_path,
            serde_json::to_vec(&serde_json::json!({
                "format": crate::statement::INPUT_DIGEST_FORMAT,
                "resolution": "TEST",
                "digest": scalar_to_decimal_string(&expected_input_digest),
            }))
            .unwrap(),
        )
        .unwrap();

        let result = prepare_dctq_circuits(
            &input_path,
            &coefficient_path,
            &input_digest_path,
            &one_step_test_spec(),
        );
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&coefficient_path);
        let _ = fs::remove_file(&input_digest_path);

        let (circuits, start, expected, digest, _, evaluation, _) = result.unwrap();
        assert_eq!(circuits.len(), 1);
        assert_eq!(expected[0], digest);
        assert_eq!(expected[1], evaluation);
        assert_eq!(expected[2], start[2]);
        assert_eq!(expected[3], Scalar::ONE);
    }

    #[test]
    fn recursive_smoke_test() {
        let steps = vec![synthetic_step(1), synthetic_step(9)];
        let circuits = steps
            .into_iter()
            .map(PreparedStep::from_step)
            .map(DctqStepCircuit::new)
            .collect::<Vec<_>>();
        let pp = NativePublicParams::setup(
            &DctqStepCircuit::new(PreparedStep::zero()),
            &*default_ck_hint(),
            &*default_ck_hint(),
        )
        .unwrap();
        let challenge = Scalar::from(23u64);
        let start_public_input = vec![Scalar::ZERO, Scalar::ZERO, challenge, Scalar::ZERO];
        let mut recursive_snark =
            NativeRecursiveSNARK::new(&pp, &circuits[0], &start_public_input).unwrap();
        recursive_snark.prove_step(&pp, &circuits[0]).unwrap();
        let first_outputs = recursive_snark.verify(&pp, 1, &start_public_input).unwrap();
        assert_eq!(first_outputs, expected_outputs(&circuits[..1], challenge));
        recursive_snark.prove_step(&pp, &circuits[1]).unwrap();
        let outputs = recursive_snark
            .verify(&pp, circuits.len(), &start_public_input)
            .unwrap();
        assert_eq!(outputs, expected_outputs(&circuits, challenge));
        for i in 0..4 {
            let mut wrong = start_public_input.clone();
            wrong[i] += Scalar::ONE;
            assert!(recursive_snark.verify(&pp, circuits.len(), &wrong).is_err());
        }
        assert!(recursive_snark.verify(&pp, 1, &start_public_input).is_err());
        let mut wrong_output = expected_outputs(&circuits, challenge);
        wrong_output[1] += Scalar::ONE;
        assert_ne!(outputs, wrong_output);
    }

    #[test]
    fn step_digest_changes_when_pixel_changes() {
        let mut step = synthetic_step(3);
        let before = step_digest(&step);
        step[DCTQ_STEP_ROWS - 1][DCTQ_HD_WIDTH - 1] = [1, 2, 3];
        let after = step_digest(&step);
        assert_ne!(before, after);
        assert_ne!(pack_pixel([1, 2, 3]), pack_pixel([1, 2, 4]));
    }

    #[test]
    fn step_circuit_is_satisfiable() {
        let circuit = DctqStepCircuit::new(PreparedStep::from_step(synthetic_step(5)));

        let mut shape_cs: ShapeCS<PrimaryEngine> = ShapeCS::new();
        let challenge = Scalar::from(29u64);
        let z_values = [Scalar::ZERO, Scalar::ZERO, challenge, Scalar::ZERO];
        let z_shape = (0..4)
            .map(|index| {
                AllocatedNum::alloc_infallible(
                    shape_cs.namespace(|| format!("z_shape_{index}")),
                    || z_values[index],
                )
            })
            .collect::<Vec<_>>();
        circuit.synthesize(&mut shape_cs, &z_shape).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let padded_dimension = shape
            .num_cons()
            .max(shape.num_vars())
            .max(shape.num_io())
            .next_power_of_two();
        println!(
            "DCT-Q step constraints: {}, variables: {}, public IO: {}, padded dimension: {}",
            shape.num_cons(),
            shape.num_vars(),
            shape.num_io(),
            padded_dimension
        );
        assert_eq!(shape.num_cons(), 97_630);
        assert_eq!(padded_dimension, 131_072);
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PrimaryEngine>::new();
        let z_witness = (0..4)
            .map(|index| {
                AllocatedNum::alloc_infallible(
                    witness_cs.namespace(|| format!("z_witness_{index}")),
                    || z_values[index],
                )
            })
            .collect::<Vec<_>>();
        let outputs = circuit.synthesize(&mut witness_cs, &z_witness).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        let expected = expected_outputs(std::slice::from_ref(&circuit), challenge);
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.get_value().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
        // Bypass host preflight. Recommit each malicious witness so rejection
        // demonstrates the arithmetic constraints, not a stale commitment.
        use nova_snark::{
            frontend::Index,
            r1cs::{R1CSInstance, R1CSWitness},
        };
        let last_horner = match outputs[1].get_variable().get_unchecked() {
            Index::Aux(i) => i,
            _ => panic!(),
        };
        let first_horner = last_horner - crate::coefficient::ACTIVE_COEFFICIENTS_PER_STEP + 1;
        let first_left = first_horner - 5120;
        for index in [4usize, first_left, first_horner, last_horner] {
            let mut values = witness.W().to_vec();
            values[index] += Scalar::ONE;
            let bad = R1CSWitness::<PrimaryEngine>::new(&shape, &values).unwrap();
            let claim = R1CSInstance::new(&shape, &bad.commit(&ck), instance.X()).unwrap();
            assert!(
                shape.is_sat(&ck, &claim, &bad).is_err(),
                "mutation at {index}"
            );
        }
    }
}
