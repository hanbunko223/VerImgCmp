use crate::{
    circuit::{DctqStepCircuit, PreparedStep},
    dctq::{DctqError, dctq_matrices},
    hash::{Scalar, chain_hash},
    input::{DCTQ_HD_STEP_COUNT, DctqInput, InputError},
};
use ff::Field;
use nova_snark::{
    neutron::{PublicParams, RecursiveSNARK},
    provider::{PallasEngine, VestaEngine},
    timing::reset_recursive_timing,
    traits::{Engine, snark::default_ck_hint},
};
use std::{path::Path, time::Instant};
use thiserror::Error;

pub type PrimaryEngine = PallasEngine;
pub type SecondaryEngine = VestaEngine;
pub type NativePublicParams = PublicParams<PrimaryEngine, SecondaryEngine, DctqStepCircuit>;
pub type NativeRecursiveSNARK =
    RecursiveSNARK<PrimaryEngine, SecondaryEngine, DctqStepCircuit>;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error(transparent)]
    Input(#[from] InputError),
    #[error(transparent)]
    Dctq(#[from] DctqError),
    #[error(transparent)]
    Nova(#[from] nova_snark::errors::NovaError),
    #[error("{0}")]
    Configuration(String),
    #[error("this native callnova branch currently supports only `dctq`")]
    UnsupportedFunction,
    #[error("this native callnova branch currently supports only `HD`, got {0}")]
    UnsupportedResolution(String),
    #[error("expected {expected} HD steps, got {actual}")]
    InvalidStepCount { expected: usize, actual: usize },
    #[error("native proof final output mismatch")]
    FinalOutputMismatch,
}

pub struct ProvingResult {
    pub pp: NativePublicParams,
    pub proof: NativeRecursiveSNARK,
    pub start_public_input: Vec<Scalar>,
    pub final_outputs: Vec<Scalar>,
    pub num_steps: usize,
    pub frontend_prepare_s: f64,
    pub recursive_creation_s: f64,
    pub verify_s: f64,
}

pub fn prove(
    input_path: &Path,
    selected_function: &str,
    resolution: &str,
) -> Result<ProvingResult, ProverError> {
    if selected_function != "dctq" {
        return Err(ProverError::UnsupportedFunction);
    }
    if resolution != "HD" {
        return Err(ProverError::UnsupportedResolution(resolution.to_string()));
    }

    println!(
        "Running NeutronNova with native Rust packed-byte frontend plus internal DCTQ checks and engine: {}",
        std::any::type_name::<PrimaryEngine>()
    );
    println!(
        "Warning: using experimental NeutronNova backend from local nova60/Nova; this branch enforces byte range checks, internal DCTQ linear layers, and packed hashing without a compressed proof."
    );

    let _ = dctq_matrices()?;

    let pp_start = Instant::now();
    let template_circuit = DctqStepCircuit::new(PreparedStep::zero());
    let pp = NativePublicParams::setup(
        &template_circuit,
        &*default_ck_hint(),
        &*default_ck_hint(),
    )?;
    let pp_s = pp_start.elapsed().as_secs_f64();
    println!("Creating keys from native step circuit took {:.3}s", pp_s);

    reset_recursive_timing();
    let recursive_start = Instant::now();

    let frontend_start = Instant::now();
    let input = DctqInput::load(input_path)?;
    let steps = input.into_steps();
    if steps.len() != DCTQ_HD_STEP_COUNT {
        return Err(ProverError::InvalidStepCount {
            expected: DCTQ_HD_STEP_COUNT,
            actual: steps.len(),
        });
    }
    let prepared_steps = steps
        .into_iter()
        .map(PreparedStep::from_step)
        .collect::<Vec<_>>();
    let step_digests = prepared_steps
        .iter()
        .map(|step| step.step_digest)
        .collect::<Vec<_>>();
    let expected_final = chain_hash(&step_digests);
    let circuits = prepared_steps
        .into_iter()
        .map(DctqStepCircuit::new)
        .collect::<Vec<_>>();
    let frontend_prepare_s = frontend_start.elapsed().as_secs_f64();
    println!("frontend preparation took {:.3}s", frontend_prepare_s);

    let start_public_input = vec![<PrimaryEngine as Engine>::Scalar::ZERO];

    println!("Creating a RecursiveSNARK...");
    let mut recursive_snark =
        NativeRecursiveSNARK::new(&pp, &circuits[0], &start_public_input)?;
    let overall_steps_start = Instant::now();
    recursive_snark.prove_step(&pp, &circuits[0])?;
    print_step_progress(1, circuits.len(), overall_steps_start.elapsed(), overall_steps_start);

    for (index, circuit) in circuits.iter().enumerate().skip(1) {
        let step_start = Instant::now();
        recursive_snark.prove_step(&pp, circuit)?;
        print_step_progress(index + 1, circuits.len(), step_start.elapsed(), overall_steps_start);
    }

    let recursive_creation_s = recursive_start.elapsed().as_secs_f64();
    println!("RecursiveSNARK creation took {:.3}s", recursive_creation_s);

    println!("Verifying a RecursiveSNARK...");
    let verify_start = Instant::now();
    let final_outputs =
        recursive_snark.verify(&pp, circuits.len(), &start_public_input)?;
    let verify_s = verify_start.elapsed().as_secs_f64();
    println!("RecursiveSNARK::verify: true, took {:.3}s", verify_s);

    if final_outputs.as_slice() != [expected_final] {
        return Err(ProverError::FinalOutputMismatch);
    }

    Ok(ProvingResult {
        pp,
        proof: recursive_snark,
        start_public_input,
        final_outputs,
        num_steps: circuits.len(),
        frontend_prepare_s,
        recursive_creation_s,
        verify_s,
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
        hash::{pack_pixel, step_digest},
        input::{DCTQ_HD_WIDTH, DCTQ_STEP_ROWS, DctqStep},
    };
    use nova_snark::{
        frontend::{
            ConstraintSystem,
            num::AllocatedNum,
            r1cs::{NovaShape, NovaWitness},
            shape_cs::ShapeCS,
            solver::SatisfyingAssignment,
        },
        r1cs::R1CSShape,
        traits::circuit::StepCircuit,
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

    #[test]
    fn prepared_step_matches_host_hash() {
        let step = synthetic_step(7);
        let prepared = PreparedStep::from_step(step);
        assert_eq!(prepared.step_digest, step_digest(&step));
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
        let start_public_input = vec![<PrimaryEngine as Engine>::Scalar::ZERO];
        let mut recursive_snark =
            NativeRecursiveSNARK::new(&pp, &circuits[0], &start_public_input).unwrap();
        recursive_snark.prove_step(&pp, &circuits[0]).unwrap();
        let first_outputs = recursive_snark
            .verify(&pp, 1, &start_public_input)
            .unwrap();
        assert_eq!(
            first_outputs,
            vec![chain_hash(&[circuits[0].prepared.step_digest])]
        );
        recursive_snark.prove_step(&pp, &circuits[1]).unwrap();
        let outputs = recursive_snark
            .verify(&pp, circuits.len(), &start_public_input)
            .unwrap();
        let expected = chain_hash(
            &circuits
                .iter()
                .map(|c| c.prepared.step_digest)
                .collect::<Vec<_>>(),
        );
        assert_eq!(outputs, vec![expected]);
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
        let z_shape =
            AllocatedNum::alloc_infallible(shape_cs.namespace(|| "z_shape"), || Scalar::ZERO);
        circuit.synthesize(&mut shape_cs, &[z_shape]).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PrimaryEngine>::new();
        let z_witness =
            AllocatedNum::alloc_infallible(witness_cs.namespace(|| "z_witness"), || Scalar::ZERO);
        let outputs = circuit.synthesize(&mut witness_cs, &[z_witness]).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert_eq!(
            outputs[0].get_value().unwrap(),
            chain_hash(&[circuit.prepared.step_digest])
        );
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
    }
}
