use crate::{
    dctq::{DCTQ_BLOCK_SIZE, DCTQ_BLOCKS_PER_STEP, DCTQ_CHANNELS, d_matrix, q_matrix},
    hash::{
        LATTICE_INPUTS, LATTICE_OUTPUTS, Scalar, lattice_matrix, step_digest,
    },
    input::{DCTQ_HD_WIDTH, DCTQ_STEP_ROWS, DctqStep, Pixel},
    poseidon::{poseidon_hash_2_allocated, poseidon_hash_8_allocated},
};
use ff::{Field, PrimeField};
use nova_snark::{
    frontend::{AllocatedBit, ConstraintSystem, SynthesisError, num::AllocatedNum},
    traits::circuit::StepCircuit,
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct PreparedStep {
    pub step: DctqStep,
    pub step_digest: Scalar,
}

impl PreparedStep {
    pub fn from_step(step: DctqStep) -> Self {
        Self {
            step_digest: step_digest(&step),
            step,
        }
    }

    pub fn zero() -> Self {
        Self::from_step([[[0u8; 3]; DCTQ_HD_WIDTH]; DCTQ_STEP_ROWS])
    }
}

#[derive(Clone, Debug)]
pub struct DctqStepCircuit {
    pub prepared: Arc<PreparedStep>,
}

impl DctqStepCircuit {
    pub fn new(prepared: PreparedStep) -> Self {
        Self {
            prepared: Arc::new(prepared),
        }
    }
}

impl StepCircuit<Scalar> for DctqStepCircuit {
    fn arity(&self) -> usize {
        1
    }

    fn synthesize<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<Scalar>],
    ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        assert_eq!(z.len(), 1, "dctq step circuit expects one rolling state");

        let step_channels = self
            .prepared
            .step
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                row.iter()
                    .enumerate()
                    .map(|(pixel_idx, pixel)| {
                        allocate_pixel_channels_with_range_check(
                            &mut cs.namespace(|| format!("row_{row_idx}_pixel_{pixel_idx}")),
                            *pixel,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        enforce_dctq_layers(&mut cs.namespace(|| "dctq"), &step_channels)?;

        let mut packed_pixels = Vec::with_capacity(LATTICE_INPUTS);
        for (row_idx, row) in step_channels.iter().enumerate() {
            for (pixel_idx, channels) in row.iter().enumerate() {
                let packed_pixel = pack_pixel_allocated(
                    &mut cs.namespace(|| format!("pack_row_{row_idx}_pixel_{pixel_idx}")),
                    channels.clone(),
                )?;
                packed_pixels.push(packed_pixel);
            }
        }
        debug_assert_eq!(packed_pixels.len(), LATTICE_INPUTS);
        let packed_pixel_array: [AllocatedNum<Scalar>; LATTICE_INPUTS] = packed_pixels
            .try_into()
            .expect("prepared step always packs into 10240 pixels");

        let lattice_outputs = lattice_matrix()
            .iter()
            .enumerate()
            .map(|(output_idx, coefficients)| {
                lattice_hash_output_allocated(
                    &mut cs.namespace(|| format!("lattice_hash_output_{output_idx}")),
                    coefficients,
                    &packed_pixel_array,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lattice_output_array: [AllocatedNum<Scalar>; LATTICE_OUTPUTS] = lattice_outputs
            .try_into()
            .expect("lattice hash always produces 32 outputs");

        let digest_allocated = poseidon_tree_32_allocated(
            &mut cs.namespace(|| "step_digest"),
            &lattice_output_array,
        )?;

        let expected_step_digest =
            AllocatedNum::alloc(cs.namespace(|| "expected_step_digest"), || {
                Ok(self.prepared.step_digest)
            })?;
        enforce_equal(
            &mut cs.namespace(|| "step_digest_matches_prepared"),
            &digest_allocated,
            &expected_step_digest,
            "step_digest_matches_prepared",
        );

        let next_state = poseidon_hash_2_allocated(
            &mut cs.namespace(|| "state_transition"),
            [z[0].clone(), digest_allocated],
        )?;

        Ok(vec![next_state])
    }
}

fn allocate_pixel_channels_with_range_check<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    pixel: Pixel,
) -> Result<[AllocatedNum<Scalar>; 3], SynthesisError> {
    let r = allocate_byte_with_range_check(&mut cs.namespace(|| "r"), pixel[0])?;
    let g = allocate_byte_with_range_check(&mut cs.namespace(|| "g"), pixel[1])?;
    let b = allocate_byte_with_range_check(&mut cs.namespace(|| "b"), pixel[2])?;
    Ok([r, g, b])
}

fn allocate_byte_with_range_check<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    value: u8,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    allocate_scalar_with_byte_range_check(cs, Scalar::from(u64::from(value)))
}

fn allocate_scalar_with_byte_range_check<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    value: Scalar,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let allocated = AllocatedNum::alloc(cs.namespace(|| "byte_value"), || Ok(value))?;
    let value_u64 = scalar_low_u64(value);
    let mut bits = Vec::with_capacity(8);
    for bit_idx in 0..8 {
        bits.push(AllocatedBit::alloc(
            cs.namespace(|| format!("bit_{bit_idx}")),
            value_u64.map(|raw| ((raw >> bit_idx) & 1) == 1),
        )?);
    }

    cs.enforce(
        || "recompose_byte".to_string(),
        |lc| {
            bits.iter().enumerate().fold(lc, |lc_acc, (bit_idx, bit)| {
                lc_acc + (Scalar::from(1u64 << bit_idx), bit.get_variable())
            }) - allocated.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(allocated)
}

fn pack_pixel_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    channels: [AllocatedNum<Scalar>; 3],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let packed = AllocatedNum::alloc(cs.namespace(|| "packed_pixel"), || {
        let r = channels[0].get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let g = channels[1].get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let b = channels[2].get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(r + (g * Scalar::from(256u64)) + (b * Scalar::from(65_536u64)))
    })?;

    cs.enforce(
        || "pack_pixel".to_string(),
        |lc| {
            lc + channels[0].get_variable()
                + (Scalar::from(256u64), channels[1].get_variable())
                + (Scalar::from(65_536u64), channels[2].get_variable())
                - packed.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(packed)
}

fn enforce_dctq_layers<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    step_channels: &[Vec<[AllocatedNum<Scalar>; 3]>],
) -> Result<(), SynthesisError> {
    debug_assert_eq!(step_channels.len(), DCTQ_STEP_ROWS);
    let d = d_matrix().expect("fixed DCT matrix must parse");
    let q = q_matrix().expect("fixed quantization matrix must parse");

    for channel_idx in 0..DCTQ_CHANNELS {
        for block_row_idx in 0..(DCTQ_STEP_ROWS / DCTQ_BLOCK_SIZE) {
            let block_row = block_row_idx * DCTQ_BLOCK_SIZE;
            for block_idx in 0..DCTQ_BLOCKS_PER_STEP {
                let block_col = block_idx * DCTQ_BLOCK_SIZE;

                let mut left_block = Vec::with_capacity(DCTQ_BLOCK_SIZE);
                for out_r in 0..DCTQ_BLOCK_SIZE {
                    let mut row = Vec::with_capacity(DCTQ_BLOCK_SIZE);
                    for col_offset in 0..DCTQ_BLOCK_SIZE {
                        let input_col = block_col + col_offset;
                        let inputs: [AllocatedNum<Scalar>; DCTQ_BLOCK_SIZE] =
                            std::array::from_fn(|k| {
                                step_channels[block_row + k][input_col][channel_idx].clone()
                            });
                        let output = allocate_linear_combination_output(
                            &mut cs.namespace(|| {
                                format!(
                                    "channel_{channel_idx}_block_row_{block_row_idx}_block_{block_idx}_left_r_{out_r}_c_{col_offset}"
                                )
                            }),
                            &inputs,
                            &d[out_r],
                        )?;
                        row.push(output);
                    }
                    left_block.push(row);
                }

                let mut right_block = Vec::with_capacity(DCTQ_BLOCK_SIZE);
                for row_idx in 0..DCTQ_BLOCK_SIZE {
                    let mut row = Vec::with_capacity(DCTQ_BLOCK_SIZE);
                    for out_c in 0..DCTQ_BLOCK_SIZE {
                        let inputs: [AllocatedNum<Scalar>; DCTQ_BLOCK_SIZE] =
                            std::array::from_fn(|k| left_block[row_idx][k].clone());
                        let coeffs: [Scalar; DCTQ_BLOCK_SIZE] =
                            std::array::from_fn(|k| d[k][out_c]);
                        let output = allocate_linear_combination_output(
                            &mut cs.namespace(|| {
                                format!(
                                    "channel_{channel_idx}_block_row_{block_row_idx}_block_{block_idx}_right_r_{row_idx}_c_{out_c}"
                                )
                            }),
                            &inputs,
                            &coeffs,
                        )?;
                        row.push(output);
                    }
                    right_block.push(row);
                }

                for row_idx in 0..DCTQ_BLOCK_SIZE {
                    for col_idx in 0..DCTQ_BLOCK_SIZE {
                        allocate_scaled_output(
                            &mut cs.namespace(|| {
                                format!(
                                    "channel_{channel_idx}_block_row_{block_row_idx}_block_{block_idx}_dctq_r_{row_idx}_c_{col_idx}"
                                )
                            }),
                            right_block[row_idx][col_idx].clone(),
                            q[row_idx][col_idx],
                        )?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn lattice_hash_output_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    coefficients: &[Scalar; LATTICE_INPUTS],
    inputs: &[AllocatedNum<Scalar>; LATTICE_INPUTS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| "lattice_output"), || {
        coefficients
            .iter()
            .zip(inputs.iter())
            .try_fold(Scalar::ZERO, |acc, (coefficient, input)| {
                input
                    .get_value()
                    .ok_or(SynthesisError::AssignmentMissing)
                    .map(|value| acc + (*coefficient * value))
            })
    })?;

    cs.enforce(
        || "lattice_hash_output".to_string(),
        |lc| {
            coefficients
                .iter()
                .zip(inputs.iter())
                .fold(lc, |lc_acc, (coefficient, input)| {
                    lc_acc + (*coefficient, input.get_variable())
                })
                - output.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(output)
}

fn allocate_linear_combination_output<CS: ConstraintSystem<Scalar>, const N: usize>(
    cs: &mut CS,
    inputs: &[AllocatedNum<Scalar>; N],
    coeffs: &[Scalar; N],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| "linear_output"), || {
        inputs
            .iter()
            .zip(coeffs.iter())
            .try_fold(Scalar::ZERO, |acc, (input, coeff)| {
                input
                    .get_value()
                    .ok_or(SynthesisError::AssignmentMissing)
                    .map(|value| acc + (value * *coeff))
            })
    })?;

    cs.enforce(
        || "linear_combination".to_string(),
        |lc| {
            inputs
                .iter()
                .zip(coeffs.iter())
                .fold(lc, |lc_acc, (input, coeff)| {
                    lc_acc + (*coeff, input.get_variable())
                })
                - output.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(output)
}

fn allocate_scaled_output<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    input: AllocatedNum<Scalar>,
    coeff: Scalar,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| "scaled_output"), || {
        input
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)
            .map(|value| value * coeff)
    })?;

    cs.enforce(
        || "scale_by_constant".to_string(),
        |lc| lc + (coeff, input.get_variable()) - output.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(output)
}

fn poseidon_tree_32_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    inputs: &[AllocatedNum<Scalar>; LATTICE_OUTPUTS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let level0 = (0..4)
        .map(|group_idx| {
            let group_inputs = std::array::from_fn(|offset| inputs[group_idx * 8 + offset].clone());
            poseidon_hash_8_allocated(
                &mut cs.namespace(|| format!("level0_hash_{group_idx}")),
                &group_inputs,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let level0: [AllocatedNum<Scalar>; 4] = level0
        .try_into()
        .expect("first tree level always has 4 digests");

    let left = poseidon_hash_2_allocated(
        &mut cs.namespace(|| "level1_left"),
        [level0[0].clone(), level0[1].clone()],
    )?;
    let right = poseidon_hash_2_allocated(
        &mut cs.namespace(|| "level1_right"),
        [level0[2].clone(), level0[3].clone()],
    )?;

    poseidon_hash_2_allocated(&mut cs.namespace(|| "level2_root"), [left, right])
}

fn scalar_low_u64(value: Scalar) -> Option<u64> {
    let repr = value.to_repr();
    let bytes = repr.as_ref();
    if bytes[8..].iter().any(|byte| *byte != 0) {
        return None;
    }
    let mut low = [0u8; 8];
    low.copy_from_slice(&bytes[..8]);
    Some(u64::from_le_bytes(low))
}

fn enforce_equal<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    left: &AllocatedNum<Scalar>,
    right: &AllocatedNum<Scalar>,
    name: &str,
) {
    cs.enforce(
        || name.to_string(),
        |lc| lc + left.get_variable() - right.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::lattice_digest;
    use nova_snark::{
        frontend::{
            r1cs::{NovaShape, NovaWitness},
            shape_cs::ShapeCS,
            solver::SatisfyingAssignment,
        },
        provider::PallasEngine,
        r1cs::R1CSShape,
        traits::snark::default_ck_hint,
    };

    #[test]
    fn byte_range_check_rejects_out_of_range_witness() {
        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        allocate_scalar_with_byte_range_check(&mut shape_cs.namespace(|| "shape_byte"), Scalar::ZERO)
            .unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PallasEngine>::new();
        allocate_scalar_with_byte_range_check(
            &mut witness_cs.namespace(|| "witness_byte"),
            Scalar::from(256u64),
        )
        .unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert!(shape.is_sat(&ck, &instance, &witness).is_err());
    }

    #[test]
    fn poseidon_tree_32_gadget_matches_host() {
        let inputs = std::array::from_fn(|idx| Scalar::from((idx + 5) as u64));
        let expected = lattice_digest(&inputs);

        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        let allocated_shape = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(shape_cs.namespace(|| format!("shape_input_{idx}")), || {
                inputs[idx]
            })
        });
        poseidon_tree_32_allocated(&mut shape_cs, &allocated_shape).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PallasEngine>::new();
        let allocated_witness = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                witness_cs.namespace(|| format!("witness_input_{idx}")),
                || inputs[idx],
            )
        });
        let digest = poseidon_tree_32_allocated(&mut witness_cs, &allocated_witness).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert_eq!(digest.get_value().unwrap(), expected);
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
    }
}
