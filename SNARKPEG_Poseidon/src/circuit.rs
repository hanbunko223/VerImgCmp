use crate::{
    coefficient::ACTIVE_COEFFICIENTS_PER_STEP,
    dctq::{d_matrix, q_value, row_active, DCTQ_BLOCKS_PER_STEP, DCTQ_BLOCK_SIZE, DCTQ_CHANNELS},
    hash::{
        pack_step_chunks, pack_step_pixels, reduce_row_hashes, row_hashes_from_chunks,
        shift24_powers, PackedPixelsStep, Scalar, PACKED_CHUNKS_PER_ROW, PIXELS_PER_CHUNK,
        STEP_DIGEST_GROUPS,
    },
    input::{DctqStep, Pixel, DCTQ_HD_WIDTH, DCTQ_STEP_ROWS},
    poseidon::{poseidon_hash_2_allocated, poseidon_hash_8_allocated},
};
use ff::{Field, PrimeField};
use nova_snark::{
    frontend::{num::AllocatedNum, AllocatedBit, ConstraintSystem, SynthesisError},
    traits::circuit::StepCircuit,
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct PreparedStep {
    pub step: DctqStep,
    pub packed_pixels: PackedPixelsStep,
    pub row_hashes: [Scalar; DCTQ_STEP_ROWS],
    pub step_digest: Scalar,
}

impl PreparedStep {
    pub fn from_step(step: DctqStep) -> Self {
        let packed_pixels = pack_step_pixels(&step);
        let packed_chunks = pack_step_chunks(&packed_pixels);
        let row_hashes = row_hashes_from_chunks(&packed_chunks);
        // step_digest(&step) would redo the three lines above from scratch
        // just to get here; reduce_row_hashes is the only work it adds on
        // top of what we've already computed.
        let step_digest = reduce_row_hashes(&row_hashes);
        Self {
            packed_pixels,
            row_hashes,
            step_digest,
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
        4
    }

    fn synthesize<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<Scalar>],
    ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        assert_eq!(
            z.len(),
            4,
            "dctq step circuit expects input hash, coefficient evaluation, challenge, and step counter"
        );

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

        let next_coefficient_evaluation = enforce_fused_dctq_evaluation(
            &mut cs.namespace(|| "dctq_polynomial_evaluation"),
            &step_channels,
            z[1].clone(),
            z[2].clone(),
        )?;

        let row_hashes = step_channels
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                let packed_pixels = row
                    .iter()
                    .enumerate()
                    .map(|(pixel_idx, channels)| {
                        pack_pixel_allocated(
                            &mut cs.namespace(|| format!("pack_row_{row_idx}_pixel_{pixel_idx}")),
                            channels.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                debug_assert_eq!(
                    packed_pixels.len(),
                    self.prepared.packed_pixels[row_idx].len()
                );
                let packed_chunks = packed_pixels
                    .chunks_exact(PIXELS_PER_CHUNK)
                    .enumerate()
                    .map(|(chunk_idx, pixels)| {
                        let pixel_array: [AllocatedNum<Scalar>; PIXELS_PER_CHUNK] = pixels
                            .to_vec()
                            .try_into()
                            .expect("row chunk size is fixed at 10 packed pixels");
                        pack_chunk_allocated(
                            &mut cs.namespace(|| format!("pack_row_{row_idx}_chunk_{chunk_idx}")),
                            pixel_array,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                debug_assert_eq!(packed_chunks.len(), PACKED_CHUNKS_PER_ROW);
                let chunk_array: [AllocatedNum<Scalar>; PACKED_CHUNKS_PER_ROW] = packed_chunks
                    .try_into()
                    .expect("row always packs into 16 chunk values");
                let left = std::array::from_fn(|idx| chunk_array[idx].clone());
                let right = std::array::from_fn(|idx| chunk_array[idx + 8].clone());
                let left_hash = poseidon_hash_8_allocated(
                    &mut cs.namespace(|| format!("row_{row_idx}_left_hash")),
                    &left,
                )?;
                let right_hash = poseidon_hash_8_allocated(
                    &mut cs.namespace(|| format!("row_{row_idx}_right_hash")),
                    &right,
                )?;
                let row_hash = poseidon_hash_2_allocated(
                    &mut cs.namespace(|| format!("row_{row_idx}_root_hash")),
                    [left_hash, right_hash],
                )?;

                let expected_row_hash = AllocatedNum::alloc(
                    cs.namespace(|| format!("expected_row_hash_{row_idx}")),
                    || Ok(self.prepared.row_hashes[row_idx]),
                )?;
                enforce_equal(
                    &mut cs.namespace(|| format!("row_hash_matches_prepared_{row_idx}")),
                    &row_hash,
                    &expected_row_hash,
                    "row_hash_matches_prepared",
                );
                Ok(row_hash)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let row_hash_array: [AllocatedNum<Scalar>; DCTQ_STEP_ROWS] = row_hashes
            .try_into()
            .expect("prepared step always hashes the fixed number of rows");

        let digest_allocated =
            reduce_row_hashes_allocated(&mut cs.namespace(|| "step_digest"), &row_hash_array)?;

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

        let next_input_state = poseidon_hash_2_allocated(
            &mut cs.namespace(|| "input_state_transition"),
            [z[0].clone(), digest_allocated],
        )?;

        let next_step = increment_allocated(&mut cs.namespace(|| "increment_step"), &z[3])?;

        Ok(vec![
            next_input_state,
            next_coefficient_evaluation,
            z[2].clone(),
            next_step,
        ])
    }
}

fn enforce_fused_dctq_evaluation<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    step_channels: &[Vec<[AllocatedNum<Scalar>; 3]>],
    initial_accumulator: AllocatedNum<Scalar>,
    challenge: AllocatedNum<Scalar>,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    debug_assert_eq!(step_channels.len(), DCTQ_STEP_ROWS);
    let d = d_matrix().expect("fixed DCT matrix must parse");
    let mut left = (0..DCTQ_CHANNELS)
        .map(|_| {
            (0..DCTQ_STEP_ROWS)
                .map(|_| vec![None; DCTQ_HD_WIDTH])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for channel_idx in 0..DCTQ_CHANNELS {
        for block_row_idx in 0..(DCTQ_STEP_ROWS / DCTQ_BLOCK_SIZE) {
            let block_row = block_row_idx * DCTQ_BLOCK_SIZE;
            for block_idx in 0..DCTQ_BLOCKS_PER_STEP {
                let block_col = block_idx * DCTQ_BLOCK_SIZE;

                for out_r in 0..DCTQ_BLOCK_SIZE {
                    if !row_active(channel_idx, out_r) {
                        continue;
                    }
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
                        left[channel_idx][block_row + out_r][input_col] = Some(output);
                    }
                }
            }
        }
    }

    let mut accumulator = initial_accumulator;
    let mut active_count = 0usize;
    for row_idx in 0..DCTQ_STEP_ROWS {
        let local_row = row_idx % DCTQ_BLOCK_SIZE;
        for col_idx in 0..DCTQ_HD_WIDTH {
            let local_col = col_idx % DCTQ_BLOCK_SIZE;
            let block_col = (col_idx / DCTQ_BLOCK_SIZE) * DCTQ_BLOCK_SIZE;
            for channel_idx in 0..DCTQ_CHANNELS {
                let q_coefficient = q_value(channel_idx, local_row, local_col);
                if q_coefficient == Scalar::ZERO {
                    continue;
                }
                let right_coefficients: [Scalar; DCTQ_BLOCK_SIZE] =
                    std::array::from_fn(|k| q_coefficient * d[local_col][k]);
                let left_inputs: [AllocatedNum<Scalar>; DCTQ_BLOCK_SIZE] =
                    std::array::from_fn(|k| {
                        left[channel_idx][row_idx][block_col + k]
                            .as_ref()
                            .expect("every first-stage DCT output is assigned")
                            .clone()
                    });
                accumulator = allocate_fused_horner_output(
                    &mut cs.namespace(|| {
                        format!(
                            "row_{row_idx}_col_{col_idx}_channel_{channel_idx}_fused_right_q_horner"
                        )
                    }),
                    accumulator,
                    challenge.clone(),
                    &left_inputs,
                    &right_coefficients,
                )?;
                active_count += 1;
            }
        }
    }
    debug_assert_eq!(active_count, ACTIVE_COEFFICIENTS_PER_STEP);
    Ok(accumulator)
}

fn increment_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    value: &AllocatedNum<Scalar>,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let incremented = AllocatedNum::alloc(cs.namespace(|| "incremented"), || {
        value
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)
            .map(|value| value + Scalar::ONE)
    })?;
    cs.enforce(
        || "increment by one".to_string(),
        |lc| lc + value.get_variable() + CS::one() - incremented.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(incremented)
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
        let r = channels[0]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
        let g = channels[1]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
        let b = channels[2]
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?;
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

fn pack_chunk_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    packed_pixels: [AllocatedNum<Scalar>; PIXELS_PER_CHUNK],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let packed_chunk = AllocatedNum::alloc(cs.namespace(|| "packed_chunk"), || {
        packed_pixels
            .iter()
            .enumerate()
            .try_fold(Scalar::ZERO, |acc, (idx, pixel)| {
                pixel
                    .get_value()
                    .ok_or(SynthesisError::AssignmentMissing)
                    .map(|value| acc + (value * shift24_powers()[idx]))
            })
    })?;

    cs.enforce(
        || "pack_chunk".to_string(),
        |lc| {
            packed_pixels
                .iter()
                .enumerate()
                .fold(lc, |lc_acc, (idx, pixel)| {
                    lc_acc + (shift24_powers()[idx], pixel.get_variable())
                })
                - packed_chunk.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(packed_chunk)
}

fn reduce_row_hashes_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    row_hashes: &[AllocatedNum<Scalar>; DCTQ_STEP_ROWS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let group_hashes = row_hashes
        .chunks_exact(8)
        .enumerate()
        .map(|(group_idx, group)| {
            let group_array: [AllocatedNum<Scalar>; 8] = group
                .to_vec()
                .try_into()
                .expect("group size is fixed at 8 row hashes");
            poseidon_hash_8_allocated(
                &mut cs.namespace(|| format!("row_hash_group_{group_idx}")),
                &group_array,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let group_hashes: [AllocatedNum<Scalar>; STEP_DIGEST_GROUPS] = group_hashes
        .try_into()
        .expect("row hashes always reduce to two Poseidon8 group digests");

    poseidon_hash_2_allocated(&mut cs.namespace(|| "row_hash_root"), group_hashes)
}

fn allocate_linear_combination_output<CS: ConstraintSystem<Scalar>, const N: usize>(
    cs: &mut CS,
    inputs: &[AllocatedNum<Scalar>; N],
    coeffs: &[Scalar; N],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let offset = -Scalar::from(128u64) * coeffs.iter().copied().sum::<Scalar>();
    let output = AllocatedNum::alloc(cs.namespace(|| "linear_output"), || {
        inputs
            .iter()
            .zip(coeffs.iter())
            .try_fold(offset, |acc, (input, coeff)| {
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
                .fold(lc + (offset, CS::one()), |lc_acc, (input, coeff)| {
                    lc_acc + (*coeff, input.get_variable())
                })
                - output.get_variable()
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(output)
}

fn allocate_fused_horner_output<CS: ConstraintSystem<Scalar>, const N: usize>(
    cs: &mut CS,
    accumulator: AllocatedNum<Scalar>,
    challenge: AllocatedNum<Scalar>,
    inputs: &[AllocatedNum<Scalar>; N],
    coefficients: &[Scalar; N],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let output = AllocatedNum::alloc(cs.namespace(|| "horner_output"), || {
        let product = accumulator
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)?
            * challenge
                .get_value()
                .ok_or(SynthesisError::AssignmentMissing)?;
        inputs
            .iter()
            .zip(coefficients.iter())
            .try_fold(product, |sum, (input, coefficient)| {
                input
                    .get_value()
                    .ok_or(SynthesisError::AssignmentMissing)
                    .map(|value| sum + (value * *coefficient))
            })
    })?;

    cs.enforce(
        || "fused_second_dct_q_and_horner".to_string(),
        |lc| lc + accumulator.get_variable(),
        |lc| lc + challenge.get_variable(),
        |lc| {
            inputs.iter().zip(coefficients.iter()).fold(
                lc + output.get_variable(),
                |lc_acc, (input, coefficient)| lc_acc - (*coefficient, input.get_variable()),
            )
        },
    );

    Ok(output)
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

    fn fused_horner_constraint_count(coefficient: u64) -> usize {
        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        let accumulator =
            AllocatedNum::alloc_infallible(shape_cs.namespace(|| "accumulator"), || {
                Scalar::from(3u64)
            });
        let challenge = AllocatedNum::alloc_infallible(shape_cs.namespace(|| "challenge"), || {
            Scalar::from(5u64)
        });
        let inputs = std::array::from_fn(|index| {
            AllocatedNum::alloc_infallible(shape_cs.namespace(|| format!("input_{index}")), || {
                Scalar::from((index + 1) as u64)
            })
        });
        let coefficients = [Scalar::from(coefficient); DCTQ_BLOCK_SIZE];
        allocate_fused_horner_output(
            &mut shape_cs,
            accumulator,
            challenge,
            &inputs,
            &coefficients,
        )
        .unwrap();
        shape_cs.r1cs_shape().unwrap().num_cons()
    }

    #[test]
    fn constant_magnitude_does_not_change_fused_horner_cost() {
        let one = fused_horner_constraint_count(1);
        assert_eq!(one, 1);
        assert_eq!(one, fused_horner_constraint_count(64));
        assert_eq!(one, fused_horner_constraint_count(93));
        assert_eq!(one, fused_horner_constraint_count(102));
    }

    #[test]
    fn byte_range_check_rejects_out_of_range_witness() {
        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        allocate_scalar_with_byte_range_check(
            &mut shape_cs.namespace(|| "shape_byte"),
            Scalar::ZERO,
        )
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
}
