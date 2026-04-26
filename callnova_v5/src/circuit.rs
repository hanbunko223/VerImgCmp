use crate::{
    dctq::{d_matrix, q_matrix, DCTQ_BLOCKS_PER_STEP, DCTQ_BLOCK_SIZE, DCTQ_CHANNELS},
    hash::{
        pack_step_chunks, pack_step_pixels, row_hashes_from_chunks, shift24_powers, step_digest,
        PackedPixelsStep, Scalar, PACKED_CHUNKS_PER_ROW, PIXELS_PER_CHUNK, STEP_DIGEST_GROUPS,
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
        Self {
            packed_pixels,
            row_hashes,
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
            .expect("prepared step always hashes exactly 64 rows");

        let digest_allocated =
            reduce_row_hashes_64_allocated(&mut cs.namespace(|| "step_digest"), &row_hash_array)?;

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

fn reduce_row_hashes_64_allocated<CS: ConstraintSystem<Scalar>>(
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
        .expect("64 row hashes always reduce to 8 Poseidon8 group digests");

    poseidon_hash_8_allocated(&mut cs.namespace(|| "row_hash_root"), &group_hashes)
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
