use crate::{
    input::{DCTQ_HD_WIDTH, DCTQ_STEP_ROWS, DctqRow, DctqStep, Pixel},
    poseidon::{HASH8_INPUTS, poseidon_hash_2, poseidon_hash_8},
};
use ff::{Field, PrimeField};
use nova_snark::{provider::PallasEngine, traits::Engine};
use num_bigint::BigUint;
use std::sync::OnceLock;

pub type Scalar = <PallasEngine as Engine>::Scalar;

pub const LATTICE_INPUTS: usize = DCTQ_STEP_ROWS * DCTQ_HD_WIDTH;
pub const LATTICE_OUTPUTS: usize = 32;

pub type LatticeOutputs = [Scalar; LATTICE_OUTPUTS];
pub type LatticeMatrix = Vec<[Scalar; LATTICE_INPUTS]>;
pub type PackedPixelsRow = [Scalar; DCTQ_HD_WIDTH];

const LATTICE_INITIAL_STATE: u64 = 3_091_352_403_337_663_489;
const LATTICE_A_CONSTANT: u64 = 3_935_559_000_370_003_845;
const LATTICE_C_CONSTANT: u64 = 2_691_343_689_449_507_681;

static LATTICE_MATRIX: OnceLock<LatticeMatrix> = OnceLock::new();

pub fn scalar_to_decimal_string(value: &Scalar) -> String {
    BigUint::from_bytes_le(value.to_repr().as_ref()).to_str_radix(10)
}

pub fn pack_pixel(pixel: Pixel) -> Scalar {
    Scalar::from(u64::from(pixel[0]))
        + Scalar::from(u64::from(pixel[1]) * 256)
        + Scalar::from(u64::from(pixel[2]) * 65_536)
}

pub fn pack_row_pixels(row: &DctqRow) -> PackedPixelsRow {
    std::array::from_fn(|pixel_idx| pack_pixel(row[pixel_idx]))
}

pub fn pack_step_pixels(step: &DctqStep) -> [PackedPixelsRow; DCTQ_STEP_ROWS] {
    std::array::from_fn(|row_idx| pack_row_pixels(&step[row_idx]))
}

pub fn flatten_step_pixels(step: &DctqStep) -> Vec<Scalar> {
    let packed_rows = pack_step_pixels(step);
    (0..LATTICE_INPUTS)
        .map(|idx| {
            let row_idx = idx / DCTQ_HD_WIDTH;
            let pixel_idx = idx % DCTQ_HD_WIDTH;
            packed_rows[row_idx][pixel_idx]
        })
        .collect()
}

pub fn lattice_hash_channels(inputs: &[Scalar]) -> LatticeOutputs {
    assert_eq!(inputs.len(), LATTICE_INPUTS, "expected {} packed pixels", LATTICE_INPUTS);
    std::array::from_fn(|out_idx| {
        lattice_matrix()[out_idx]
            .iter()
            .zip(inputs.iter())
            .fold(Scalar::ZERO, |acc, (coefficient, input)| {
                acc + (*coefficient * *input)
            })
    })
}

pub fn lattice_digest(outputs: &LatticeOutputs) -> Scalar {
    let level0: [Scalar; 4] = std::array::from_fn(|group_idx| {
        let inputs = std::array::from_fn(|offset| outputs[group_idx * HASH8_INPUTS + offset]);
        poseidon_hash_8(&inputs)
    });
    let left = poseidon_hash_2([level0[0], level0[1]]);
    let right = poseidon_hash_2([level0[2], level0[3]]);
    poseidon_hash_2([left, right])
}

pub fn step_digest(step: &DctqStep) -> Scalar {
    let flattened = flatten_step_pixels(step);
    let lattice_outputs = lattice_hash_channels(&flattened);
    lattice_digest(&lattice_outputs)
}

pub fn chain_hash(step_digests: &[Scalar]) -> Scalar {
    step_digests.iter().fold(Scalar::ZERO, |state, step_digest| {
        poseidon_hash_2([state, *step_digest])
    })
}

pub fn lattice_matrix() -> &'static LatticeMatrix {
    LATTICE_MATRIX.get_or_init(|| {
        let mut matrix = Vec::with_capacity(LATTICE_OUTPUTS);
        let mut state = LATTICE_INITIAL_STATE;
        for _ in 0..LATTICE_OUTPUTS {
            let row = std::array::from_fn(|_| Scalar::from(next_lattice_coefficient(&mut state)));
            matrix.push(row);
        }
        matrix
    })
}

fn next_lattice_coefficient(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(LATTICE_A_CONSTANT)
        .wrapping_add(LATTICE_C_CONSTANT);
    let x1 = *state >> 32;
    *state = state
        .wrapping_mul(LATTICE_A_CONSTANT)
        .wrapping_add(LATTICE_C_CONSTANT);
    (*state & 0xffff_ffff_0000_0000).wrapping_add(x1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(offset: usize) -> DctqRow {
        std::array::from_fn(|idx| {
            [
                ((offset + idx) % 256) as u8,
                ((offset + idx + 1) % 256) as u8,
                ((offset + idx + 2) % 256) as u8,
            ]
        })
    }

    fn sample_step() -> DctqStep {
        std::array::from_fn(|row_idx| sample_row(row_idx * DCTQ_HD_WIDTH))
    }

    #[test]
    fn pack_pixel_is_rgb_base256() {
        let packed = pack_pixel([1, 2, 3]);
        let expected = Scalar::from(1u64) + Scalar::from(512u64) + Scalar::from(196_608u64);
        assert_eq!(packed, expected);
    }

    #[test]
    fn flatten_step_pixels_is_row_major() {
        let mut step = [[[0u8; 3]; DCTQ_HD_WIDTH]; DCTQ_STEP_ROWS];
        step[0][0] = [1, 2, 3];
        step[0][1] = [4, 5, 6];
        step[1][0] = [7, 8, 9];

        let flattened = flatten_step_pixels(&step);
        assert_eq!(flattened[0], pack_pixel([1, 2, 3]));
        assert_eq!(flattened[1], pack_pixel([4, 5, 6]));
        assert_eq!(flattened[DCTQ_HD_WIDTH], pack_pixel([7, 8, 9]));
    }

    #[test]
    fn lattice_matrix_is_deterministic() {
        assert_eq!(lattice_matrix(), lattice_matrix());
        assert_ne!(lattice_matrix()[0][0], lattice_matrix()[0][1]);
    }

    #[test]
    fn lattice_hash_changes_when_one_channel_changes() {
        let step = sample_step();
        let before = lattice_hash_channels(&flatten_step_pixels(&step));
        let mut changed = step;
        changed[DCTQ_STEP_ROWS - 1][159][2] = changed[DCTQ_STEP_ROWS - 1][159][2].wrapping_add(1);
        let after = lattice_hash_channels(&flatten_step_pixels(&changed));
        assert_ne!(before, after);
    }

    #[test]
    fn step_digest_changes_when_last_channel_changes() {
        let step = sample_step();
        let mut changed = step;
        changed[DCTQ_STEP_ROWS - 1][159][2] = changed[DCTQ_STEP_ROWS - 1][159][2].wrapping_add(1);
        assert_ne!(step_digest(&step), step_digest(&changed));
    }

    #[test]
    fn chain_hash_changes_when_step_digest_changes() {
        let step = sample_step();
        let digest_a = step_digest(&step);
        let mut changed = step;
        changed[DCTQ_STEP_ROWS - 1][159][2] = changed[DCTQ_STEP_ROWS - 1][159][2].wrapping_add(1);
        let digest_b = step_digest(&changed);
        assert_ne!(digest_a, digest_b);
        assert_ne!(chain_hash(&[digest_a]), chain_hash(&[digest_b]));
    }

    #[test]
    fn lattice_digest_uses_fixed_poseidon_tree() {
        let outputs = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        let level0: [Scalar; 4] = std::array::from_fn(|group_idx| {
            let inputs = std::array::from_fn(|offset| outputs[group_idx * HASH8_INPUTS + offset]);
            poseidon_hash_8(&inputs)
        });
        let expected = poseidon_hash_2([
            poseidon_hash_2([level0[0], level0[1]]),
            poseidon_hash_2([level0[2], level0[3]]),
        ]);
        assert_eq!(lattice_digest(&outputs), expected);
    }
}
