use crate::{
    input::{DCTQ_HD_WIDTH, DCTQ_STEP_ROWS, DctqRow, DctqStep, Pixel},
    poseidon::{HASH8_INPUTS, poseidon_hash_2, poseidon_hash_8},
};
use ff::{Field, PrimeField};
use nova_snark::{provider::PallasEngine, traits::Engine};
use num_bigint::BigUint;
use std::sync::OnceLock;

pub type Scalar = <PallasEngine as Engine>::Scalar;

pub const PIXELS_PER_CHUNK: usize = 10;
pub const PACKED_CHUNKS_PER_ROW: usize = DCTQ_HD_WIDTH / PIXELS_PER_CHUNK;

pub type PackedPixelsRow = [Scalar; DCTQ_HD_WIDTH];
pub type PackedChunksRow = [Scalar; PACKED_CHUNKS_PER_ROW];
pub type PackedPixelsStep = Box<[PackedPixelsRow; DCTQ_STEP_ROWS]>;
pub type PackedChunksStep = Box<[PackedChunksRow; DCTQ_STEP_ROWS]>;
pub const STEP_DIGEST_GROUPS: usize = DCTQ_STEP_ROWS / HASH8_INPUTS;

static SHIFT24_POWERS: OnceLock<[Scalar; PIXELS_PER_CHUNK]> = OnceLock::new();

pub fn scalar_to_decimal_string(value: &Scalar) -> String {
    BigUint::from_bytes_le(value.to_repr().as_ref()).to_str_radix(10)
}

pub fn pack_pixel(pixel: Pixel) -> Scalar {
    Scalar::from(u64::from(pixel[0]))
        + Scalar::from(u64::from(pixel[1]) * 256)
        + Scalar::from(u64::from(pixel[2]) * 65_536)
}

pub fn pack_ten_pixels(packed_pixels: &[Scalar; PIXELS_PER_CHUNK]) -> Scalar {
    packed_pixels
        .iter()
        .zip(shift24_powers().iter())
        .fold(Scalar::ZERO, |acc, (pixel, coeff)| acc + (*pixel * *coeff))
}

pub fn pack_row_pixels(row: &DctqRow) -> PackedPixelsRow {
    std::array::from_fn(|pixel_idx| pack_pixel(row[pixel_idx]))
}

pub fn pack_row_chunks(packed_pixels: &PackedPixelsRow) -> PackedChunksRow {
    std::array::from_fn(|chunk_idx| {
        let start = chunk_idx * PIXELS_PER_CHUNK;
        let chunk_pixels = std::array::from_fn(|offset| packed_pixels[start + offset]);
        pack_ten_pixels(&chunk_pixels)
    })
}

pub fn row_hash_from_chunks(chunks: &PackedChunksRow) -> Scalar {
    let left = std::array::from_fn(|idx| chunks[idx]);
    let right = std::array::from_fn(|idx| chunks[idx + 8]);
    poseidon_hash_2([poseidon_hash_8(&left), poseidon_hash_8(&right)])
}

#[cfg(test)]
pub fn row_hash(row: &DctqRow) -> Scalar {
    let packed_pixels = pack_row_pixels(row);
    let packed_chunks = pack_row_chunks(&packed_pixels);
    row_hash_from_chunks(&packed_chunks)
}

pub fn pack_step_pixels(step: &DctqStep) -> PackedPixelsStep {
    (0..DCTQ_STEP_ROWS)
        .map(|row_idx| pack_row_pixels(&step[row_idx]))
        .collect::<Vec<_>>()
        .into_boxed_slice()
        .try_into()
        .expect("step always packs into 64 rows")
}

pub fn pack_step_chunks(
    packed_pixels: &[PackedPixelsRow; DCTQ_STEP_ROWS],
) -> PackedChunksStep {
    (0..DCTQ_STEP_ROWS)
        .map(|row_idx| pack_row_chunks(&packed_pixels[row_idx]))
        .collect::<Vec<_>>()
        .into_boxed_slice()
        .try_into()
        .expect("step always packs into 64 chunk rows")
}

pub fn row_hashes_from_chunks(
    packed_chunks: &[PackedChunksRow; DCTQ_STEP_ROWS],
) -> [Scalar; DCTQ_STEP_ROWS] {
    std::array::from_fn(|row_idx| row_hash_from_chunks(&packed_chunks[row_idx]))
}

pub fn reduce_row_hashes_64(row_hashes: &[Scalar; DCTQ_STEP_ROWS]) -> Scalar {
    let group_hashes: [Scalar; STEP_DIGEST_GROUPS] = std::array::from_fn(|group_idx| {
        let inputs: [Scalar; HASH8_INPUTS] =
            std::array::from_fn(|offset| row_hashes[group_idx * HASH8_INPUTS + offset]);
        poseidon_hash_8(&inputs)
    });
    poseidon_hash_8(&group_hashes)
}

pub fn step_digest(step: &DctqStep) -> Scalar {
    let packed_pixels = pack_step_pixels(step);
    let packed_chunks = pack_step_chunks(&packed_pixels);
    let row_hashes = row_hashes_from_chunks(&packed_chunks);
    reduce_row_hashes_64(&row_hashes)
}

pub fn chain_hash(step_digests: &[Scalar]) -> Scalar {
    step_digests.iter().fold(Scalar::ZERO, |state, step_digest| {
        poseidon_hash_2([state, *step_digest])
    })
}

pub fn shift24_powers() -> &'static [Scalar; PIXELS_PER_CHUNK] {
    SHIFT24_POWERS.get_or_init(|| {
        std::array::from_fn(|idx| {
            if idx == 0 {
                Scalar::ONE
            } else {
                Scalar::from(2u64).pow_vartime([(24 * idx) as u64])
            }
        })
    })
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
    fn pack_ten_pixels_uses_24_bit_slots() {
        let pixels = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        let expected = pixels
            .iter()
            .enumerate()
            .fold(Scalar::ZERO, |acc, (idx, pixel)| {
                acc + (*pixel * shift24_powers()[idx])
            });
        assert_eq!(pack_ten_pixels(&pixels), expected);
    }

    #[test]
    fn row_hash_changes_when_one_channel_changes() {
        let mut row = sample_row(11);
        let before = row_hash(&row);
        row[159][2] = row[159][2].wrapping_add(1);
        let after = row_hash(&row);
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
    fn reduce_row_hashes_64_uses_fixed_poseidon8_tree() {
        let row_hashes = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        let group_hashes: [Scalar; STEP_DIGEST_GROUPS] = std::array::from_fn(|group_idx| {
            let inputs: [Scalar; HASH8_INPUTS] =
                std::array::from_fn(|offset| row_hashes[group_idx * HASH8_INPUTS + offset]);
            poseidon_hash_8(&inputs)
        });
        assert_eq!(reduce_row_hashes_64(&row_hashes), poseidon_hash_8(&group_hashes));
    }
}
