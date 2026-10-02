use crate::input::{DctqStep, DCTQ_HD_WIDTH, DCTQ_STEP_ROWS};
use ff::PrimeField;
use nova_snark::{provider::PallasEngine, traits::Engine};
use std::sync::OnceLock;
use thiserror::Error;
pub type Scalar = <PallasEngine as Engine>::Scalar;
pub const DCTQ_BLOCK_SIZE: usize = 8;
pub const DCTQ_CHANNELS: usize = 3;
pub const DCTQ_BLOCKS_PER_STEP: usize = DCTQ_HD_WIDTH / 8;
pub const A: [[i64; 8]; 8] = [
    [45, 45, 45, 45, 45, 45, 45, 45],
    [63, 53, 36, 12, -12, -36, -53, -63],
    [59, 24, -24, -59, -59, -24, 24, 59],
    [53, -12, -63, -36, 36, 63, 12, -53],
    [45, -45, -45, 45, 45, -45, -45, 45],
    [36, -63, 12, 53, -53, -12, 63, -36],
    [24, -59, 59, -24, -24, 59, -59, 24],
    [12, -36, 53, -63, 63, -53, 36, -12],
];
pub const DIVISORS: [[[i64; 8]; 8]; 3] = [
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
];
pub const MULTIPLIERS: [[[i64; 8]; 8]; 3] = [
    [
        [64, 93, 102, 64, 43, 26, 20, 17],
        [85, 85, 73, 54, 39, 18, 17, 19],
        [73, 79, 64, 43, 26, 18, 15, 18],
        [73, 60, 47, 35, 20, 12, 13, 17],
        [57, 47, 28, 18, 15, 0, 0, 13],
        [43, 29, 19, 16, 13, 0, 0, 11],
        [21, 16, 13, 12, 0, 0, 0, 0],
        [14, 11, 11, 0, 0, 0, 0, 0],
    ],
    [
        [60, 57, 43, 22, 0, 0, 0, 0],
        [57, 49, 39, 16, 0, 0, 0, 0],
        [43, 39, 18, 0, 0, 0, 0, 0],
        [22, 16, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [60, 57, 43, 22, 0, 0, 0, 0],
        [57, 49, 39, 16, 0, 0, 0, 0],
        [43, 39, 18, 0, 0, 0, 0, 0],
        [22, 16, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
];
pub type BlockMatrix = [[Scalar; 8]; 8];
#[derive(Debug, Error, Clone)]
pub enum DctqError {
    #[error("integer outside supported field encoding: {value}")]
    ScalarConversion { value: i128 },
}
pub fn scalar(value: i64) -> Scalar {
    if value >= 0 {
        Scalar::from(value as u64)
    } else {
        -Scalar::from(value.unsigned_abs())
    }
}
pub fn dctq_matrices() -> Result<&'static BlockMatrix, DctqError> {
    d_matrix()
}
pub fn d_matrix() -> Result<&'static BlockMatrix, DctqError> {
    static D: OnceLock<BlockMatrix> = OnceLock::new();
    Ok(D.get_or_init(|| A.map(|r| r.map(scalar))))
}
pub fn active(ch: usize, r: usize, c: usize) -> bool {
    MULTIPLIERS[ch][r][c] != 0
}
pub fn row_active(ch: usize, r: usize) -> bool {
    (0..8).any(|c| active(ch, r, c))
}
pub fn q_value(ch: usize, r: usize, c: usize) -> Scalar {
    scalar(MULTIPLIERS[ch][r][c])
}
pub fn bounds(ch: usize, r: usize, c: usize) -> (i64, i64) {
    static B: OnceLock<[[[(i64, i64); 8]; 8]; 3]> = OnceLock::new();
    B.get_or_init(|| {
        std::array::from_fn(|ch| {
            std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    let (mut lo, mut hi) = (0, 0);
                    for i in 0..8 {
                        for j in 0..8 {
                            let w = A[r][i] * A[c][j] * MULTIPLIERS[ch][r][c];
                            lo += (-128 * w).min(127 * w);
                            hi += (-128 * w).max(127 * w);
                        }
                    }
                    (lo, hi)
                })
            })
        })
    })[ch][r][c]
}
pub fn compute_dctq_flattened(step: &DctqStep) -> Result<Vec<Scalar>, DctqError> {
    let mut result = vec![scalar(0); DCTQ_STEP_ROWS * DCTQ_HD_WIDTH * 3];
    for br in (0..DCTQ_STEP_ROWS).step_by(8) {
        for bc in (0..DCTQ_HD_WIDTH).step_by(8) {
            for ch in 0..3 {
                let mut left = [[0i64; 8]; 8];
                for r in 0..8 {
                    if !row_active(ch, r) {
                        continue;
                    }
                    for c in 0..8 {
                        left[r][c] = (0..8)
                            .map(|i| A[r][i] * (i64::from(step[br + i][bc + c][ch]) - 128))
                            .sum();
                    }
                }
                for r in 0..8 {
                    for c in 0..8 {
                        if active(ch, r, c) {
                            let y: i64 = (0..8).map(|j| left[r][j] * A[c][j]).sum();
                            let k = y * MULTIPLIERS[ch][r][c];
                            result[((br + r) * DCTQ_HD_WIDTH + bc + c) * 3 + ch] = scalar(k);
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}
pub fn scalar_to_i128(value: Scalar) -> i128 {
    let repr = value.to_repr();
    let bytes = repr.as_ref();
    let mut low = [0u8; 8];
    low.copy_from_slice(&bytes[..8]);
    let low_u64 = u64::from_le_bytes(low);
    if bytes[8..].iter().all(|byte| *byte == 0) {
        return i128::from(low_u64);
    }

    let neg = -value;
    let neg_repr = neg.to_repr();
    let neg_bytes = neg_repr.as_ref();
    let mut neg_low = [0u8; 8];
    neg_low.copy_from_slice(&neg_bytes[..8]);
    -i128::from(u64::from_le_bytes(neg_low))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_and_masks() {
        let mut counts = [0; 3];
        let mut rows = [0; 3];
        let mut global = [(0i64, 0i64); 3];
        for ch in 0..3 {
            for r in 0..8 {
                rows[ch] += row_active(ch, r) as usize;
                for c in 0..8 {
                    assert_eq!(active(ch, r, c), DIVISORS[ch][r][c] <= 97);
                    let q = DIVISORS[ch][r][c];
                    assert_eq!(
                        MULTIPLIERS[ch][r][c],
                        if q <= 97 { (2048 + q) / (2 * q) } else { 0 }
                    );
                    if active(ch, r, c) {
                        counts[ch] += 1;
                    }
                    let (l, h) = bounds(ch, r, c);
                    global[ch].0 = global[ch].0.min(l);
                    global[ch].1 = global[ch].1.max(h);
                }
            }
        }
        assert_eq!(counts, [51, 13, 13]);
        assert_eq!(rows, [8, 4, 4]);
        assert_eq!(A[0], [45; 8]);
        assert_eq!(
            global,
            [
                (-1554357600, 1554357600),
                (-995328000, 987552000),
                (-995328000, 987552000)
            ]
        );
    }
    #[test]
    fn constant_centered_zero() {
        let step = [[[128u8; 3]; DCTQ_HD_WIDTH]; DCTQ_STEP_ROWS];
        assert!(compute_dctq_flattened(&step)
            .unwrap()
            .iter()
            .all(|x| scalar_to_i128(*x) == 0));
    }
    #[test]
    fn python_differential_vectors() {
        let vectors: serde_json::Value =
            serde_json::from_str(include_str!("../experiment/test_vectors.json")).unwrap();
        for v in vectors.as_array().unwrap() {
            let mut step = [[[128u8; 3]; DCTQ_HD_WIDTH]; DCTQ_STEP_ROWS];
            for r in 0..8 {
                for c in 0..8 {
                    for ch in 0..3 {
                        step[r][c][ch] = v["pixels"][r][c][ch].as_u64().unwrap() as u8;
                    }
                }
            }
            let result = compute_dctq_flattened(&step).unwrap();
            for r in 0..8 {
                for c in 0..8 {
                    for ch in 0..3 {
                        assert_eq!(
                            scalar_to_i128(result[(r * DCTQ_HD_WIDTH + c) * 3 + ch]),
                            v["coefficients"][r][c][ch].as_i64().unwrap() as i128
                        );
                    }
                }
            }
        }
    }
}
