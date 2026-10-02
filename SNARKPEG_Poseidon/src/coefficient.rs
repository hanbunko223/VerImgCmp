use crate::{
    dctq::{active, bounds},
    hash::Scalar,
    input::{resolution_tag, ResolutionSpec, DCTQ_HD_WIDTH, DCTQ_STEP_ROWS},
};
use ff::Field;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
use thiserror::Error;

pub const COEFFICIENT_FORMAT: &str = "poseidon-97-centered-dct128-recip1024-coefficients-v1";
pub const COEFFICIENT_HASH_DOMAIN: &[u8] =
    b"POSEIDON_97_CENTERED_DCT128_RECIP1024_COEFFICIENTS_V1\0";
pub const COEFFICIENT_CHANNELS: usize = 3;
pub const COEFFICIENT_BIAS: u64 = 1u64 << 39;
pub const MAX_ABS_COEFFICIENT: i64 = 1_554_357_600;
pub const COEFFICIENTS_PER_STEP: usize = DCTQ_STEP_ROWS * DCTQ_HD_WIDTH * COEFFICIENT_CHANNELS;
pub const ACTIVE_COEFFICIENTS_PER_STEP: usize = 40 * (51 + 13 + 13);

pub type CoefficientStep = Vec<i64>;

#[derive(Debug, Error)]
pub enum CoefficientError {
    #[error("failed to open public coefficient file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read public coefficient file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse public coefficient JSON {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported public coefficient format {actual}; expected {expected}")]
    InvalidFormat {
        expected: &'static str,
        actual: String,
    },
    #[error(
        "public coefficient resolution {actual} does not match requested resolution {expected}"
    )]
    InvalidResolution {
        expected: &'static str,
        actual: String,
    },
    #[error(
        "invalid {resolution} public coefficients: expected {expected} packed rows, got {actual}"
    )]
    InvalidRowCount {
        resolution: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "invalid public coefficient row {row_index}: expected {expected} entries, got {actual}"
    )]
    InvalidRowWidth {
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("invalid public coefficient at row {row_index}, col {col_index}: expected 3 channels, got {actual}")]
    InvalidChannelCount {
        row_index: usize,
        col_index: usize,
        actual: usize,
    },
    #[error("public coefficient at row {row_index}, col {col_index}, channel {channel_index} is out of range: {value}")]
    OutOfRange {
        row_index: usize,
        col_index: usize,
        channel_index: usize,
        value: i64,
    },
    #[error("public coefficient at fixed zero-Q position row {row_index}, col {col_index}, channel {channel_index} must be zero, got {value}")]
    NonZeroAtZeroQ {
        row_index: usize,
        col_index: usize,
        channel_index: usize,
        value: i64,
    },
}

#[derive(Debug, Deserialize)]
struct PublicCoefficientJson {
    format: String,
    resolution: String,
    coefficients: Vec<Vec<Vec<i64>>>,
}

#[derive(Clone, Debug)]
pub struct PublicCoefficients {
    steps: Vec<CoefficientStep>,
}

impl PublicCoefficients {
    pub fn load(path: &Path, spec: &ResolutionSpec) -> Result<Self, CoefficientError> {
        let mut file = File::open(path).map_err(|source| CoefficientError::Open {
            path: path.display().to_string(),
            source,
        })?;
        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|source| CoefficientError::Read {
                path: path.display().to_string(),
                source,
            })?;
        let parsed: PublicCoefficientJson =
            serde_json::from_str(&json).map_err(|source| CoefficientError::Parse {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_parsed(parsed, spec)
    }

    fn from_parsed(
        parsed: PublicCoefficientJson,
        spec: &ResolutionSpec,
    ) -> Result<Self, CoefficientError> {
        if parsed.format != COEFFICIENT_FORMAT {
            return Err(CoefficientError::InvalidFormat {
                expected: COEFFICIENT_FORMAT,
                actual: parsed.format,
            });
        }
        if parsed.resolution != spec.name {
            return Err(CoefficientError::InvalidResolution {
                expected: spec.name,
                actual: parsed.resolution,
            });
        }
        if parsed.coefficients.len() != spec.packed_rows {
            return Err(CoefficientError::InvalidRowCount {
                resolution: spec.name,
                expected: spec.packed_rows,
                actual: parsed.coefficients.len(),
            });
        }

        let mut flattened = Vec::with_capacity(spec.packed_rows * DCTQ_HD_WIDTH * 3);
        for (row_index, row) in parsed.coefficients.into_iter().enumerate() {
            if row.len() != DCTQ_HD_WIDTH {
                return Err(CoefficientError::InvalidRowWidth {
                    row_index,
                    expected: DCTQ_HD_WIDTH,
                    actual: row.len(),
                });
            }
            for (col_index, channels) in row.into_iter().enumerate() {
                if channels.len() != COEFFICIENT_CHANNELS {
                    return Err(CoefficientError::InvalidChannelCount {
                        row_index,
                        col_index,
                        actual: channels.len(),
                    });
                }
                for (channel_index, value) in channels.into_iter().enumerate() {
                    let is_zero_q = !active(channel_index, row_index % 8, col_index % 8);
                    let (lo, hi) = bounds(channel_index, row_index % 8, col_index % 8);
                    if !is_zero_q && !(lo..=hi).contains(&value) {
                        return Err(CoefficientError::OutOfRange {
                            row_index,
                            col_index,
                            channel_index,
                            value,
                        });
                    }
                    if is_zero_q && value != 0 {
                        return Err(CoefficientError::NonZeroAtZeroQ {
                            row_index,
                            col_index,
                            channel_index,
                            value,
                        });
                    }
                    flattened.push(value);
                }
            }
        }

        let steps = flattened
            .chunks_exact(COEFFICIENTS_PER_STEP)
            .map(|step| step.to_vec())
            .collect::<Vec<_>>();
        debug_assert_eq!(steps.len(), spec.step_count);
        Ok(Self { steps })
    }

    pub fn into_steps(self) -> Vec<CoefficientStep> {
        self.steps
    }

    pub fn canonical_digest(&self, spec: &ResolutionSpec) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(COEFFICIENT_HASH_DOMAIN);
        hasher.update(resolution_tag(spec).to_le_bytes());
        hasher.update((spec.step_count as u64).to_le_bytes());
        let total = self.steps.len() * COEFFICIENTS_PER_STEP;
        hasher.update((total as u64).to_le_bytes());
        for value in self.steps.iter().flatten() {
            let biased = (i128::from(*value) + i128::from(COEFFICIENT_BIAS)) as u64;
            hasher.update(&biased.to_le_bytes()[..5]);
        }
        hasher.finalize().into()
    }

    pub fn active_evaluation(&self, challenge: Scalar) -> Scalar {
        self.steps
            .iter()
            .flat_map(|step| step.iter().enumerate())
            .filter(|(index, _)| {
                let pixel_index = index / COEFFICIENT_CHANNELS;
                let row = pixel_index / DCTQ_HD_WIDTH;
                let col = pixel_index % DCTQ_HD_WIDTH;
                active(index % 3, row % 8, col % 8)
            })
            .fold(Scalar::ZERO, |accumulator, (_, coefficient)| {
                accumulator * challenge + signed_coefficient_to_scalar(*coefficient)
            })
    }
}

pub fn signed_coefficient_to_scalar(value: i64) -> Scalar {
    if value >= 0 {
        Scalar::from(value as u64)
    } else {
        -Scalar::from(value.unsigned_abs())
    }
}

pub fn coefficient_digest_limbs(digest: &[u8; 32]) -> [Scalar; 4] {
    std::array::from_fn(|index| {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[index * 8..(index + 1) * 8]);
        Scalar::from(u64::from_le_bytes(bytes))
    })
}

pub fn digest_hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statement::derive_coefficient_challenge;

    fn test_spec() -> ResolutionSpec {
        ResolutionSpec {
            name: "TEST",
            width: 160,
            height: 16,
            logical_rows: 2,
            padded_logical_rows: 2,
            packed_rows: DCTQ_STEP_ROWS,
            step_count: 1,
            padded_pixels: 0,
        }
    }

    fn valid_parsed() -> PublicCoefficientJson {
        PublicCoefficientJson {
            format: COEFFICIENT_FORMAT.to_string(),
            resolution: "TEST".to_string(),
            coefficients: vec![vec![vec![0; 3]; DCTQ_HD_WIDTH]; DCTQ_STEP_ROWS],
        }
    }

    #[test]
    fn constants_match_transform_shape() {
        assert!(MAX_ABS_COEFFICIENT < COEFFICIENT_BIAS as i64);
        assert_eq!(COEFFICIENTS_PER_STEP, 7_680);
        assert_eq!(ACTIVE_COEFFICIENTS_PER_STEP, 3_080);
    }

    #[test]
    fn semantic_digest_and_evaluation_bind_value_sign_and_order() {
        let spec = test_spec();
        let mut first = valid_parsed();
        first.coefficients[0][0] = vec![1, -2, 3];
        first.coefficients[1][0] = vec![4, -5, 6];
        let first = PublicCoefficients::from_parsed(first, &spec).unwrap();

        let mut changed = valid_parsed();
        changed.coefficients[0][0] = vec![-1, -2, 3];
        changed.coefficients[1][0] = vec![4, -5, 6];
        let changed = PublicCoefficients::from_parsed(changed, &spec).unwrap();
        assert_ne!(
            first.canonical_digest(&spec),
            changed.canonical_digest(&spec)
        );
        assert_ne!(
            first.active_evaluation(Scalar::from(17u64)),
            changed.active_evaluation(Scalar::from(17u64))
        );

        let mut reordered = valid_parsed();
        reordered.coefficients[0][0] = vec![3, -2, 1];
        reordered.coefficients[1][0] = vec![4, -5, 6];
        let reordered = PublicCoefficients::from_parsed(reordered, &spec).unwrap();
        assert_ne!(
            first.canonical_digest(&spec),
            reordered.canonical_digest(&spec)
        );
    }

    #[test]
    fn zero_q_positions_are_rejected_before_evaluation() {
        let (row, col) = (0..8)
            .flat_map(|row| (0..8).map(move |col| (row, col)))
            .find(|(row, col)| !active(1, *row, *col))
            .unwrap();
        let mut parsed = valid_parsed();
        parsed.coefficients[row][col][1] = 9;
        assert!(matches!(
            PublicCoefficients::from_parsed(parsed, &test_spec()),
            Err(CoefficientError::NonZeroAtZeroQ { .. })
        ));
    }

    #[test]
    fn digest_limbs_are_unambiguous_little_endian_chunks() {
        let digest = std::array::from_fn(|index| index as u8);
        let limbs = coefficient_digest_limbs(&digest);
        assert_eq!(limbs[0], Scalar::from(0x0706_0504_0302_0100u64));
        assert_eq!(limbs[3], Scalar::from(0x1f1e_1d1c_1b1a_1918u64));
        assert_eq!(digest_hex(&digest).len(), 64);
    }

    #[test]
    fn transcript_rebinding_defeats_a_public_vector_forged_for_an_old_point() {
        let spec = test_spec();
        let old_point = Scalar::from(7u64);
        let mut parsed = valid_parsed();
        parsed.coefficients[0][0] = vec![1, -7, 0];
        let forged = PublicCoefficients::from_parsed(parsed, &spec).unwrap();
        assert_eq!(forged.active_evaluation(old_point), Scalar::ZERO);

        let digest = forged.canonical_digest(&spec);
        let rebound = derive_coefficient_challenge(Scalar::from(31u64), &digest, &spec).unwrap();
        assert_ne!(rebound, old_point);
        assert_ne!(forged.active_evaluation(rebound), Scalar::ZERO);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    #[test]
    fn versions_and_position_bounds_are_enforced() {
        let spec = crate::input::ResolutionSpec {
            name: "TEST",
            width: 160,
            height: 16,
            logical_rows: 2,
            padded_logical_rows: 2,
            packed_rows: 16,
            step_count: 1,
            padded_pixels: 0,
        };
        let make = || PublicCoefficientJson {
            format: COEFFICIENT_FORMAT.into(),
            resolution: "TEST".into(),
            coefficients: vec![vec![vec![0; 3]; 160]; 16],
        };
        let mut old = make();
        old.format = "snarkpeg-approx-dctq-coefficients-v1".into();
        assert!(matches!(
            PublicCoefficients::from_parsed(old, &spec),
            Err(CoefficientError::InvalidFormat { .. })
        ));
        for ch in 0..3 {
            for r in 0..8 {
                for c in 0..8 {
                    let (lo, hi) = bounds(ch, r, c);
                    for value in [lo, hi] {
                        let mut v = make();
                        v.coefficients[r][c][ch] = value;
                        assert!(PublicCoefficients::from_parsed(v, &spec).is_ok());
                    }
                    for value in [lo - 1, hi + 1] {
                        let mut v = make();
                        v.coefficients[r][c][ch] = value;
                        assert!(PublicCoefficients::from_parsed(v, &spec).is_err());
                    }
                }
            }
        }
    }
}
