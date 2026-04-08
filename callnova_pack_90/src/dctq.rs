use crate::{
    input::{DCTQ_HD_WIDTH, DCTQ_STEP_ROWS, DctqStep},
};
use ff::PrimeField;
use nova_snark::{provider::PallasEngine, traits::Engine};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::OnceLock,
};
use thiserror::Error;

pub type Scalar = <PallasEngine as Engine>::Scalar;

pub const DCTQ_BLOCK_SIZE: usize = 8;
pub const DCTQ_CHANNELS: usize = 3;
pub const DCTQ_BLOCKS_PER_STEP: usize = DCTQ_HD_WIDTH / DCTQ_BLOCK_SIZE;

#[cfg_attr(not(test), allow(dead_code))]
pub type ChannelPlane = Vec<[Scalar; DCTQ_HD_WIDTH]>;
pub type BlockMatrix = [[Scalar; DCTQ_BLOCK_SIZE]; DCTQ_BLOCK_SIZE];

static DCTQ_MATRICES: OnceLock<Result<DctqMatrices, DctqError>> = OnceLock::new();

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct DctqMatrices {
    d_signed: BlockIntMatrix,
    q_signed: BlockIntMatrix,
    d_field: BlockMatrix,
    q_field: BlockMatrix,
}

type BlockIntMatrix = [[i64; DCTQ_BLOCK_SIZE]; DCTQ_BLOCK_SIZE];

#[derive(Debug, Error, Clone)]
pub enum DctqError {
    #[error("failed to open DCTQ matrix file {path}: {message}")]
    Open {
        path: String,
        message: String,
    },
    #[error("failed to read DCTQ matrix file {path}: {message}")]
    Read {
        path: String,
        message: String,
    },
    #[error("invalid DCTQ matrix row count in {path}: expected {expected}, got {actual}")]
    InvalidRowCount {
        path: String,
        expected: usize,
        actual: usize,
    },
    #[error("invalid DCTQ matrix row width in {path} at row {row_index}: expected {expected}, got {actual}")]
    InvalidRowWidth {
        path: String,
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("failed to parse DCTQ matrix entry in {path} at row {row_index}, col {col_index}: {value}")]
    ParseEntry {
        path: String,
        row_index: usize,
        col_index: usize,
        value: String,
    },
    #[error("failed to convert signed DCTQ value {value} into field scalar")]
    ScalarConversion { value: i128 },
}

pub fn dctq_matrices() -> Result<&'static DctqMatrices, DctqError> {
    DCTQ_MATRICES
        .get_or_init(|| {
            let d_signed = load_matrix_file(&repo_matrix_path("DCTmat.txt"))?;
            let q_signed = load_matrix_file(&repo_matrix_path("Qmat.txt"))?;
            Ok(DctqMatrices {
                d_field: int_matrix_to_field(&d_signed)?,
                q_field: int_matrix_to_field(&q_signed)?,
                d_signed,
                q_signed,
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn split_step_channels(step: &DctqStep) -> [ChannelPlane; DCTQ_CHANNELS] {
    std::array::from_fn(|channel_idx| {
        step.iter()
            .map(|row| {
                std::array::from_fn(|col_idx| Scalar::from(u64::from(row[col_idx][channel_idx])))
            })
            .collect()
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn dct_left_plane(plane: &ChannelPlane) -> Result<ChannelPlane, DctqError> {
    let d = &dctq_matrices()?.d_signed;
    Ok((0..DCTQ_STEP_ROWS)
        .map(|row_idx| {
            std::array::from_fn(|col_idx| {
                let block_row = (row_idx / DCTQ_BLOCK_SIZE) * DCTQ_BLOCK_SIZE;
                let out_r = row_idx % DCTQ_BLOCK_SIZE;
                let block_col = (col_idx / DCTQ_BLOCK_SIZE) * DCTQ_BLOCK_SIZE;
                let local_col = col_idx % DCTQ_BLOCK_SIZE;
                let sum = (0..DCTQ_BLOCK_SIZE).fold(0i128, |acc, k| {
                    let input = scalar_to_i128(plane[block_row + k][block_col + local_col]);
                    acc + input * i128::from(d[out_r][k])
                });
                i128_to_scalar(sum).expect("DCT left output fits in the field")
            })
        })
        .collect())
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn dct_right_plane(left: &ChannelPlane) -> Result<ChannelPlane, DctqError> {
    let d = &dctq_matrices()?.d_signed;
    Ok((0..DCTQ_STEP_ROWS)
        .map(|row_idx| {
            std::array::from_fn(|col_idx| {
                let block_col = (col_idx / DCTQ_BLOCK_SIZE) * DCTQ_BLOCK_SIZE;
                let out_c = col_idx % DCTQ_BLOCK_SIZE;
                let sum = (0..DCTQ_BLOCK_SIZE).fold(0i128, |acc, k| {
                    let input = scalar_to_i128(left[row_idx][block_col + k]);
                    acc + input * i128::from(d[k][out_c])
                });
                i128_to_scalar(sum).expect("DCT right output fits in the field")
            })
        })
        .collect())
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn hadamard_q_plane(right: &ChannelPlane) -> Result<ChannelPlane, DctqError> {
    let q = &dctq_matrices()?.q_signed;
    Ok((0..DCTQ_STEP_ROWS)
        .map(|row_idx| {
            std::array::from_fn(|col_idx| {
                let qr = row_idx % DCTQ_BLOCK_SIZE;
                let qc = col_idx % DCTQ_BLOCK_SIZE;
                let value = scalar_to_i128(right[row_idx][col_idx]) * i128::from(q[qr][qc]);
                i128_to_scalar(value).expect("Hadamard output fits in the field")
            })
        })
        .collect())
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn compute_dctq_planes(step: &DctqStep) -> Result<[ChannelPlane; DCTQ_CHANNELS], DctqError> {
    let channels = split_step_channels(step);
    let planes = (0..DCTQ_CHANNELS)
        .map(|channel_idx| {
        let left = dct_left_plane(&channels[channel_idx])?;
        let right = dct_right_plane(&left)?;
        hadamard_q_plane(&right)
    })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(planes.try_into().expect("fixed channel count"))
}

pub fn d_matrix() -> Result<&'static BlockMatrix, DctqError> {
    Ok(&dctq_matrices()?.d_field)
}

pub fn q_matrix() -> Result<&'static BlockMatrix, DctqError> {
    Ok(&dctq_matrices()?.q_field)
}

fn repo_matrix_path(file_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate directory is inside the repo")
        .join(file_name)
}

fn load_matrix_file(path: &Path) -> Result<BlockIntMatrix, DctqError> {
    let file = File::open(path).map_err(|source| DctqError::Open {
        path: path.display().to_string(),
        message: source.to_string(),
    })?;
    let reader = BufReader::new(file);
    let mut rows = Vec::with_capacity(DCTQ_BLOCK_SIZE);

    for line in reader.lines() {
        let line = line.map_err(|source| DctqError::Read {
            path: path.display().to_string(),
            message: source.to_string(),
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        rows.push(trimmed.to_string());
    }

    if rows.len() != DCTQ_BLOCK_SIZE {
        return Err(DctqError::InvalidRowCount {
            path: path.display().to_string(),
            expected: DCTQ_BLOCK_SIZE,
            actual: rows.len(),
        });
    }

    let mut matrix = [[0i64; DCTQ_BLOCK_SIZE]; DCTQ_BLOCK_SIZE];
    for (row_index, row) in rows.iter().enumerate() {
        let entries = row.split_whitespace().collect::<Vec<_>>();
        if entries.len() != DCTQ_BLOCK_SIZE {
            return Err(DctqError::InvalidRowWidth {
                path: path.display().to_string(),
                row_index,
                expected: DCTQ_BLOCK_SIZE,
                actual: entries.len(),
            });
        }
        for (col_index, entry) in entries.iter().enumerate() {
            matrix[row_index][col_index] = entry.parse::<i64>().map_err(|_| DctqError::ParseEntry {
                path: path.display().to_string(),
                row_index,
                col_index,
                value: (*entry).to_string(),
            })?;
        }
    }

    Ok(matrix)
}

fn int_matrix_to_field(matrix: &BlockIntMatrix) -> Result<BlockMatrix, DctqError> {
    let rows = (0..DCTQ_BLOCK_SIZE)
        .map(|row_idx| {
            (0..DCTQ_BLOCK_SIZE)
                .map(|col_idx| i128_to_scalar(i128::from(matrix[row_idx][col_idx])))
                .collect::<Result<Vec<_>, _>>()
                .map(|row| row.try_into().expect("fixed 8x8 row shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows.try_into().expect("fixed 8x8 matrix shape"))
}

fn i128_to_scalar(value: i128) -> Result<Scalar, DctqError> {
    if value >= 0 {
        Ok(Scalar::from(
            u64::try_from(value).map_err(|_| DctqError::ScalarConversion { value })?,
        ))
    } else {
        Ok(-Scalar::from(
            u64::try_from(-value).map_err(|_| DctqError::ScalarConversion { value })?,
        ))
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn scalar_to_i128(value: Scalar) -> i128 {
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

    fn synthetic_step() -> DctqStep {
        std::array::from_fn(|row_idx| {
            std::array::from_fn(|col_idx| {
                [
                    ((row_idx * 17 + col_idx) % 256) as u8,
                    ((row_idx * 17 + col_idx + 3) % 256) as u8,
                    ((row_idx * 17 + col_idx + 9) % 256) as u8,
                ]
            })
        })
    }

    fn manual_block_entry(
        step: &DctqStep,
        channel_idx: usize,
        global_row: usize,
        out_c: usize,
    ) -> i128 {
        let matrices = dctq_matrices().unwrap();
        let block_row = (global_row / DCTQ_BLOCK_SIZE) * DCTQ_BLOCK_SIZE;
        let out_r = global_row % DCTQ_BLOCK_SIZE;
        let mut left_row = [0i128; DCTQ_BLOCK_SIZE];
        for c in 0..DCTQ_BLOCK_SIZE {
            left_row[c] = (0..DCTQ_BLOCK_SIZE).fold(0i128, |acc, k| {
                acc + i128::from(step[block_row + k][c][channel_idx])
                    * i128::from(matrices.d_signed[out_r][k])
            });
        }
        let right = (0..DCTQ_BLOCK_SIZE).fold(0i128, |acc, k| {
            acc + left_row[k] * i128::from(matrices.d_signed[k][out_c])
        });
        right * i128::from(matrices.q_signed[out_r][out_c])
    }

    #[test]
    fn matrices_load_and_cache() {
        let matrices = dctq_matrices().unwrap();
        assert_eq!(matrices.d_signed[0][0], 91);
        assert_eq!(matrices.q_signed[0][0], 39);
        assert_eq!(dctq_matrices().unwrap().d_signed, matrices.d_signed);
    }

    #[test]
    fn split_step_channels_is_deterministic() {
        let step = synthetic_step();
        let channels = split_step_channels(&step);
        assert_eq!(channels[0][0][0], Scalar::from(u64::from(step[0][0][0])));
        assert_eq!(
            channels[2][DCTQ_STEP_ROWS - 1][159],
            Scalar::from(u64::from(step[DCTQ_STEP_ROWS - 1][159][2]))
        );
    }

    #[test]
    fn dctq_host_pipeline_matches_manual_entry() {
        let step = synthetic_step();
        let channels = split_step_channels(&step);
        let left = dct_left_plane(&channels[0]).unwrap();
        let right = dct_right_plane(&left).unwrap();
        let final_plane = hadamard_q_plane(&right).unwrap();

        let expected = manual_block_entry(&step, 0, 2, 5);
        assert_eq!(final_plane[2][5], i128_to_scalar(expected).unwrap());
    }

    #[test]
    fn dctq_host_pipeline_matches_manual_entry_in_later_vertical_slab() {
        let step = synthetic_step();
        let channels = split_step_channels(&step);
        let left = dct_left_plane(&channels[1]).unwrap();
        let right = dct_right_plane(&left).unwrap();
        let final_plane = hadamard_q_plane(&right).unwrap();

        let target_row = DCTQ_BLOCK_SIZE + 3;
        let expected = manual_block_entry(&step, 1, target_row, 6);
        assert_eq!(final_plane[target_row][6], i128_to_scalar(expected).unwrap());
    }

    #[test]
    fn dctq_host_pipeline_is_deterministic() {
        let step = synthetic_step();
        assert_eq!(compute_dctq_planes(&step).unwrap(), compute_dctq_planes(&step).unwrap());
    }

    #[test]
    fn dctq_host_pipeline_changes_when_input_changes() {
        let mut step = synthetic_step();
        let before = compute_dctq_planes(&step).unwrap();
        step[0][0] = [step[0][0][0].wrapping_add(1), step[0][0][1], step[0][0][2]];
        let after = compute_dctq_planes(&step).unwrap();
        assert_ne!(before, after);
    }
}
