use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};
use thiserror::Error;

pub const DCTQ_HD_ROWS: usize = 5760;
pub const DCTQ_HD_WIDTH: usize = 160;
pub const DCTQ_STEP_ROWS: usize = 8;
pub const DCTQ_HD_STEP_COUNT: usize = DCTQ_HD_ROWS / DCTQ_STEP_ROWS;

pub type Pixel = [u8; 3];
pub type DctqRow = [Pixel; DCTQ_HD_WIDTH];
pub type DctqStep = [DctqRow; DCTQ_STEP_ROWS];

#[derive(Debug, Error)]
pub enum InputError {
    #[error("failed to open input file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read input file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse input JSON {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid dctq input: expected {expected} rows, got {actual}")]
    InvalidRowCount { expected: usize, actual: usize },
    #[error("invalid dctq input: expected row {row_index} to have {expected} pixels, got {actual}")]
    InvalidRowWidth {
        row_index: usize,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Deserialize)]
struct DctqInputJson {
    original: Vec<Vec<Pixel>>,
}

#[derive(Clone, Debug)]
pub struct DctqInput {
    rows: Vec<DctqRow>,
}

impl DctqInput {
    pub fn load(path: &Path) -> Result<Self, InputError> {
        let mut file = File::open(path).map_err(|source| InputError::Open {
            path: path.display().to_string(),
            source,
        })?;
        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|source| InputError::Read {
                path: path.display().to_string(),
                source,
            })?;
        let parsed: DctqInputJson =
            serde_json::from_str(&json).map_err(|source| InputError::Parse {
                path: path.display().to_string(),
                source,
            })?;

        if parsed.original.len() != DCTQ_HD_ROWS {
            return Err(InputError::InvalidRowCount {
                expected: DCTQ_HD_ROWS,
                actual: parsed.original.len(),
            });
        }

        let rows = parsed
            .original
            .into_iter()
            .enumerate()
            .map(|(row_index, row)| {
                if row.len() != DCTQ_HD_WIDTH {
                    return Err(InputError::InvalidRowWidth {
                        row_index,
                        expected: DCTQ_HD_WIDTH,
                        actual: row.len(),
                    });
                }
                Ok(row
                    .try_into()
                    .expect("validated row width matches the fixed row type"))
            })
            .collect::<Result<Vec<DctqRow>, InputError>>()?;

        Ok(Self { rows })
    }

    pub fn into_steps(self) -> Vec<DctqStep> {
        self.rows
            .chunks_exact(DCTQ_STEP_ROWS)
            .map(|rows| {
                rows.try_into()
                    .expect("validated HD input always splits into 8-row steps")
            })
            .collect()
    }
}
