use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};
use thiserror::Error;

pub const DCTQ_HD_WIDTH: usize = 160;
pub const DCTQ_STEP_ROWS: usize = 64;
pub const PIXELS_PER_STEP: usize = DCTQ_HD_WIDTH * DCTQ_STEP_ROWS;

pub type Pixel = [u8; 3];
pub type DctqRow = [Pixel; DCTQ_HD_WIDTH];
pub type DctqStep = [DctqRow; DCTQ_STEP_ROWS];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolutionSpec {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub logical_rows: usize,
    pub padded_logical_rows: usize,
    pub packed_rows: usize,
    pub step_count: usize,
    pub padded_pixels: usize,
}

pub const SD_SPEC: ResolutionSpec = ResolutionSpec {
    name: "SD",
    width: 640,
    height: 480,
    logical_rows: 240,
    padded_logical_rows: 240,
    packed_rows: 1920,
    step_count: 30,
    padded_pixels: 0,
};

pub const HD_SPEC: ResolutionSpec = ResolutionSpec {
    name: "HD",
    width: 1280,
    height: 720,
    logical_rows: 720,
    padded_logical_rows: 720,
    packed_rows: 5760,
    step_count: 90,
    padded_pixels: 0,
};

pub const FHD_SPEC: ResolutionSpec = ResolutionSpec {
    name: "FHD",
    width: 1920,
    height: 1080,
    logical_rows: 1620,
    padded_logical_rows: 1624,
    packed_rows: 12992,
    step_count: 203,
    padded_pixels: 5120,
};

pub const QHD_SPEC: ResolutionSpec = ResolutionSpec {
    name: "QHD",
    width: 2560,
    height: 1440,
    logical_rows: 2880,
    padded_logical_rows: 2880,
    packed_rows: 23040,
    step_count: 360,
    padded_pixels: 0,
};

pub const K4_SPEC: ResolutionSpec = ResolutionSpec {
    name: "4K",
    width: 3840,
    height: 2160,
    logical_rows: 6480,
    padded_logical_rows: 6480,
    packed_rows: 51840,
    step_count: 810,
    padded_pixels: 0,
};

pub fn resolution_spec(name: &str) -> Option<&'static ResolutionSpec> {
    match name {
        "SD" => Some(&SD_SPEC),
        "HD" => Some(&HD_SPEC),
        "FHD" => Some(&FHD_SPEC),
        "QHD" => Some(&QHD_SPEC),
        "4K" => Some(&K4_SPEC),
        _ => None,
    }
}

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
    #[error("invalid {resolution} input: expected {expected} packed rows, got {actual}")]
    InvalidRowCount {
        resolution: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "invalid {resolution} input: expected row {row_index} to have {expected} packed pixels, got {actual}"
    )]
    InvalidRowWidth {
        resolution: &'static str,
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
    pub fn load(path: &Path, spec: &ResolutionSpec) -> Result<Self, InputError> {
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

        if parsed.original.len() != spec.packed_rows {
            return Err(InputError::InvalidRowCount {
                resolution: spec.name,
                expected: spec.packed_rows,
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
                        resolution: spec.name,
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
                    .expect("validated input always splits into 64-row steps")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_specs_match_expected_counts() {
        let expected = [
            ("SD", 640, 480, 240, 240, 1920, 30, 0),
            ("HD", 1280, 720, 720, 720, 5760, 90, 0),
            ("FHD", 1920, 1080, 1620, 1624, 12992, 203, 5120),
            ("QHD", 2560, 1440, 2880, 2880, 23040, 360, 0),
            ("4K", 3840, 2160, 6480, 6480, 51840, 810, 0),
        ];

        for (
            name,
            width,
            height,
            logical_rows,
            padded_logical_rows,
            packed_rows,
            steps,
            padded_pixels,
        ) in expected
        {
            let spec = resolution_spec(name).expect("spec exists");
            assert_eq!(spec.width, width);
            assert_eq!(spec.height, height);
            assert_eq!(spec.logical_rows, logical_rows);
            assert_eq!(spec.padded_logical_rows, padded_logical_rows);
            assert_eq!(spec.packed_rows, packed_rows);
            assert_eq!(spec.step_count, steps);
            assert_eq!(spec.padded_pixels, padded_pixels);
            assert_eq!(spec.packed_rows, spec.padded_logical_rows * 8);
            assert_eq!(spec.step_count, spec.packed_rows / DCTQ_STEP_ROWS);
            assert_eq!(
                spec.step_count * PIXELS_PER_STEP,
                (spec.width * spec.height) + spec.padded_pixels
            );
        }
    }

    #[test]
    fn fhd_padding_is_exactly_5120_pixels() {
        assert_eq!(FHD_SPEC.padded_pixels, 5120);
        assert_eq!(FHD_SPEC.step_count, 203);
    }

    #[test]
    fn four_k_uses_exact_fixed_pixel_schedule() {
        assert_eq!(K4_SPEC.step_count, 810);
        assert_eq!(K4_SPEC.padded_pixels, 0);
        assert_eq!(K4_SPEC.width * K4_SPEC.height, K4_SPEC.step_count * PIXELS_PER_STEP);
    }
}
