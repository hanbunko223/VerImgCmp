use crate::{
    coefficient::{coefficient_digest_limbs, ACTIVE_COEFFICIENTS_PER_STEP},
    hash::Scalar,
    input::{resolution_tag, ResolutionSpec},
    poseidon::{poseidon_hash_8_with_domain, HH_CHALLENGE_DOMAIN_SEPARATOR},
};
use ff::PrimeField;
use num_bigint::BigUint;
use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};
use thiserror::Error;

pub const INPUT_DIGEST_FORMAT: &str = "speg-poseidon-hh-input-digest-v1";
pub const CHALLENGE_SCHEME: &str = "poseidon-97-centered-dct128-recip1024-evaluation-v1";

#[derive(Debug, Error)]
pub enum StatementError {
    #[error("failed to open public input-digest file {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read public input-digest file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse public input-digest JSON {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported input-digest format {actual}; expected {expected}")]
    InvalidFormat {
        expected: &'static str,
        actual: String,
    },
    #[error("input-digest resolution {actual} does not match requested resolution {expected}")]
    InvalidResolution {
        expected: &'static str,
        actual: String,
    },
    #[error("input digest is not a canonical decimal field element: {0}")]
    InvalidDigest(String),
}

#[derive(Debug, Deserialize)]
struct InputDigestJson {
    format: String,
    resolution: String,
    digest: String,
}

#[derive(Clone, Debug)]
pub struct PublicInputDigest {
    pub value: Scalar,
}

impl PublicInputDigest {
    pub fn load(path: &Path, spec: &ResolutionSpec) -> Result<Self, StatementError> {
        let mut file = File::open(path).map_err(|source| StatementError::Open {
            path: path.display().to_string(),
            source,
        })?;
        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|source| StatementError::Read {
                path: path.display().to_string(),
                source,
            })?;
        let parsed: InputDigestJson =
            serde_json::from_str(&json).map_err(|source| StatementError::Parse {
                path: path.display().to_string(),
                source,
            })?;
        if parsed.format != INPUT_DIGEST_FORMAT {
            return Err(StatementError::InvalidFormat {
                expected: INPUT_DIGEST_FORMAT,
                actual: parsed.format,
            });
        }
        if parsed.resolution != spec.name {
            return Err(StatementError::InvalidResolution {
                expected: spec.name,
                actual: parsed.resolution,
            });
        }
        Ok(Self {
            value: parse_canonical_scalar(&parsed.digest)?,
        })
    }
}

pub fn derive_coefficient_challenge(
    input_digest: Scalar,
    coefficient_digest: &[u8; 32],
    spec: &ResolutionSpec,
) -> Result<Scalar, StatementError> {
    let digest_limbs = coefficient_digest_limbs(coefficient_digest);
    let active_count = spec
        .step_count
        .checked_mul(ACTIVE_COEFFICIENTS_PER_STEP)
        .expect("supported resolutions fit usize");
    let transcript = [
        input_digest,
        digest_limbs[0],
        digest_limbs[1],
        digest_limbs[2],
        digest_limbs[3],
        Scalar::from(resolution_tag(spec)),
        Scalar::from(active_count as u64),
        Scalar::from(spec.step_count as u64),
    ];
    Ok(poseidon_hash_8_with_domain(
        &transcript,
        HH_CHALLENGE_DOMAIN_SEPARATOR,
    ))
}

pub(crate) fn parse_canonical_scalar(text: &str) -> Result<Scalar, StatementError> {
    if text.is_empty()
        || !text.bytes().all(|byte| byte.is_ascii_digit())
        || (text.len() > 1 && text.starts_with('0'))
    {
        return Err(StatementError::InvalidDigest(text.to_string()));
    }
    let value = BigUint::parse_bytes(text.as_bytes(), 10)
        .ok_or_else(|| StatementError::InvalidDigest(text.to_string()))?;
    let modulus = if let Some(hexadecimal) = Scalar::MODULUS.strip_prefix("0x") {
        BigUint::parse_bytes(hexadecimal.as_bytes(), 16)
    } else {
        BigUint::parse_bytes(Scalar::MODULUS.as_bytes(), 10)
    }
    .expect("PrimeField modulus is valid");
    if value >= modulus {
        return Err(StatementError::InvalidDigest(text.to_string()));
    }
    let bytes = value.to_bytes_le();
    let mut repr = <Scalar as PrimeField>::Repr::default();
    if bytes.len() > repr.as_ref().len() {
        return Err(StatementError::InvalidDigest(text.to_string()));
    }
    repr.as_mut()[..bytes.len()].copy_from_slice(&bytes);
    Option::<Scalar>::from(Scalar::from_repr(repr))
        .ok_or_else(|| StatementError::InvalidDigest(text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hash::scalar_to_decimal_string,
        input::{HD_SPEC, SD_SPEC},
    };
    use ff::Field;

    #[test]
    fn canonical_scalar_parser_rejects_malleable_encodings() {
        assert_eq!(parse_canonical_scalar("0").unwrap(), Scalar::ZERO);
        assert_eq!(parse_canonical_scalar("19").unwrap(), Scalar::from(19u64));
        for invalid in ["", "00", "019", "-1", "+1", "1 ", " 1", "1.0"] {
            assert!(
                parse_canonical_scalar(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
        assert!(parse_canonical_scalar(Scalar::MODULUS).is_err());
        assert_eq!(
            parse_canonical_scalar(&scalar_to_decimal_string(&Scalar::from(42u64))).unwrap(),
            Scalar::from(42u64)
        );
    }

    #[test]
    fn challenge_binds_every_public_transcript_component() {
        let digest = [7u8; 32];
        let challenge =
            derive_coefficient_challenge(Scalar::from(11u64), &digest, &HD_SPEC).unwrap();
        let mut changed_digest = digest;
        changed_digest[31] ^= 1;
        assert_ne!(
            challenge,
            derive_coefficient_challenge(Scalar::from(11u64), &changed_digest, &HD_SPEC).unwrap()
        );
        assert_ne!(
            challenge,
            derive_coefficient_challenge(Scalar::from(12u64), &digest, &HD_SPEC).unwrap()
        );
        assert_ne!(
            challenge,
            derive_coefficient_challenge(Scalar::from(11u64), &digest, &SD_SPEC).unwrap()
        );

        let mut different_steps = HD_SPEC;
        different_steps.step_count += 1;
        assert_ne!(
            challenge,
            derive_coefficient_challenge(Scalar::from(11u64), &digest, &different_steps).unwrap()
        );
    }

    #[test]
    fn rebinding_coefficients_invalidates_a_fingerprint_engineered_for_an_old_point() {
        let old_point = Scalar::from(7u64);
        let forged_coefficients = [Scalar::ONE, -old_point];
        let old_evaluation = forged_coefficients
            .iter()
            .fold(Scalar::ZERO, |acc, coefficient| {
                acc * old_point + coefficient
            });
        assert_eq!(old_evaluation, Scalar::ZERO);

        let digest = [91u8; 32];
        let rebound_point =
            derive_coefficient_challenge(Scalar::from(19u64), &digest, &HD_SPEC).unwrap();
        assert_ne!(rebound_point, old_point);
        let rebound_evaluation = forged_coefficients
            .iter()
            .fold(Scalar::ZERO, |acc, coefficient| {
                acc * rebound_point + coefficient
            });
        assert_ne!(rebound_evaluation, Scalar::ZERO);
    }
}
