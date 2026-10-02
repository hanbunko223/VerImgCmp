use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RecursiveProofArtifact<P> {
    pub artifact_version: String,
    pub backend: String,
    pub proof_kind: String,
    pub function: String,
    pub resolution: String,
    pub num_steps: usize,
    pub input_digest_path: String,
    pub input_digest_format: String,
    pub public_input_digest: String,
    pub coefficient_path: String,
    pub coefficient_format: String,
    pub public_coefficients_sha256: String,
    pub challenge_scheme: String,
    pub coefficient_challenge: String,
    pub public_coefficient_evaluation: String,
    pub start_public_input: Vec<String>,
    pub final_outputs: Vec<String>,
    pub proof: P,
}
