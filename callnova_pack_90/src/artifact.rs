use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RecursiveProofArtifact<P> {
    pub backend: String,
    pub proof_kind: String,
    pub function: String,
    pub resolution: String,
    pub num_steps: usize,
    pub start_public_input: Vec<String>,
    pub final_outputs: Vec<String>,
    pub proof: P,
}
