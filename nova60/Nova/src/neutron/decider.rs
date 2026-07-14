//! Spartan decider for NeutronNova's final folded state.
#![allow(non_snake_case)]

use crate::{
  constants::NUM_HASH_BITS,
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  neutron::{
    relation::{FoldedInstance, FoldedWitness, Structure},
    PublicParams, RecursiveSNARK,
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix},
  spartan::{
    compute_eval_table_sparse,
    math::Math,
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, multilinear::SparsePolynomial},
    snark::{batch_eval_reduce, batch_eval_verify},
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    commitment::CommitmentEngineTrait,
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    AbsorbInRO2Trait, Engine, RO2Constants, ROTrait, TranscriptEngineTrait,
  },
  CommitmentKey, DerandKey,
};
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Prover key for the Neutron folded-relation Spartan proof.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  vk_digest: E::Scalar,
}

/// Verifier key for the Neutron folded-relation Spartan proof.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedVerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  vk_ee: EE::VerifierKey,
  structure: Structure<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for FoldedVerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> FoldedVerifierKey<E, EE> {
  fn new(structure: Structure<E>, vk_ee: EE::VerifierKey) -> Self {
    Self {
      vk_ee,
      structure,
      digest: OnceCell::new(),
    }
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for FoldedVerifierKey<E, EE> {
  fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure to retrieve digest")
  }
}

/// Spartan proof for Neutron's folded residual relation.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FoldedSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar, E::Scalar),
  eval_E_left: E::Scalar,
  eval_E_right: E::Scalar,
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  sc_proof_batch: SumcheckProof<E>,
  evals_batch: Vec<E::Scalar>,
  eval_arg: EE::EvaluationArgument,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> FoldedSNARK<E, EE> {
  /// Create proving and verifying keys for the folded Neutron relation.
  pub fn setup(
    ck: &CommitmentKey<E>,
    structure: &Structure<E>,
  ) -> Result<(FoldedProverKey<E, EE>, FoldedVerifierKey<E, EE>), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck)?;
    let vk = FoldedVerifierKey::new(structure.clone(), vk_ee);
    let pk = FoldedProverKey {
      pk_ee,
      vk_digest: vk.digest(),
    };
    Ok((pk, vk))
  }

  /// Prove satisfiability of a derandomized Neutron folded instance.
  pub fn prove(
    ck: &CommitmentKey<E>,
    pk: &FoldedProverKey<E, EE>,
    structure: &Structure<E>,
    U: &FoldedInstance<E>,
    W: &FoldedWitness<E>,
  ) -> Result<Self, NovaError> {
    assert_eq!(W.W.len(), structure.S.num_vars);
    assert_eq!(W.E.len(), structure.left + structure.right);
    assert_eq!(U.X.len(), structure.S.num_io);

    let mut transcript = E::TE::new(b"NeutronFoldedSNARK");
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();
    let (Az, Bz, Cz) = structure.S.multiply_vec(&z)?;
    let full_E = full_tensor_E::<E>(structure, &W.E);

    let (sc_proof_outer, r_x, claims_outer_vec) = prove_weighted_residual_sumcheck::<E>(
      &U.T,
      structure.ell,
      &mut MultilinearPolynomial::new(full_E),
      &mut MultilinearPolynomial::new(Az),
      &mut MultilinearPolynomial::new(Bz),
      &mut MultilinearPolynomial::new(Cz),
      &mut transcript,
    )?;
    let claims_outer = (
      claims_outer_vec[0],
      claims_outer_vec[1],
      claims_outer_vec[2],
      claims_outer_vec[3],
    );

    let (E_poly, E_left_point, E_right_point) =
      E_eval_polynomial_and_points::<E>(structure, &W.E, &r_x);
    let eval_E_left = MultilinearPolynomial::evaluate_with(&E_poly, &E_left_point);
    let eval_E_right = MultilinearPolynomial::evaluate_with(&E_poly, &E_right_point);
    if claims_outer.0 != eval_E_left * eval_E_right {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"claims_outer",
      &[
        claims_outer.0,
        claims_outer.1,
        claims_outer.2,
        claims_outer.3,
        eval_E_left,
        eval_E_right,
      ]
      .as_slice(),
    );

    let beta = transcript.squeeze(b"r")?;
    let claim_inner_joint = claims_outer.1 + beta * claims_outer.2 + beta * beta * claims_outer.3;
    let poly_ABC = {
      let evals_rx = EqPolynomial::evals_from_points(&r_x);
      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&structure.S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());

      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + beta * evals_B[i] + beta * beta * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };

    let mut poly_z = z;
    poly_z.resize(structure.S.num_vars * 2, E::Scalar::ZERO);
    let (sc_proof_inner, r_y, _) = SumcheckProof::prove_quad_prod(
      &claim_inner_joint,
      structure.S.num_vars.log_2() + 1,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      &mut transcript,
    )?;

    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);
    let opening_rounds = structure.S.num_vars.log_2();
    let (W_open_poly, W_open_point) = pad_opening::<E>(&W.W, &r_y[1..], opening_rounds);
    let (E_left_open_poly, E_left_open_point) =
      pad_opening::<E>(&E_poly, &E_left_point, opening_rounds);
    let (E_right_open_poly, E_right_open_point) =
      pad_opening::<E>(&E_poly, &E_right_point, opening_rounds);

    let (batched_u, batched_w, sc_proof_batch, evals_batch) = batch_eval_reduce(
      vec![
        PolyEvalInstance::new(U.comm_W, W_open_point, eval_W),
        PolyEvalInstance::new(U.comm_E, E_left_open_point, eval_E_left),
        PolyEvalInstance::new(U.comm_E, E_right_open_point, eval_E_right),
      ],
      vec![
        PolyEvalWitness::new(W_open_poly),
        PolyEvalWitness::new(E_left_open_poly),
        PolyEvalWitness::new(E_right_open_poly),
      ],
      &mut transcript,
    )?;

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      batched_u.c(),
      batched_w.p(),
      batched_u.x(),
      &batched_u.e(),
    )?;

    Ok(Self {
      sc_proof_outer,
      claims_outer,
      eval_E_left,
      eval_E_right,
      sc_proof_inner,
      eval_W,
      sc_proof_batch,
      evals_batch,
      eval_arg,
    })
  }

  /// Verify a folded Neutron relation proof.
  pub fn verify(
    &self,
    vk: &FoldedVerifierKey<E, EE>,
    U: &FoldedInstance<E>,
  ) -> Result<(), NovaError> {
    let structure = &vk.structure;
    let mut transcript = E::TE::new(b"NeutronFoldedSNARK");
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(U.T, structure.ell, 3, &mut transcript)?;
    let (eval_E_full, eval_Az, eval_Bz, eval_Cz) = self.claims_outer;
    let claim_outer_final_expected = eval_E_full * (eval_Az * eval_Bz - eval_Cz);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }
    if eval_E_full != self.eval_E_left * self.eval_E_right {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let (_E_poly, E_left_point, E_right_point) =
      E_eval_polynomial_and_points::<E>(structure, &[], &r_x);

    transcript.absorb(
      b"claims_outer",
      &[
        eval_E_full,
        eval_Az,
        eval_Bz,
        eval_Cz,
        self.eval_E_left,
        self.eval_E_right,
      ]
      .as_slice(),
    );

    let beta = transcript.squeeze(b"r")?;
    let claim_inner_joint = eval_Az + beta * eval_Bz + beta * beta * eval_Cz;
    let (claim_inner_final, r_y) = self.sc_proof_inner.verify(
      claim_inner_joint,
      structure.S.num_vars.log_2() + 1,
      2,
      &mut transcript,
    )?;

    let eval_Z = {
      let X = vec![U.u]
        .into_iter()
        .chain(U.X.iter().cloned())
        .collect::<Vec<E::Scalar>>();
      let eval_X = SparsePolynomial::new(structure.S.num_vars.log_2(), X).evaluate(&r_y[1..]);
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    let evals = multi_evaluate_matrices::<E>(
      &[&structure.S.A, &structure.S.B, &structure.S.C],
      &r_x,
      &r_y,
    );
    let claim_inner_final_expected = (evals[0] + beta * evals[1] + beta * beta * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let opening_rounds = structure.S.num_vars.log_2();
    let W_open_point = pad_point::<E>(&r_y[1..], opening_rounds);
    let E_left_open_point = pad_point::<E>(&E_left_point, opening_rounds);
    let E_right_open_point = pad_point::<E>(&E_right_point, opening_rounds);

    let batched_u = batch_eval_verify(
      vec![
        PolyEvalInstance::new(U.comm_W, W_open_point, self.eval_W),
        PolyEvalInstance::new(U.comm_E, E_left_open_point, self.eval_E_left),
        PolyEvalInstance::new(U.comm_E, E_right_open_point, self.eval_E_right),
      ],
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      batched_u.c(),
      batched_u.x(),
      &batched_u.e(),
      &self.eval_arg,
    )?;

    Ok(())
  }
}

/// Prover key for the Neutron-native compressed proof.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E1, E2, C, EE, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: crate::traits::circuit::StepCircuit<E1::Scalar>,
  EE: EvaluationEngineTrait<E1>,
  S: RelaxedR1CSSNARKTrait<E1>,
{
  folded_pk: FoldedProverKey<E1, EE>,
  local_pk: S::ProverKey,
  _p: PhantomData<(E2, C)>,
}

/// Verifier key for the Neutron-native compressed proof.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E1, E2, C, EE, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: crate::traits::circuit::StepCircuit<E1::Scalar>,
  EE: EvaluationEngineTrait<E1>,
  S: RelaxedR1CSSNARKTrait<E1>,
{
  F_arity: usize,
  ro_consts: RO2Constants<E1>,
  pp_digest: E1::Scalar,
  folded_vk: FoldedVerifierKey<E1, EE>,
  local_vk: S::VerifierKey,
  dk: DerandKey<E1>,
  _p: PhantomData<(E2, C)>,
}

/// A Neutron-native compressed proof: Spartan for the folded Neutron relation
/// plus Spartan for the latest local augmented R1CS instance.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E1, E2, C, EE, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: crate::traits::circuit::StepCircuit<E1::Scalar>,
  EE: EvaluationEngineTrait<E1>,
  S: RelaxedR1CSSNARKTrait<E1>,
{
  r_U: FoldedInstance<E1>,
  l_u: R1CSInstance<E1>,
  ri: E1::Scalar,
  folded_blind_r_W: E1::Scalar,
  folded_blind_r_E: E1::Scalar,
  local_blind_r_W: E1::Scalar,
  local_blind_r_E: E1::Scalar,
  folded_snark: FoldedSNARK<E1, EE>,
  local_snark: S,
  zn: Vec<E1::Scalar>,
  _p: PhantomData<(E2, C)>,
}

impl<E1, E2, C, EE, S> CompressedSNARK<E1, E2, C, EE, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: crate::traits::circuit::StepCircuit<E1::Scalar>,
  EE: EvaluationEngineTrait<E1>,
  S: RelaxedR1CSSNARKTrait<E1>,
{
  /// Create keys for Neutron-native compression.
  pub fn setup(
    pp: &PublicParams<E1, E2, C>,
  ) -> Result<(ProverKey<E1, E2, C, EE, S>, VerifierKey<E1, E2, C, EE, S>), NovaError> {
    let (folded_pk, folded_vk) = FoldedSNARK::<E1, EE>::setup(&pp.ck, &pp.structure)?;
    let (local_pk, local_vk) = S::setup(&pp.ck, &pp.structure.S)?;
    let pk = ProverKey {
      folded_pk,
      local_pk,
      _p: PhantomData,
    };
    let vk = VerifierKey {
      F_arity: pp.F_arity,
      ro_consts: pp.ro_consts.clone(),
      pp_digest: pp.digest(),
      folded_vk,
      local_vk,
      dk: E1::CE::derand_key(&pp.ck),
      _p: PhantomData,
    };
    Ok((pk, vk))
  }

  /// Create a compressed Neutron proof.
  pub fn prove(
    pp: &PublicParams<E1, E2, C>,
    pk: &ProverKey<E1, E2, C, EE, S>,
    recursive_snark: &RecursiveSNARK<E1, E2, C>,
  ) -> Result<Self, NovaError> {
    let folded_W_derand = FoldedWitness {
      W: recursive_snark.r_W.W.clone(),
      r_W: E1::Scalar::ZERO,
      E: recursive_snark.r_W.E.clone(),
      r_E: E1::Scalar::ZERO,
    };
    let folded_U_derand = FoldedInstance {
      comm_W: E1::CE::derandomize(
        &E1::CE::derand_key(&pp.ck),
        &recursive_snark.r_U.comm_W,
        &recursive_snark.r_W.r_W,
      ),
      comm_E: E1::CE::derandomize(
        &E1::CE::derand_key(&pp.ck),
        &recursive_snark.r_U.comm_E,
        &recursive_snark.r_W.r_E,
      ),
      T: recursive_snark.r_U.T,
      u: recursive_snark.r_U.u,
      X: recursive_snark.r_U.X.clone(),
    };
    let folded_snark = FoldedSNARK::<E1, EE>::prove(
      &pp.ck,
      &pk.folded_pk,
      &pp.structure,
      &folded_U_derand,
      &folded_W_derand,
    )?;

    let local_W = RelaxedR1CSWitness::from_r1cs_witness(&pp.structure.S, &recursive_snark.l_w);
    let local_U =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck, &pp.structure.S, &recursive_snark.l_u);
    let (local_W_derand, local_blind_r_W, local_blind_r_E) = local_W.derandomize();
    let local_U_derand = local_U.derandomize(
      &E1::CE::derand_key(&pp.ck),
      &local_blind_r_W,
      &local_blind_r_E,
    );
    let local_snark = S::prove(
      &pp.ck,
      &pk.local_pk,
      &pp.structure.S,
      &local_U_derand,
      &local_W_derand,
    )?;

    Ok(Self {
      r_U: recursive_snark.r_U.clone(),
      l_u: recursive_snark.l_u.clone(),
      ri: recursive_snark.ri,
      folded_blind_r_W: recursive_snark.r_W.r_W,
      folded_blind_r_E: recursive_snark.r_W.r_E,
      local_blind_r_W,
      local_blind_r_E,
      folded_snark,
      local_snark,
      zn: recursive_snark.zi.clone(),
      _p: PhantomData,
    })
  }

  /// Verify a compressed Neutron proof.
  pub fn verify(
    &self,
    vk: &VerifierKey<E1, E2, C, EE, S>,
    num_steps: usize,
    z0: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    if num_steps == 0 || z0.len() != vk.F_arity || self.l_u.X.len() != 1 || self.r_U.X.len() != 1 {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid number of steps, inputs, or public output width".to_string(),
      });
    }

    let hash = {
      let mut hasher = E1::RO2::new(vk.ro_consts.clone());
      hasher.absorb(vk.pp_digest);
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0 {
        hasher.absorb(*e);
      }
      for e in &self.zn {
        hasher.absorb(*e);
      }
      self.r_U.absorb_in_ro2(&mut hasher);
      hasher.absorb(self.ri);
      hasher.squeeze(NUM_HASH_BITS, false)
    };
    if hash != self.l_u.X[0] {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid output hash in local R1CS instance".to_string(),
      });
    }

    let folded_U_derand = FoldedInstance {
      comm_W: E1::CE::derandomize(&vk.dk, &self.r_U.comm_W, &self.folded_blind_r_W),
      comm_E: E1::CE::derandomize(&vk.dk, &self.r_U.comm_E, &self.folded_blind_r_E),
      T: self.r_U.T,
      u: self.r_U.u,
      X: self.r_U.X.clone(),
    };
    self.folded_snark.verify(&vk.folded_vk, &folded_U_derand)?;

    let local_U = RelaxedR1CSInstance::from_r1cs_instance_unchecked(&self.l_u.comm_W, &self.l_u.X);
    let local_U_derand = local_U.derandomize(&vk.dk, &self.local_blind_r_W, &self.local_blind_r_E);
    self.local_snark.verify(&vk.local_vk, &local_U_derand)?;

    Ok(self.zn.clone())
  }
}

fn full_tensor_E<E: Engine>(structure: &Structure<E>, E_split: &[E::Scalar]) -> Vec<E::Scalar> {
  let (E1, E2) = E_split.split_at(structure.left);
  assert_eq!(E1.len(), structure.left);
  assert_eq!(E2.len(), structure.right);

  (0..structure.right)
    .into_par_iter()
    .flat_map_iter(|i| (0..structure.left).map(move |j| E2[i] * E1[j]))
    .collect()
}

fn E_eval_polynomial_and_points<E: Engine>(
  structure: &Structure<E>,
  E_split: &[E::Scalar],
  r_x: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  let ell_left = structure.left.log_2();
  let ell_right = structure.right.log_2();
  assert_eq!(r_x.len(), ell_left + ell_right);

  let mut E_poly = vec![E::Scalar::ZERO; structure.left * 2];
  if !E_split.is_empty() {
    let (E_left, E_right) = E_split.split_at(structure.left);
    E_poly[..structure.left].copy_from_slice(E_left);
    E_poly[structure.left..structure.left + structure.right].copy_from_slice(E_right);
  }

  let (r_right, r_left) = r_x.split_at(ell_right);
  let mut E_left_point = Vec::with_capacity(1 + ell_left);
  E_left_point.push(E::Scalar::ZERO);
  E_left_point.extend_from_slice(r_left);

  let mut E_right_point = Vec::with_capacity(1 + ell_left);
  E_right_point.push(E::Scalar::ONE);
  E_right_point.extend(std::iter::repeat(E::Scalar::ZERO).take(ell_left - ell_right));
  E_right_point.extend_from_slice(r_right);

  (E_poly, E_left_point, E_right_point)
}

fn pad_point<E: Engine>(point: &[E::Scalar], target_rounds: usize) -> Vec<E::Scalar> {
  assert!(target_rounds >= point.len());
  let mut padded = vec![E::Scalar::ZERO; target_rounds - point.len()];
  padded.extend_from_slice(point);
  padded
}

fn pad_opening<E: Engine>(
  poly: &[E::Scalar],
  point: &[E::Scalar],
  target_rounds: usize,
) -> (Vec<E::Scalar>, Vec<E::Scalar>) {
  let mut padded_poly = poly.to_vec();
  padded_poly.resize(1 << target_rounds, E::Scalar::ZERO);
  (padded_poly, pad_point::<E>(point, target_rounds))
}

fn prove_weighted_residual_sumcheck<E: Engine>(
  claim: &E::Scalar,
  num_rounds: usize,
  poly_E: &mut MultilinearPolynomial<E::Scalar>,
  poly_A: &mut MultilinearPolynomial<E::Scalar>,
  poly_B: &mut MultilinearPolynomial<E::Scalar>,
  poly_C: &mut MultilinearPolynomial<E::Scalar>,
  transcript: &mut E::TE,
) -> Result<(SumcheckProof<E>, Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
  let mut r = Vec::new();
  let mut polys = Vec::new();
  let mut claim_per_round = *claim;

  for _ in 0..num_rounds {
    let (eval_0, leading_coeff, eval_neg_1) =
      compute_eval_points_weighted_residual::<E>(poly_E, poly_A, poly_B, poly_C);
    let evals = vec![eval_0, claim_per_round - eval_0, leading_coeff, eval_neg_1];
    let poly = crate::spartan::polys::univariate::UniPoly::from_evals_deg3(&evals);
    transcript.absorb(b"p", &poly);
    let r_i = transcript.squeeze(b"c")?;
    r.push(r_i);
    polys.push(poly.compress());
    claim_per_round = poly.evaluate(&r_i);

    rayon::join(
      || poly_E.bind_poly_var_top(&r_i),
      || poly_A.bind_poly_var_top(&r_i),
    );
    rayon::join(
      || poly_B.bind_poly_var_top(&r_i),
      || poly_C.bind_poly_var_top(&r_i),
    );
  }

  Ok((
    SumcheckProof::new(polys),
    r,
    vec![poly_E[0], poly_A[0], poly_B[0], poly_C[0]],
  ))
}

fn compute_eval_points_weighted_residual<E: Engine>(
  poly_E: &MultilinearPolynomial<E::Scalar>,
  poly_A: &MultilinearPolynomial<E::Scalar>,
  poly_B: &MultilinearPolynomial<E::Scalar>,
  poly_C: &MultilinearPolynomial<E::Scalar>,
) -> (E::Scalar, E::Scalar, E::Scalar) {
  let len = poly_E.len() / 2;
  (0..len)
    .into_par_iter()
    .map(|i| {
      let e0 = poly_E[i];
      let a0 = poly_A[i];
      let b0 = poly_B[i];
      let c0 = poly_C[i];
      let de = poly_E[len + i] - e0;
      let da = poly_A[len + i] - a0;
      let db = poly_B[len + i] - b0;
      let dc = poly_C[len + i] - c0;

      let eval_0 = e0 * (a0 * b0 - c0);
      let leading_coeff = de * da * db;
      let e_neg = e0 - de;
      let a_neg = a0 - da;
      let b_neg = b0 - db;
      let c_neg = c0 - dc;
      let eval_neg_1 = e_neg * (a_neg * b_neg - c_neg);
      (eval_0, leading_coeff, eval_neg_1)
    })
    .reduce(
      || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
      |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
    )
}

fn multi_evaluate_matrices<E: Engine>(
  M_vec: &[&SparseMatrix<E::Scalar>],
  r_x: &[E::Scalar],
  r_y: &[E::Scalar],
) -> Vec<E::Scalar> {
  let evaluate_with_table =
    |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
      M.indptr
        .par_windows(2)
        .enumerate()
        .map(|(row_idx, ptrs)| {
          M.get_row_unchecked(ptrs.try_into().unwrap())
            .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
            .sum::<E::Scalar>()
        })
        .sum()
    };

  let (T_x, T_y) = rayon::join(
    || EqPolynomial::evals_from_points(r_x),
    || EqPolynomial::evals_from_points(r_y),
  );

  M_vec
    .par_iter()
    .map(|M| evaluate_with_table(M, &T_x, &T_y))
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    provider::{ipa_pc::EvaluationEngine, PallasEngine, VestaEngine},
    spartan::snark::RelaxedR1CSSNARK,
    traits::{
      circuit::StepCircuit,
      snark::{default_ck_hint, RelaxedR1CSSNARKTrait},
    },
  };
  use core::marker::PhantomData;
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for CubicCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      Ok(vec![y])
    }
  }

  #[test]
  fn neutron_spartan_decider_compresses_small_ivc() {
    type E1 = PallasEngine;
    type E2 = VestaEngine;
    type EE = EvaluationEngine<E1>;
    type S = RelaxedR1CSSNARK<E1, EE>;
    type C = CubicCircuit<<E1 as Engine>::Scalar>;
    type Decider = CompressedSNARK<E1, E2, C, EE, S>;

    let circuit = C::default();
    let pp =
      PublicParams::<E1, E2, C>::setup(&circuit, &*S::ck_floor(), &*default_ck_hint()).unwrap();
    let z0 = vec![<E1 as Engine>::Scalar::ONE];
    let num_steps = 3;
    let mut recursive_snark = RecursiveSNARK::<E1, E2, C>::new(&pp, &circuit, &z0).unwrap();
    for _ in 0..num_steps {
      recursive_snark.prove_step(&pp, &circuit).unwrap();
    }
    let recursive_out = recursive_snark.verify(&pp, num_steps, &z0).unwrap();

    let (pk, vk) = Decider::setup(&pp).unwrap();
    let compressed = Decider::prove(&pp, &pk, &recursive_snark).unwrap();
    let compressed_out = compressed.verify(&vk, num_steps, &z0).unwrap();

    assert_eq!(compressed_out, recursive_out);
  }
}
