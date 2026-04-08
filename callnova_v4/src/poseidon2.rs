use ff::{Field, PrimeField};
use num_bigint::BigUint;
use nova_snark::{
    frontend::{
        num::{self, AllocatedNum},
        Boolean, ConstraintSystem, LinearCombination, SynthesisError,
    },
    provider::PallasEngine,
    traits::Engine,
};
use std::sync::OnceLock;

pub type Scalar = <PallasEngine as Engine>::Scalar;

pub const WIDTH: usize = 12;
pub const RATE: usize = 11;
pub const STEP_DIGEST_INPUTS: usize = 3840;
pub const CHAIN_INPUTS: usize = 2;
const ROUNDS_F: usize = 8;
const ROUNDS_P: usize = 57;
const TOTAL_ROUNDS: usize = ROUNDS_F + ROUNDS_P;
const DOMAIN_TAG_STEP3840_U64: u64 = 0x5354_4550_3338_3430;
const DOMAIN_TAG_CHAIN2_U64: u64 = 0x4348_4149_4e32_0001;

static PARAMS: OnceLock<Poseidon2Params> = OnceLock::new();
static FULL_MATRIX: OnceLock<[[Scalar; WIDTH]; WIDTH]> = OnceLock::new();
static SMALL_MATRIX: OnceLock<[[Scalar; 4]; 4]> = OnceLock::new();

#[derive(Clone, Debug)]
struct Poseidon2Params {
    mat_internal_diag_m_1: [Scalar; WIDTH],
    round_constants: [[Scalar; WIDTH]; TOTAL_ROUNDS],
}

#[derive(Clone)]
enum Elt {
    Allocated(AllocatedNum<Scalar>),
    Num(num::Num<Scalar>),
}

impl From<AllocatedNum<Scalar>> for Elt {
    fn from(allocated: AllocatedNum<Scalar>) -> Self {
        Self::Allocated(allocated)
    }
}

impl Elt {
    fn num_from_fr<CS: ConstraintSystem<Scalar>>(fr: Scalar) -> Self {
        let num = num::Num::<Scalar>::zero();
        Self::Num(num.add_bool_with_coeff(CS::one(), &Boolean::Constant(true), fr))
    }

    fn ensure_allocated<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        enforce: bool,
    ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
        match self {
            Self::Allocated(v) => Ok(v.clone()),
            Self::Num(num) => {
                let v = AllocatedNum::alloc(cs.namespace(|| "allocate_elt_num"), || {
                    num.get_value().ok_or(SynthesisError::AssignmentMissing)
                })?;

                if enforce {
                    cs.enforce(
                        || "enforce_elt_num_preserves_lc".to_string(),
                        |_| num.lc(Scalar::ONE),
                        |lc| lc + CS::one(),
                        |lc| lc + v.get_variable(),
                    );
                }
                Ok(v)
            }
        }
    }

    fn val(&self) -> Option<Scalar> {
        match self {
            Self::Allocated(v) => v.get_value(),
            Self::Num(num) => num.get_value(),
        }
    }

    fn lc(&self) -> LinearCombination<Scalar> {
        match self {
            Self::Num(num) => num.lc(Scalar::ONE),
            Self::Allocated(v) => LinearCombination::<Scalar>::zero() + v.get_variable(),
        }
    }

    fn add(self, other: Elt) -> Result<Elt, SynthesisError> {
        match (self, other) {
            (Elt::Num(a), Elt::Num(b)) => Ok(Elt::Num(a.add(&b))),
            (a, b) => Ok(Elt::Num(a.num().add(&b.num()))),
        }
    }

    fn scale<CS: ConstraintSystem<Scalar>>(self, scalar: Scalar) -> Result<Elt, SynthesisError> {
        match self {
            Elt::Num(num) => Ok(Elt::Num(num.scale(scalar))),
            Elt::Allocated(a) => Elt::Num(a.into()).scale::<CS>(scalar),
        }
    }

    fn square<CS: ConstraintSystem<Scalar>>(
        &self,
        mut cs: CS,
    ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
        match self {
            Elt::Num(num) => {
                let allocated = AllocatedNum::alloc(cs.namespace(|| "squared_num"), || {
                    num.get_value()
                        .ok_or(SynthesisError::AssignmentMissing)
                        .map(|tmp| tmp * tmp)
                })?;
                cs.enforce(
                    || "squaring_constraint",
                    |_| num.lc(Scalar::ONE),
                    |_| num.lc(Scalar::ONE),
                    |lc| lc + allocated.get_variable(),
                );
                Ok(allocated)
            }
            Elt::Allocated(a) => a.square(cs),
        }
    }

    fn num(&self) -> num::Num<Scalar> {
        match self {
            Elt::Num(num) => num.clone(),
            Elt::Allocated(a) => a.clone().into(),
        }
    }
}

pub fn poseidon2_compress_3840(inputs: &[Scalar; STEP_DIGEST_INPUTS]) -> Scalar {
    compress_native(inputs.as_slice(), domain_tag_step3840())
}

pub fn poseidon2_compress_3840_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    inputs: &[AllocatedNum<Scalar>; STEP_DIGEST_INPUTS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    compress_allocated(cs, inputs.as_slice(), domain_tag_step3840())
}

pub fn poseidon2_chain_2(inputs: [Scalar; CHAIN_INPUTS]) -> Scalar {
    compress_native(&inputs, domain_tag_chain2())
}

pub fn poseidon2_chain_2_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    inputs: [AllocatedNum<Scalar>; CHAIN_INPUTS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    compress_allocated(cs, &inputs, domain_tag_chain2())
}

fn compress_native(inputs: &[Scalar], domain_tag: Scalar) -> Scalar {
    let params = params();
    let mut state = [Scalar::ZERO; WIDTH];
    state[0] = domain_tag;

    let full_blocks = inputs.len() / RATE;
    let remainder = inputs.len() % RATE;

    for block_idx in 0..full_blocks {
        let start = block_idx * RATE;
        absorb_block(&mut state, &inputs[start..start + RATE]);
        permute(&mut state, params);
    }

    if remainder > 0 {
        absorb_block(&mut state, &inputs[(full_blocks * RATE)..]);
    }
    state[remainder + 1] += Scalar::ONE;
    permute(&mut state, params);

    state[1]
}

fn compress_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    inputs: &[AllocatedNum<Scalar>],
    domain_tag: Scalar,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let mut state = initial_state::<CS>(domain_tag);
    let full_blocks = inputs.len() / RATE;
    let remainder = inputs.len() % RATE;

    for block_idx in 0..full_blocks {
        let start = block_idx * RATE;
        absorb_block_elts::<CS>(&mut state, &inputs[start..start + RATE])?;
        state = permute_allocated(&mut cs.namespace(|| format!("permute_full_{block_idx}")), state)?;
        state = materialize_state(
            &mut cs.namespace(|| format!("materialize_full_{block_idx}")),
            &state,
        )?;
    }

    if remainder > 0 {
        absorb_block_elts::<CS>(&mut state, &inputs[(full_blocks * RATE)..])?;
    }
    state[remainder + 1] = state[remainder + 1]
        .clone()
        .add(Elt::num_from_fr::<CS>(Scalar::ONE))?;
    state = permute_allocated(&mut cs.namespace(|| "permute_final"), state)?;

    state[1].ensure_allocated(&mut cs.namespace(|| "digest_output"), true)
}

fn params() -> &'static Poseidon2Params {
    PARAMS.get_or_init(load_checked_in_params)
}

fn full_matrix() -> &'static [[Scalar; WIDTH]; WIDTH] {
    FULL_MATRIX.get_or_init(|| {
        let small = small_matrix();
        let mut matrix = [[Scalar::ZERO; WIDTH]; WIDTH];
        let blocks = WIDTH / 4;
        for block_row in 0..blocks {
            for block_col in 0..blocks {
                let multiplier = if block_row == block_col {
                    Scalar::from(2u64)
                } else {
                    Scalar::ONE
                };
                for row in 0..4 {
                    for col in 0..4 {
                        matrix[(block_row * 4) + row][(block_col * 4) + col] =
                            small[row][col] * multiplier;
                    }
                }
            }
        }
        matrix
    })
}

fn small_matrix() -> &'static [[Scalar; 4]; 4] {
    SMALL_MATRIX.get_or_init(|| {
        [
            [
                Scalar::from(5u64),
                Scalar::from(7u64),
                Scalar::ONE,
                Scalar::from(3u64),
            ],
            [
                Scalar::from(4u64),
                Scalar::from(6u64),
                Scalar::ONE,
                Scalar::ONE,
            ],
            [
                Scalar::ONE,
                Scalar::from(3u64),
                Scalar::from(5u64),
                Scalar::from(7u64),
            ],
            [
                Scalar::ONE,
                Scalar::ONE,
                Scalar::from(4u64),
                Scalar::from(6u64),
            ],
        ]
    })
}

fn domain_tag_step3840() -> Scalar {
    Scalar::from(DOMAIN_TAG_STEP3840_U64)
}

fn domain_tag_chain2() -> Scalar {
    Scalar::from(DOMAIN_TAG_CHAIN2_U64)
}

fn initial_state<CS: ConstraintSystem<Scalar>>(domain_tag: Scalar) -> Vec<Elt> {
    let mut state = std::iter::repeat_with(|| Elt::num_from_fr::<CS>(Scalar::ZERO))
        .take(WIDTH)
        .collect::<Vec<_>>();
    state[0] = Elt::num_from_fr::<CS>(domain_tag);
    state
}

fn absorb_block(state: &mut [Scalar; WIDTH], block: &[Scalar]) {
    for (lane_idx, input) in block.iter().enumerate() {
        state[lane_idx + 1] += *input;
    }
}

fn absorb_block_elts<CS: ConstraintSystem<Scalar>>(
    state: &mut [Elt],
    block: &[AllocatedNum<Scalar>],
) -> Result<(), SynthesisError> {
    for (lane_idx, input) in block.iter().enumerate() {
        state[lane_idx + 1] = state[lane_idx + 1]
            .clone()
            .add(Elt::Allocated(input.clone()))?;
    }
    Ok(())
}

fn permute(state: &mut [Scalar; WIDTH], params: &Poseidon2Params) {
    apply_external_matrix_native(state);

    for round in 0..(ROUNDS_F / 2) {
        add_round_constants(state, &params.round_constants[round]);
        apply_sbox_full_native(state);
        apply_external_matrix_native(state);
    }

    for round in (ROUNDS_F / 2)..((ROUNDS_F / 2) + ROUNDS_P) {
        state[0] += params.round_constants[round][0];
        state[0] = quintic_sbox_native(state[0]);
        apply_internal_matrix_native(state, &params.mat_internal_diag_m_1);
    }

    for round in ((ROUNDS_F / 2) + ROUNDS_P)..TOTAL_ROUNDS {
        add_round_constants(state, &params.round_constants[round]);
        apply_sbox_full_native(state);
        apply_external_matrix_native(state);
    }
}

fn permute_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    state: Vec<Elt>,
) -> Result<Vec<Elt>, SynthesisError> {
    let params = params();
    let mut current = apply_external_matrix_allocated::<CS>(state.as_slice())?;

    for round in 0..(ROUNDS_F / 2) {
        current = add_round_constants_allocated::<CS>(&current, &params.round_constants[round])?;
        current = apply_sbox_full_allocated(
            &mut cs.namespace(|| format!("full_round_{round}_sbox")),
            &current,
        )?;
        current = apply_external_matrix_allocated::<CS>(current.as_slice())?;
    }

    for round in (ROUNDS_F / 2)..((ROUNDS_F / 2) + ROUNDS_P) {
        let with_constant = current[0]
            .clone()
            .add(Elt::num_from_fr::<CS>(params.round_constants[round][0]))?;
        current[0] = quintic_sbox_allocated(
            &mut cs.namespace(|| format!("partial_round_{round}_sbox")),
            &with_constant,
        )?;
        current = apply_internal_matrix_allocated::<CS>(&current, &params.mat_internal_diag_m_1)?;
    }

    for round in ((ROUNDS_F / 2) + ROUNDS_P)..TOTAL_ROUNDS {
        current = add_round_constants_allocated::<CS>(&current, &params.round_constants[round])?;
        current = apply_sbox_full_allocated(
            &mut cs.namespace(|| format!("last_full_round_{round}_sbox")),
            &current,
        )?;
        current = apply_external_matrix_allocated::<CS>(current.as_slice())?;
    }

    Ok(current)
}

fn add_round_constants(state: &mut [Scalar; WIDTH], round_constants: &[Scalar; WIDTH]) {
    for (lane, constant) in state.iter_mut().zip(round_constants.iter()) {
        *lane += *constant;
    }
}

fn add_round_constants_allocated<CS: ConstraintSystem<Scalar>>(
    state: &[Elt],
    round_constants: &[Scalar; WIDTH],
) -> Result<Vec<Elt>, SynthesisError> {
    state
        .iter()
        .enumerate()
        .map(|(idx, lane)| lane.clone().add(Elt::num_from_fr::<CS>(round_constants[idx])))
        .collect()
}

fn apply_sbox_full_native(state: &mut [Scalar; WIDTH]) {
    for lane in state.iter_mut() {
        *lane = quintic_sbox_native(*lane);
    }
}

fn apply_sbox_full_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    state: &[Elt],
) -> Result<Vec<Elt>, SynthesisError> {
    state
        .iter()
        .enumerate()
        .map(|(idx, lane)| quintic_sbox_allocated(&mut cs.namespace(|| format!("lane_{idx}")), lane))
        .collect()
}

fn quintic_sbox_native(input: Scalar) -> Scalar {
    let input_sq = input.square();
    input_sq.square() * input
}

fn quintic_sbox_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    input: &Elt,
) -> Result<Elt, SynthesisError> {
    let square = input.square(cs.namespace(|| "square"))?;
    let fourth = square.square(cs.namespace(|| "fourth"))?;
    let fifth = mul_sum(
        cs.namespace(|| "fifth"),
        &fourth,
        input,
        None,
        None,
        true,
    )?;
    Ok(Elt::Allocated(fifth))
}

fn apply_external_matrix_native(state: &mut [Scalar; WIDTH]) {
    let matrix = full_matrix();
    let current = *state;
    for row in 0..WIDTH {
        let mut acc = Scalar::ZERO;
        for col in 0..WIDTH {
            acc += matrix[row][col] * current[col];
        }
        state[row] = acc;
    }
}

fn apply_external_matrix_allocated<CS: ConstraintSystem<Scalar>>(
    state: &[Elt],
) -> Result<Vec<Elt>, SynthesisError> {
    let small = small_matrix();
    let mut block_products = Vec::with_capacity(WIDTH / 4);
    for block in state.chunks_exact(4) {
        let mut rows = Vec::with_capacity(4);
        for row in small.iter() {
            rows.push(scalar_product::<CS>(block, row)?);
        }
        block_products.push(rows);
    }

    let mut global_rows = Vec::with_capacity(4);
    for row_idx in 0..4 {
        let mut acc = Elt::Num(num::Num::zero());
        for block in &block_products {
            acc = acc.add(block[row_idx].clone())?;
        }
        global_rows.push(acc);
    }

    let mut out = Vec::with_capacity(WIDTH);
    for block in &block_products {
        for row_idx in 0..4 {
            out.push(global_rows[row_idx].clone().add(block[row_idx].clone())?);
        }
    }
    Ok(out)
}

fn apply_internal_matrix_native(state: &mut [Scalar; WIDTH], diag_m_1: &[Scalar; WIDTH]) {
    let input = *state;
    let mut sum = Scalar::ZERO;
    for value in input {
        sum += value;
    }
    for idx in 0..WIDTH {
        state[idx] = sum + (diag_m_1[idx] * input[idx]);
    }
}

fn apply_internal_matrix_allocated<CS: ConstraintSystem<Scalar>>(
    state: &[Elt],
    diag_m_1: &[Scalar; WIDTH],
) -> Result<Vec<Elt>, SynthesisError> {
    let sum = state.iter().try_fold(Elt::Num(num::Num::zero()), |acc, lane| {
        acc.add(lane.clone())
    })?;

    (0..WIDTH)
        .map(|idx| sum.clone().add(state[idx].clone().scale::<CS>(diag_m_1[idx])?))
        .collect()
}

fn scalar_product<CS: ConstraintSystem<Scalar>>(
    elts: &[Elt],
    scalars: &[Scalar],
) -> Result<Elt, SynthesisError> {
    elts.iter()
        .zip(scalars)
        .try_fold(Elt::Num(num::Num::zero()), |acc, (elt, &scalar)| {
            acc.add(elt.clone().scale::<CS>(scalar)?)
        })
}

fn materialize_state<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    state: &[Elt],
) -> Result<Vec<Elt>, SynthesisError> {
    state
        .iter()
        .enumerate()
        .map(|(idx, lane)| {
            lane.ensure_allocated(&mut cs.namespace(|| format!("lane_{idx}")), true)
                .map(Elt::Allocated)
        })
        .collect()
}

fn mul_sum<CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    a: &AllocatedNum<Scalar>,
    b: &Elt,
    pre_add: Option<Scalar>,
    post_add: Option<Scalar>,
    enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
        let mut tmp = b.val().ok_or(SynthesisError::AssignmentMissing)?;
        if let Some(x) = pre_add {
            tmp += x;
        }
        tmp *= a.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        if let Some(x) = post_add {
            tmp += x;
        }
        Ok(tmp)
    })?;

    if enforce {
        if let Some(x) = post_add {
            let neg = -x;
            if let Some(pre) = pre_add {
                cs.enforce(
                    || "mul_sum_constraint_pre_post_add",
                    |_| b.lc() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            } else {
                cs.enforce(
                    || "mul_sum_constraint_post_add",
                    |_| b.lc(),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            }
        } else if let Some(pre) = pre_add {
            cs.enforce(
                || "mul_sum_constraint_pre_add",
                |_| b.lc() + (pre, CS::one()),
                |lc| lc + a.get_variable(),
                |lc| lc + res.get_variable(),
            );
        } else {
            cs.enforce(
                || "mul_sum_constraint",
                |_| b.lc(),
                |lc| lc + a.get_variable(),
                |lc| lc + res.get_variable(),
            );
        }
    }
    Ok(res)
}

fn load_checked_in_params() -> Poseidon2Params {
    const GENERATED_PARAMS: &str =
        include_str!("../../poseidon2/generated/poseidon2_instance_pallas_12.rs");

    let diag = extract_hex_block(
        GENERATED_PARAMS,
        "pub static ref MAT_DIAG12_M_1",
        "pub static ref MAT_INTERNAL12",
    );
    let round_constants = extract_hex_block(
        GENERATED_PARAMS,
        "pub static ref RC12",
        "pub static ref POSEIDON2_PALLAS_12_PARAMS",
    );

    Poseidon2Params {
        mat_internal_diag_m_1: diag
            .try_into()
            .expect("checked-in Poseidon2 diagonal has width 12"),
        round_constants: round_constants
            .chunks_exact(WIDTH)
            .map(|row| row.try_into().expect("checked-in Poseidon2 round width is 12"))
            .collect::<Vec<[Scalar; WIDTH]>>()
            .try_into()
            .expect("checked-in Poseidon2 round count matches width-12 instance"),
    }
}

fn extract_hex_block(source: &str, start_marker: &str, end_marker: &str) -> Vec<Scalar> {
    let start = source
        .find(start_marker)
        .expect("Poseidon2 generated file contains the requested block");
    let end = source[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .expect("Poseidon2 generated file contains the block terminator");
    let block = &source[start..end];

    let mut out = Vec::new();
    let bytes = block.as_bytes();
    let mut idx = 0;
    while idx + 1 < bytes.len() {
        if bytes[idx] == b'0' && bytes[idx + 1] == b'x' {
            let start_hex = idx;
            idx += 2;
            while idx < bytes.len() && bytes[idx].is_ascii_hexdigit() {
                idx += 1;
            }
            out.push(scalar_from_hex(&block[start_hex..idx]));
        } else {
            idx += 1;
        }
    }
    out
}

fn scalar_from_hex(hex: &str) -> Scalar {
    let magnitude = BigUint::parse_bytes(hex.trim_start_matches("0x").as_bytes(), 16)
        .expect("checked-in Poseidon2 constant is valid hex");
    Scalar::from_str_vartime(&magnitude.to_str_radix(10))
        .expect("checked-in Poseidon2 constant fits the Pallas scalar field")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_snark::{
        frontend::{
            r1cs::{NovaShape, NovaWitness},
            shape_cs::ShapeCS,
            solver::SatisfyingAssignment,
        },
        provider::PallasEngine,
        r1cs::R1CSShape,
        traits::snark::default_ck_hint,
    };

    #[test]
    fn poseidon2_compress_3840_is_deterministic() {
        let inputs = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        assert_eq!(
            poseidon2_compress_3840(&inputs),
            poseidon2_compress_3840(&inputs)
        );
    }

    #[test]
    fn poseidon2_chain_2_is_deterministic() {
        let inputs = [Scalar::from(1u64), Scalar::from(2u64)];
        assert_eq!(poseidon2_chain_2(inputs), poseidon2_chain_2(inputs));
    }

    #[test]
    fn poseidon2_compress_3840_gadget_matches_native() {
        let inputs = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        let expected = poseidon2_compress_3840(&inputs);

        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        let allocated_shape = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(shape_cs.namespace(|| format!("shape_input_{idx}")), || {
                Scalar::from((idx + 1) as u64)
            })
        });
        poseidon2_compress_3840_allocated(&mut shape_cs, &allocated_shape).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PallasEngine>::new();
        let allocated_witness = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                witness_cs.namespace(|| format!("witness_input_{idx}")),
                || Scalar::from((idx + 1) as u64),
            )
        });
        let digest =
            poseidon2_compress_3840_allocated(&mut witness_cs, &allocated_witness).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert_eq!(digest.get_value().unwrap(), expected);
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
    }

    #[test]
    fn poseidon2_chain_2_gadget_matches_native() {
        let inputs = [Scalar::from(7u64), Scalar::from(11u64)];
        let expected = poseidon2_chain_2(inputs);

        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        let allocated_shape = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(shape_cs.namespace(|| format!("shape_input_{idx}")), || {
                inputs[idx]
            })
        });
        poseidon2_chain_2_allocated(&mut shape_cs, allocated_shape).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PallasEngine>::new();
        let allocated_witness = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                witness_cs.namespace(|| format!("witness_input_{idx}")),
                || inputs[idx],
            )
        });
        let digest = poseidon2_chain_2_allocated(&mut witness_cs, allocated_witness).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert_eq!(digest.get_value().unwrap(), expected);
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
    }
}
