use nova_snark::{
    frontend::gadgets::poseidon::{
        IOPattern, Simplex, Sponge, SpongeAPI, SpongeOp, SpongeTrait, Strength,
    },
    frontend::{
        num::AllocatedNum, ConstraintSystem, Elt, PoseidonConstants, SpongeCircuit, SynthesisError,
    },
    provider::PallasEngine,
    traits::Engine,
};
use std::sync::OnceLock;
use typenum::{U2, U8};

pub type Scalar = <PallasEngine as Engine>::Scalar;

pub const HASH8_INPUTS: usize = 8;
pub const HASH2_INPUTS: usize = 2;

type Hash8Arity = U8;
type Hash2Arity = U2;

const HASH8_DOMAIN_SEPARATOR: u32 = 0x4841_5348;
const HASH2_DOMAIN_SEPARATOR: u32 = 0x5041_4952;

static HASH8_CONSTANTS: OnceLock<PoseidonConstants<Scalar, Hash8Arity>> = OnceLock::new();
static HASH2_CONSTANTS: OnceLock<PoseidonConstants<Scalar, Hash2Arity>> = OnceLock::new();

pub fn poseidon_hash_8(inputs: &[Scalar; HASH8_INPUTS]) -> Scalar {
    let mut sponge = Sponge::<Scalar, Hash8Arity>::new_with_constants(hash8_constants(), Simplex);
    sponge.start(hash8_pattern(), Some(HASH8_DOMAIN_SEPARATOR), &mut ());
    sponge.absorb(HASH8_INPUTS as u32, inputs.as_slice(), &mut ());
    let digest = SpongeAPI::squeeze(&mut sponge, 1, &mut ())
        .into_iter()
        .next()
        .expect("Poseidon8 squeezes one digest");
    sponge
        .finish(&mut ())
        .expect("fixed Poseidon8 pattern always finishes");
    digest
}

pub fn poseidon_hash_8_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    inputs: &[AllocatedNum<Scalar>; HASH8_INPUTS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let mut sponge: SpongeCircuit<'_, Scalar, Hash8Arity, CS::Root> =
        SpongeCircuit::new_with_constants(hash8_constants(), Simplex);
    let input_elts = inputs.iter().cloned().map(Elt::from).collect::<Vec<_>>();
    let mut sponge_cs = cs.namespace(|| "poseidon_hash_8_sponge");
    sponge.start(
        hash8_pattern(),
        Some(HASH8_DOMAIN_SEPARATOR),
        &mut sponge_cs,
    );
    sponge.absorb(HASH8_INPUTS as u32, &input_elts, &mut sponge_cs);
    let digest = SpongeAPI::squeeze(&mut sponge, 1, &mut sponge_cs)
        .into_iter()
        .next()
        .expect("Poseidon8 squeezes one digest");
    sponge.finish(&mut sponge_cs).map_err(|_| {
        SynthesisError::Unsatisfiable("poseidon hash8 sponge pattern mismatch".to_string())
    })?;
    let allocated = digest.ensure_allocated(&mut sponge_cs.namespace(|| "digest_output"), true)?;
    Ok(allocated)
}

pub fn poseidon_hash_2(inputs: [Scalar; HASH2_INPUTS]) -> Scalar {
    let mut sponge = Sponge::<Scalar, Hash2Arity>::new_with_constants(hash2_constants(), Simplex);
    sponge.start(hash2_pattern(), Some(HASH2_DOMAIN_SEPARATOR), &mut ());
    sponge.absorb(HASH2_INPUTS as u32, &inputs, &mut ());
    let digest = SpongeAPI::squeeze(&mut sponge, 1, &mut ())
        .into_iter()
        .next()
        .expect("Poseidon2 squeezes one digest");
    sponge
        .finish(&mut ())
        .expect("fixed Poseidon2 pattern always finishes");
    digest
}

pub fn poseidon_hash_2_allocated<CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    inputs: [AllocatedNum<Scalar>; HASH2_INPUTS],
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let mut sponge: SpongeCircuit<'_, Scalar, Hash2Arity, CS::Root> =
        SpongeCircuit::new_with_constants(hash2_constants(), Simplex);
    let input_elts = inputs.into_iter().map(Elt::from).collect::<Vec<_>>();
    let mut sponge_cs = cs.namespace(|| "poseidon_hash_2_sponge");
    sponge.start(
        hash2_pattern(),
        Some(HASH2_DOMAIN_SEPARATOR),
        &mut sponge_cs,
    );
    sponge.absorb(HASH2_INPUTS as u32, &input_elts, &mut sponge_cs);
    let digest = SpongeAPI::squeeze(&mut sponge, 1, &mut sponge_cs)
        .into_iter()
        .next()
        .expect("Poseidon2 squeezes one digest");
    sponge.finish(&mut sponge_cs).map_err(|_| {
        SynthesisError::Unsatisfiable("poseidon hash2 sponge pattern mismatch".to_string())
    })?;
    let allocated = digest.ensure_allocated(&mut sponge_cs.namespace(|| "digest_output"), true)?;
    Ok(allocated)
}

fn hash8_constants() -> &'static PoseidonConstants<Scalar, Hash8Arity> {
    HASH8_CONSTANTS.get_or_init(|| {
        <Sponge<'static, Scalar, Hash8Arity> as SpongeTrait<'static, Scalar, Hash8Arity>>::api_constants(
            Strength::Standard,
        )
    })
}

fn hash2_constants() -> &'static PoseidonConstants<Scalar, Hash2Arity> {
    HASH2_CONSTANTS.get_or_init(|| {
        <Sponge<'static, Scalar, Hash2Arity> as SpongeTrait<'static, Scalar, Hash2Arity>>::api_constants(
            Strength::Standard,
        )
    })
}

fn hash8_pattern() -> IOPattern {
    IOPattern(vec![
        SpongeOp::Absorb(HASH8_INPUTS as u32),
        SpongeOp::Squeeze(1),
    ])
}

fn hash2_pattern() -> IOPattern {
    IOPattern(vec![
        SpongeOp::Absorb(HASH2_INPUTS as u32),
        SpongeOp::Squeeze(1),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_snark::frontend::{
        r1cs::{NovaShape, NovaWitness},
        shape_cs::ShapeCS,
        solver::SatisfyingAssignment,
    };
    use nova_snark::{r1cs::R1CSShape, traits::snark::default_ck_hint};

    #[test]
    fn poseidon_hash_8_is_deterministic() {
        let inputs = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        assert_eq!(poseidon_hash_8(&inputs), poseidon_hash_8(&inputs));
    }

    #[test]
    fn poseidon_hash_2_is_deterministic() {
        let inputs = [Scalar::from(1u64), Scalar::from(2u64)];
        assert_eq!(poseidon_hash_2(inputs), poseidon_hash_2(inputs));
    }

    #[test]
    fn poseidon_hash_8_gadget_matches_native() {
        let inputs = std::array::from_fn(|idx| Scalar::from((idx + 1) as u64));
        let expected = poseidon_hash_8(&inputs);

        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        let allocated_shape = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                shape_cs.namespace(|| format!("shape_input_{idx}")),
                || Scalar::from((idx + 1) as u64),
            )
        });
        poseidon_hash_8_allocated(&mut shape_cs, &allocated_shape).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PallasEngine>::new();
        let allocated_witness = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                witness_cs.namespace(|| format!("witness_input_{idx}")),
                || Scalar::from((idx + 1) as u64),
            )
        });
        let digest = poseidon_hash_8_allocated(&mut witness_cs, &allocated_witness).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert_eq!(digest.get_value().unwrap(), expected);
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
    }

    #[test]
    fn poseidon_hash_2_gadget_matches_native() {
        let inputs = [Scalar::from(7u64), Scalar::from(11u64)];
        let expected = poseidon_hash_2(inputs);

        let mut shape_cs: ShapeCS<PallasEngine> = ShapeCS::new();
        let allocated_shape = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                shape_cs.namespace(|| format!("shape_input_{idx}")),
                || inputs[idx],
            )
        });
        poseidon_hash_2_allocated(&mut shape_cs, allocated_shape).unwrap();
        let shape = shape_cs.r1cs_shape().unwrap();
        let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).unwrap();

        let mut witness_cs = SatisfyingAssignment::<PallasEngine>::new();
        let allocated_witness = std::array::from_fn(|idx| {
            AllocatedNum::alloc_infallible(
                witness_cs.namespace(|| format!("witness_input_{idx}")),
                || inputs[idx],
            )
        });
        let digest = poseidon_hash_2_allocated(&mut witness_cs, allocated_witness).unwrap();
        let (instance, witness) = witness_cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

        assert_eq!(digest.get_value().unwrap(), expected);
        assert!(shape.is_sat(&ck, &instance, &witness).is_ok());
    }
}
