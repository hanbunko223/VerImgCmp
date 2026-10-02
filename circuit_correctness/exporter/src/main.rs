#![allow(dead_code)]
// These are the unmodified production modules, not a hand-translated circuit.
#[path = "../../../SNARKPEG_Poseidon/src/circuit.rs"] mod circuit;
#[path = "../../../SNARKPEG_Poseidon/src/coefficient.rs"] mod coefficient;
#[path = "../../../SNARKPEG_Poseidon/src/dctq.rs"] mod dctq;
#[path = "../../../SNARKPEG_Poseidon/src/hash.rs"] mod hash;
#[path = "../../../SNARKPEG_Poseidon/src/input.rs"] mod input;
#[path = "../../../SNARKPEG_Poseidon/src/poseidon.rs"] mod poseidon;

use std::{collections::BTreeMap, fs::File, io::BufWriter, path::Path};
use ff::{Field, PrimeField};
use nova_snark::{frontend::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
    num::AllocatedNum, shape_cs::ShapeCS, solver::SatisfyingAssignment},
    provider::PallasEngine, traits::circuit::StepCircuit};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use hash::Scalar;

type LC = LinearCombination<Scalar>;
type Row = [Vec<(usize, String)>; 3];

struct TraceCS {
    inner: ShapeCS<PallasEngine>,
    namespace: Vec<String>,
    allocation_names: Vec<String>,
    constraint_names: Vec<String>,
}
impl TraceCS {
    fn path(&self, name: String) -> String {
        self.namespace.iter().cloned().chain(std::iter::once(name)).collect::<Vec<_>>().join("/")
    }
}
impl ConstraintSystem<Scalar> for TraceCS {
    type Root = Self;
    fn new() -> Self { Self { inner: ShapeCS::new(), namespace: vec![], allocation_names: vec![], constraint_names: vec![] } }
    fn alloc<F,A,AR>(&mut self, annotation:A, f:F)->Result<Variable,SynthesisError>
    where F:FnOnce()->Result<Scalar,SynthesisError>, A:FnOnce()->AR, AR:Into<String> {
        let name = self.path(annotation().into());
        let v = self.inner.alloc(|| name.clone(), f)?;
        assert_eq!(v.get_unchecked(), Index::Aux(self.allocation_names.len()));
        self.allocation_names.push(name);
        Ok(v)
    }
    fn alloc_input<F,A,AR>(&mut self, _:A, _:F)->Result<Variable,SynthesisError>
    where F:FnOnce()->Result<Scalar,SynthesisError>, A:FnOnce()->AR, AR:Into<String> {
        panic!("The application step must not allocate public inputs; its interface is explicit wire maps")
    }
    fn enforce<A,AR,LA,LB,LCF>(&mut self, annotation:A, a:LA,b:LB,c:LCF)
    where A:FnOnce()->AR,AR:Into<String>,LA:FnOnce(LC)->LC,LB:FnOnce(LC)->LC,LCF:FnOnce(LC)->LC {
        self.constraint_names.push(self.path(annotation().into()));
        self.inner.enforce(|| "captured", a,b,c);
    }
    fn push_namespace<NR,N>(&mut self,n:N) where NR:Into<String>,N:FnOnce()->NR {self.namespace.push(n().into());}
    fn pop_namespace(&mut self) {self.namespace.pop().expect("namespace underflow");}
    fn get_root(&mut self)->&mut Self {self}
}
fn decimal(x:&Scalar)->String {hash::scalar_to_decimal_string(x)}
fn index(v:Variable,n:usize)->usize {match v.get_unchecked(){Index::Aux(i)=>i,Index::Input(i)=>n+i}}
fn rows(cs:&ShapeCS<PallasEngine>)->Vec<Row> {
    cs.constraints.iter().map(|(a,b,c)| [a,b,c].map(|lc| {
        let mut terms = BTreeMap::<usize,Scalar>::new();
        for (v,k) in lc.iter() { *terms.entry(index(v,cs.num_aux())).or_insert(Scalar::ZERO) += k; }
        terms.into_iter().filter(|(_,k)| *k!=Scalar::ZERO).map(|(v,k)|(v,decimal(&k))).collect()
    })).collect()
}
fn synthesize<CS:ConstraintSystem<Scalar>>(cs:&mut CS,p:circuit::PreparedStep,z:[Scalar;4])->Vec<AllocatedNum<Scalar>> {
    let inputs = z.into_iter().enumerate().map(|(i,x)|AllocatedNum::alloc_infallible(cs.namespace(||format!("incoming_{i}")),||x)).collect::<Vec<_>>();
    circuit::DctqStepCircuit::new(p).synthesize(cs,&inputs).unwrap()
}
fn write_json(path:&Path,value:&impl Serialize) {
    serde_json::to_writer(BufWriter::new(File::create(path).unwrap()),value).unwrap();
}
fn fingerprint(rows:&[Row])->String {format!("{:x}",Sha256::digest(serde_json::to_vec(rows).unwrap()))}
fn fixture(kind:usize)->input::DctqStep {
    let mut state=0x97630123456789abu64;
    std::array::from_fn(|r|std::array::from_fn(|c|std::array::from_fn(|ch|{
        state ^= state<<13; state ^= state>>7; state ^= state<<17;
        match kind {0=>0,1=>255,2=>if (r+c+ch)%2==0 {0}else{255},3=>if r==3&&c==5&&ch==1 {255}else{0},_=>state as u8}
    })))
}
type FieldRow = [Vec<(usize,Scalar)>;3];
fn eval(lc:&[(usize,Scalar)],w:&[Scalar])->Scalar {
    lc.iter().fold(Scalar::ZERO,|s,(i,k)|s+w[*i]*k)
}
fn failures(rs:&[FieldRow],w:&[Scalar])->Vec<usize> {
    rs.iter().enumerate().filter(|(_,r)|eval(&r[0],w)*eval(&r[1],w)!=eval(&r[2],w)).map(|(i,_)|i).collect()
}
fn source_check() {
    let script=Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts/sources.py");
    assert!(std::process::Command::new("python3").arg(script).arg("check-source").status().unwrap().success(),"source identity check failed");
}
fn main() {
    source_check();
    let args=std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str)==Some("check-source"){return}
    let command=args.get(1).map(String::as_str).unwrap_or("export");
    assert!(matches!(command,"export"|"check-export"));
    let base=Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let out=base.join("artifacts");
    let p=circuit::PreparedStep::from_step(fixture(0));
    let mut trace=TraceCS::new();
    let outputs=synthesize(&mut trace,p.clone(),[Scalar::ZERO;4]);
    let n=trace.inner.num_aux();
    assert_eq!(n,97634); assert_eq!(trace.inner.num_constraints(),97630); assert_eq!(trace.inner.num_inputs(),1);
    let rs=rows(&trace.inner);
    let mut plain=ShapeCS::<PallasEngine>::new();
    let plain_out=synthesize(&mut plain,p,[Scalar::ZERO;4]);
    assert_eq!(rs,rows(&plain));
    let output_wires=outputs.iter().map(|v|index(v.get_variable(),n)).collect::<Vec<_>>();
    assert_eq!(output_wires,plain_out.iter().map(|v|index(v.get_variable(),n)).collect::<Vec<_>>());
    assert_eq!(output_wires[2],2);
    let mut pixels=vec![];
    for r in 0..16 {for c in 0..160 {for (ch,name) in ["r","g","b"].iter().enumerate(){
        let name=format!("row_{r}_pixel_{c}/{name}/byte_value/num");
        let i=trace.allocation_names.iter().position(|s|*s==name).unwrap_or_else(||panic!("missing pixel {name}"));
        pixels.push(json!({"row":r,"column":c,"channel":ch,"wire":i}));
    }}}
    let artifact=json!({"format":"poseidon97-step-r1cs-v1","modulus":Scalar::MODULUS,
        "auxiliary_count":n,"variable_count":n+1,"constraint_count":rs.len(),"one_wire":n,
        "incoming_state":[0,1,2,3],"outgoing_state":output_wires,"pixels":pixels,
        "rows":rs,"rows_sha256":fingerprint(&rs)});
    let save = |name: &str, value: &serde_json::Value| {
        if command=="check-export" {
            let old:serde_json::Value=serde_json::from_reader(File::open(out.join(name)).unwrap()).unwrap();
            assert_eq!(&old,value,"regenerated {name} differs");
        } else { write_json(&out.join(name),value); }
    };
    save("step.json",&artifact);
    save("trace.json",&json!({"allocations":trace.allocation_names,"constraints":trace.constraint_names}));
    {
        use nova_snark::frontend::gadgets::poseidon::{Sponge,SpongeTrait,Strength};
        let c2=<Sponge<'static,Scalar,typenum::U2> as SpongeTrait<'static,Scalar,typenum::U2>>::api_constants(Strength::Standard);
        let c8=<Sponge<'static,Scalar,typenum::U8> as SpongeTrait<'static,Scalar,typenum::U8>>::api_constants(Strength::Standard);
        save("parameters.json",&json!({"dct":dctq::A,"divisors":dctq::DIVISORS,"multipliers":dctq::MULTIPLIERS,
            "poseidon2":c2,"poseidon8":c8,"hash2_domain":0x50414952u32,"hash8_domain":0x48415348u32,
            "hash2_pattern":[["absorb",2],["squeeze",1]],"hash8_pattern":[["absorb",8],["squeeze",1]]}));
    }
    let field_rows = rs.iter().map(|r| r.each_ref().map(|lc| lc.iter().map(|(i,k)| (*i,Scalar::from_str_vartime(k).unwrap())).collect())).collect::<Vec<FieldRow>>();
    let mut cases=vec![];
    for kind in 0..5 {
        let prepared=circuit::PreparedStep::from_step(fixture(kind));
        let z=[-Scalar::ONE,Scalar::from(11),Scalar::from(kind as u64),-Scalar::from(2)];
        let mut cs=ShapeCS::<PallasEngine>::new();
        synthesize(&mut cs,prepared.clone(),z);
        assert_eq!(rs,rows(&cs),"shape changed with input");
        let mut wg=SatisfyingAssignment::<PallasEngine>::new();
        let o=synthesize(&mut wg,prepared.clone(),z);
        let mut w=wg.aux_assignment().to_vec(); w.extend_from_slice(wg.input_assignment());
        assert!(failures(&field_rows,&w).is_empty(),"honest witness rejected");
        let coeff=dctq::compute_dctq_flattened(&prepared.step).unwrap();
        let mut acc=z[1];
        for r in 0..16 {for c in 0..160 {for ch in 0..3 {if dctq::active(ch,r%8,c%8){acc=acc*z[2]+coeff[(r*160+c)*3+ch];}}}}
        let expected=[poseidon::poseidon_hash_2([z[0],prepared.step_digest]),acc,z[2],z[3]+Scalar::ONE];
        assert_eq!(o.iter().map(|v|v.get_value().unwrap()).collect::<Vec<_>>(),expected.to_vec());
        let targets=[("pixel_bit","row_0_pixel_0/r/bit_0/boolean"),
            ("dct_aux","dctq_polynomial_evaluation/channel_0_block_row_0_block_0_left_r_0_c_0/linear_output/num"),
            ("prepared_hash","expected_row_hash_0/num")];
        let mut adversarial=vec![];
        for (label,name) in targets {
            let wire=trace.allocation_names.iter().position(|s|s==name).unwrap_or_else(||panic!("missing mutation target {name}"));
            w[wire]+=Scalar::ONE; let bad=failures(&field_rows,&w); w[wire]-=Scalar::ONE;
            assert!(!bad.is_empty()); adversarial.push(json!({"case":label,"wire":wire,"violated_rows":bad}));
        }
        for state_index in [0,1,3] {
            let wire=output_wires[state_index];w[wire]+=Scalar::ONE;let bad=failures(&field_rows,&w);w[wire]-=Scalar::ONE;
            assert!(!bad.is_empty());adversarial.push(json!({"case":format!("output_{state_index}"),"wire":wire,"violated_rows":bad}));
        }
        println!("validated fixture {kind}");
        cases.push(json!({"fixture":kind,"shape_matches":true,"honest_witness_satisfies":true,
            "native_output_matches":true,"incoming":z.iter().map(decimal).collect::<Vec<_>>(),
            "outgoing":expected.iter().map(decimal).collect::<Vec<_>>(),"adversarial":adversarial}));
    }
    write_json(&base.join("results/rust_validation.json"),&json!({"status":"pass","plain_vs_traced":true,"cases":cases}));
    source_check();
    println!("Export verified: {} constraints, {} auxiliary variables, digest {}",rs.len(),n,fingerprint(&rs));
}
