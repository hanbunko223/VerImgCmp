import CircuitCorrectness.HashTrace8.Output
import CircuitCorrectness.HashTrace
import CircuitCorrectness.HashWiringTemplates
set_option maxRecDepth 20000
set_option maxHeartbeats 2000000
namespace CircuitCorrectness.HashTrace8
open PoseidonProgram StraightLine
def states : Nat → PoseidonProgram.State
  | 0 => state00
  | 1 => state01
  | 2 => state02
  | 3 => state03
  | 4 => state04
  | 5 => state05
  | 6 => state06
  | 7 => state07
  | 8 => state08
  | 9 => state09
  | 10 => state10
  | 11 => state11
  | 12 => state12
  | 13 => state13
  | 14 => state14
  | 15 => state15
  | 16 => state16
  | 17 => state17
  | 18 => state18
  | 19 => state19
  | 20 => state20
  | 21 => state21
  | 22 => state22
  | 23 => state23
  | 24 => state24
  | 25 => state25
  | 26 => state26
  | 27 => state27
  | 28 => state28
  | 29 => state29
  | 30 => state30
  | 31 => state31
  | 32 => state32
  | 33 => state33
  | 34 => state34
  | 35 => state35
  | 36 => state36
  | 37 => state37
  | 38 => state38
  | 39 => state39
  | 40 => state40
  | 41 => state41
  | 42 => state42
  | 43 => state43
  | 44 => state44
  | 45 => state45
  | 46 => state46
  | 47 => state47
  | 48 => state48
  | 49 => state49
  | 50 => state50
  | 51 => state51
  | 52 => state52
  | 53 => state53
  | 54 => state54
  | 55 => state55
  | 56 => state56
  | 57 => state57
  | 58 => state58
  | 59 => state59
  | 60 => state60
  | 61 => state61
  | 62 => state62
  | 63 => state63
  | 64 => state64
  | 65 => state65
  | _ => state65
def ops : Nat → Program
  | 0 => ops00
  | 1 => ops01
  | 2 => ops02
  | 3 => ops03
  | 4 => ops04
  | 5 => ops05
  | 6 => ops06
  | 7 => ops07
  | 8 => ops08
  | 9 => ops09
  | 10 => ops10
  | 11 => ops11
  | 12 => ops12
  | 13 => ops13
  | 14 => ops14
  | 15 => ops15
  | 16 => ops16
  | 17 => ops17
  | 18 => ops18
  | 19 => ops19
  | 20 => ops20
  | 21 => ops21
  | 22 => ops22
  | 23 => ops23
  | 24 => ops24
  | 25 => ops25
  | 26 => ops26
  | 27 => ops27
  | 28 => ops28
  | 29 => ops29
  | 30 => ops30
  | 31 => ops31
  | 32 => ops32
  | 33 => ops33
  | 34 => ops34
  | 35 => ops35
  | 36 => ops36
  | 37 => ops37
  | 38 => ops38
  | 39 => ops39
  | 40 => ops40
  | 41 => ops41
  | 42 => ops42
  | 43 => ops43
  | 44 => ops44
  | 45 => ops45
  | 46 => ops46
  | 47 => ops47
  | 48 => ops48
  | 49 => ops49
  | 50 => ops50
  | 51 => ops51
  | 52 => ops52
  | 53 => ops53
  | 54 => ops54
  | 55 => ops55
  | 56 => ops56
  | 57 => ops57
  | 58 => ops58
  | 59 => ops59
  | 60 => ops60
  | 61 => ops61
  | 62 => ops62
  | 63 => ops63
  | 64 => ops64
  | _ => ops64
def codes : Nat → List Row
  | 0 => codes00
  | 1 => codes01
  | 2 => codes02
  | 3 => codes03
  | 4 => codes04
  | 5 => codes05
  | 6 => codes06
  | 7 => codes07
  | 8 => codes08
  | 9 => codes09
  | 10 => codes10
  | 11 => codes11
  | 12 => codes12
  | 13 => codes13
  | 14 => codes14
  | 15 => codes15
  | 16 => codes16
  | 17 => codes17
  | 18 => codes18
  | 19 => codes19
  | 20 => codes20
  | 21 => codes21
  | 22 => codes22
  | 23 => codes23
  | 24 => codes24
  | 25 => codes25
  | 26 => codes26
  | 27 => codes27
  | 28 => codes28
  | 29 => codes29
  | 30 => codes30
  | 31 => codes31
  | 32 => codes32
  | 33 => codes33
  | 34 => codes34
  | 35 => codes35
  | 36 => codes36
  | 37 => codes37
  | 38 => codes38
  | 39 => codes39
  | 40 => codes40
  | 41 => codes41
  | 42 => codes42
  | 43 => codes43
  | 44 => codes44
  | 45 => codes45
  | 46 => codes46
  | 47 => codes47
  | 48 => codes48
  | 49 => codes49
  | 50 => codes50
  | 51 => codes51
  | 52 => codes52
  | 53 => codes53
  | 54 => codes54
  | 55 => codes55
  | 56 => codes56
  | 57 => codes57
  | 58 => codes58
  | 59 => codes59
  | 60 => codes60
  | 61 => codes61
  | 62 => codes62
  | 63 => codes63
  | 64 => codes64
  | _ => codes64
def allCodes : List Row := (List.range 65).flatMap codes ++ outputCodes

theorem actual_codes : HashWiring.template8Codes = allCodes := by
  rw [HashWiring.template8_literal]
  decide +kernel

theorem checkpoints : ∀ r<65,
    round Spec.params8 97634 (states r) r = (states (r+1),ops r) := by
  intro r hr
  interval_cases r
  · exact round_exact00
  · exact round_exact01
  · exact round_exact02
  · exact round_exact03
  · exact round_exact04
  · exact round_exact05
  · exact round_exact06
  · exact round_exact07
  · exact round_exact08
  · exact round_exact09
  · exact round_exact10
  · exact round_exact11
  · exact round_exact12
  · exact round_exact13
  · exact round_exact14
  · exact round_exact15
  · exact round_exact16
  · exact round_exact17
  · exact round_exact18
  · exact round_exact19
  · exact round_exact20
  · exact round_exact21
  · exact round_exact22
  · exact round_exact23
  · exact round_exact24
  · exact round_exact25
  · exact round_exact26
  · exact round_exact27
  · exact round_exact28
  · exact round_exact29
  · exact round_exact30
  · exact round_exact31
  · exact round_exact32
  · exact round_exact33
  · exact round_exact34
  · exact round_exact35
  · exact round_exact36
  · exact round_exact37
  · exact round_exact38
  · exact round_exact39
  · exact round_exact40
  · exact round_exact41
  · exact round_exact42
  · exact round_exact43
  · exact round_exact44
  · exact round_exact45
  · exact round_exact46
  · exact round_exact47
  · exact round_exact48
  · exact round_exact49
  · exact round_exact50
  · exact round_exact51
  · exact round_exact52
  · exact round_exact53
  · exact round_exact54
  · exact round_exact55
  · exact round_exact56
  · exact round_exact57
  · exact round_exact58
  · exact round_exact59
  · exact round_exact60
  · exact round_exact61
  · exact round_exact62
  · exact round_exact63
  · exact round_exact64
theorem extracted : ∀ r<65, ConcreteProgram.extractRows 97634 (states r).next
    ((codes r).map ConcreteBytes.Codes.expandRow) = some (ops r) := by
  intro r hr
  interval_cases r
  · exact rows_exact00
  · exact rows_exact01
  · exact rows_exact02
  · exact rows_exact03
  · exact rows_exact04
  · exact rows_exact05
  · exact rows_exact06
  · exact rows_exact07
  · exact rows_exact08
  · exact rows_exact09
  · exact rows_exact10
  · exact rows_exact11
  · exact rows_exact12
  · exact rows_exact13
  · exact rows_exact14
  · exact rows_exact15
  · exact rows_exact16
  · exact rows_exact17
  · exact rows_exact18
  · exact rows_exact19
  · exact rows_exact20
  · exact rows_exact21
  · exact rows_exact22
  · exact rows_exact23
  · exact rows_exact24
  · exact rows_exact25
  · exact rows_exact26
  · exact rows_exact27
  · exact rows_exact28
  · exact rows_exact29
  · exact rows_exact30
  · exact rows_exact31
  · exact rows_exact32
  · exact rows_exact33
  · exact rows_exact34
  · exact rows_exact35
  · exact rows_exact36
  · exact rows_exact37
  · exact rows_exact38
  · exact rows_exact39
  · exact rows_exact40
  · exact rows_exact41
  · exact rows_exact42
  · exact rows_exact43
  · exact rows_exact44
  · exact rows_exact45
  · exact rows_exact46
  · exact rows_exact47
  · exact rows_exact48
  · exact rows_exact49
  · exact rows_exact50
  · exact rows_exact51
  · exact rows_exact52
  · exact rows_exact53
  · exact rows_exact54
  · exact rows_exact55
  · exact rows_exact56
  · exact rows_exact57
  · exact rows_exact58
  · exact rows_exact59
  · exact rows_exact60
  · exact rows_exact61
  · exact rows_exact62
  · exact rows_exact63
  · exact rows_exact64
def input : Array LinearCombination := #[Affine.wire 77484,Affine.wire 77485,Affine.wire 77486,Affine.wire 77487,Affine.wire 77488,Affine.wire 77489,Affine.wire 77490,Affine.wire 77491]

theorem initial : states 0 =
    ⟨#[Affine.constant 97634 (Spec.domainTag Spec.params8.arity 1212240712)] ++ input,0,77500⟩ := by decide +kernel

theorem output_extracted : ConcreteProgram.extractRows 97634 77887
    (outputCodes.map ConcreteBytes.Codes.expandRow) =
    some [⟨77887,(states 65).values[1]!,Affine.constant 97634 1,[]⟩] := by decide +kernel

theorem template_sound (w : Assignment) (h1 : w 97634=1)
    (hs : ∀ row∈HashWiring.template8,row.Sat w) :
    w 77887 = Spec.hash Spec.params8 1212240712 (evalState w input) := by
  simp only [HashWiring.template8,actual_codes] at hs
  have hlocal : ∀ r<65, Satisfies (ops r) w := by
    intro r hr
    apply (ConcreteProgram.extractRows_correct (extracted r hr) h1).mp
    intro row hm
    rcases List.mem_map.mp hm with ⟨code,hcode,rfl⟩
    apply hs _
    apply List.mem_map.mpr
    refine ⟨code,List.mem_append_left _ ?_,rfl⟩
    exact List.mem_flatMap.mpr ⟨r,List.mem_range.mpr hr,hcode⟩
  have hout : Satisfies
      [⟨77887,(states 65).values[1]!,Affine.constant 97634 1,[]⟩] w := by
    apply (ConcreteProgram.extractRows_correct output_extracted h1).mp
    intro row hm
    rcases List.mem_map.mp hm with ⟨code,hcode,rfl⟩
    exact hs _ (List.mem_map.mpr ⟨code,List.mem_append_right _ hcode,rfl⟩)
  have ho := hout _ (List.mem_cons_self)
  simp only [Instruction.Sat,Instruction.value,Affine.eval_constant,Nat.cast_one,
    h1,mul_one,Affine.eval_nil,add_zero] at ho
  apply HashTrace.hash_sound w Spec.params8 97634 1212240712 77887
    (evalState w input) states ops h1 checkpoints hlocal ?_ ho
  rw [initial]
  simp [evalState,h1,Array.map_append]

#print axioms template_sound
end CircuitCorrectness.HashTrace8
