import CircuitCorrectness.ConcreteProgram

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows560 := Exported.decodeRows ExportedData.chunk560.1 ExportedData.chunk560.2
def codes560 := Codes.rows ExportedData.chunk560.1 ExportedData.chunk560.2
def program560 := Coded.program 97634 71684 codes560
theorem checked560 : Coded.checkRows 97634 71684 codes560 = true := by decide
theorem ordered560 : Ordered 97634 71684 program560 := Coded.checked_ordered checked560
theorem length560 : program560.length = 128 := by
  rw [program560, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct560 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows560, row.Sat w) ↔ Satisfies program560 w := by
  rw [rows560, Codes.rows_expand]
  exact Coded.checked_correct checked560 h1

def rows561 := Exported.decodeRows ExportedData.chunk561.1 ExportedData.chunk561.2
def codes561 := Codes.rows ExportedData.chunk561.1 ExportedData.chunk561.2
def program561 := Coded.program 97634 71812 codes561
theorem checked561 : Coded.checkRows 97634 71812 codes561 = true := by decide
theorem ordered561 : Ordered 97634 71812 program561 := Coded.checked_ordered checked561
theorem length561 : program561.length = 128 := by
  rw [program561, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct561 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows561, row.Sat w) ↔ Satisfies program561 w := by
  rw [rows561, Codes.rows_expand]
  exact Coded.checked_correct checked561 h1

def rows562 := Exported.decodeRows ExportedData.chunk562.1 ExportedData.chunk562.2
def codes562 := Codes.rows ExportedData.chunk562.1 ExportedData.chunk562.2
def program562 := Coded.program 97634 71940 codes562
theorem checked562 : Coded.checkRows 97634 71940 codes562 = true := by decide
theorem ordered562 : Ordered 97634 71940 program562 := Coded.checked_ordered checked562
theorem length562 : program562.length = 128 := by
  rw [program562, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct562 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows562, row.Sat w) ↔ Satisfies program562 w := by
  rw [rows562, Codes.rows_expand]
  exact Coded.checked_correct checked562 h1

def rows563 := Exported.decodeRows ExportedData.chunk563.1 ExportedData.chunk563.2
def codes563 := Codes.rows ExportedData.chunk563.1 ExportedData.chunk563.2
def program563 := Coded.program 97634 72068 codes563
theorem checked563 : Coded.checkRows 97634 72068 codes563 = true := by decide
theorem ordered563 : Ordered 97634 72068 program563 := Coded.checked_ordered checked563
theorem length563 : program563.length = 128 := by
  rw [program563, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct563 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows563, row.Sat w) ↔ Satisfies program563 w := by
  rw [rows563, Codes.rows_expand]
  exact Coded.checked_correct checked563 h1

def rows564 := Exported.decodeRows ExportedData.chunk564.1 ExportedData.chunk564.2
def codes564 := Codes.rows ExportedData.chunk564.1 ExportedData.chunk564.2
def program564 := Coded.program 97634 72196 codes564
theorem checked564 : Coded.checkRows 97634 72196 codes564 = true := by decide
theorem ordered564 : Ordered 97634 72196 program564 := Coded.checked_ordered checked564
theorem length564 : program564.length = 128 := by
  rw [program564, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct564 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows564, row.Sat w) ↔ Satisfies program564 w := by
  rw [rows564, Codes.rows_expand]
  exact Coded.checked_correct checked564 h1

def rows565 := Exported.decodeRows ExportedData.chunk565.1 ExportedData.chunk565.2
def codes565 := Codes.rows ExportedData.chunk565.1 ExportedData.chunk565.2
def program565 := Coded.program 97634 72324 codes565
theorem checked565 : Coded.checkRows 97634 72324 codes565 = true := by decide
theorem ordered565 : Ordered 97634 72324 program565 := Coded.checked_ordered checked565
theorem length565 : program565.length = 128 := by
  rw [program565, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct565 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows565, row.Sat w) ↔ Satisfies program565 w := by
  rw [rows565, Codes.rows_expand]
  exact Coded.checked_correct checked565 h1

def rows566 := Exported.decodeRows ExportedData.chunk566.1 ExportedData.chunk566.2
def codes566 := Codes.rows ExportedData.chunk566.1 ExportedData.chunk566.2
def program566 := Coded.program 97634 72452 codes566
theorem checked566 : Coded.checkRows 97634 72452 codes566 = true := by decide
theorem ordered566 : Ordered 97634 72452 program566 := Coded.checked_ordered checked566
theorem length566 : program566.length = 128 := by
  rw [program566, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct566 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows566, row.Sat w) ↔ Satisfies program566 w := by
  rw [rows566, Codes.rows_expand]
  exact Coded.checked_correct checked566 h1

def rows567 := Exported.decodeRows ExportedData.chunk567.1 ExportedData.chunk567.2
def codes567 := Codes.rows ExportedData.chunk567.1 ExportedData.chunk567.2
def program567 := Coded.program 97634 72580 codes567
theorem checked567 : Coded.checkRows 97634 72580 codes567 = true := by decide
theorem ordered567 : Ordered 97634 72580 program567 := Coded.checked_ordered checked567
theorem length567 : program567.length = 128 := by
  rw [program567, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct567 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows567, row.Sat w) ↔ Satisfies program567 w := by
  rw [rows567, Codes.rows_expand]
  exact Coded.checked_correct checked567 h1

def rows568 := Exported.decodeRows ExportedData.chunk568.1 ExportedData.chunk568.2
def codes568 := Codes.rows ExportedData.chunk568.1 ExportedData.chunk568.2
def program568 := Coded.program 97634 72708 codes568
theorem checked568 : Coded.checkRows 97634 72708 codes568 = true := by decide
theorem ordered568 : Ordered 97634 72708 program568 := Coded.checked_ordered checked568
theorem length568 : program568.length = 128 := by
  rw [program568, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct568 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows568, row.Sat w) ↔ Satisfies program568 w := by
  rw [rows568, Codes.rows_expand]
  exact Coded.checked_correct checked568 h1

def rows569 := Exported.decodeRows ExportedData.chunk569.1 ExportedData.chunk569.2
def codes569 := Codes.rows ExportedData.chunk569.1 ExportedData.chunk569.2
def program569 := Coded.program 97634 72836 codes569
theorem checked569 : Coded.checkRows 97634 72836 codes569 = true := by decide
theorem ordered569 : Ordered 97634 72836 program569 := Coded.checked_ordered checked569
theorem length569 : program569.length = 128 := by
  rw [program569, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct569 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows569, row.Sat w) ↔ Satisfies program569 w := by
  rw [rows569, Codes.rows_expand]
  exact Coded.checked_correct checked569 h1

def rows570 := Exported.decodeRows ExportedData.chunk570.1 ExportedData.chunk570.2
def codes570 := Codes.rows ExportedData.chunk570.1 ExportedData.chunk570.2
def program570 := Coded.program 97634 72964 codes570
theorem checked570 : Coded.checkRows 97634 72964 codes570 = true := by decide
theorem ordered570 : Ordered 97634 72964 program570 := Coded.checked_ordered checked570
theorem length570 : program570.length = 128 := by
  rw [program570, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct570 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows570, row.Sat w) ↔ Satisfies program570 w := by
  rw [rows570, Codes.rows_expand]
  exact Coded.checked_correct checked570 h1

def rows571 := Exported.decodeRows ExportedData.chunk571.1 ExportedData.chunk571.2
def codes571 := Codes.rows ExportedData.chunk571.1 ExportedData.chunk571.2
def program571 := Coded.program 97634 73092 codes571
theorem checked571 : Coded.checkRows 97634 73092 codes571 = true := by decide
theorem ordered571 : Ordered 97634 73092 program571 := Coded.checked_ordered checked571
theorem length571 : program571.length = 128 := by
  rw [program571, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct571 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows571, row.Sat w) ↔ Satisfies program571 w := by
  rw [rows571, Codes.rows_expand]
  exact Coded.checked_correct checked571 h1

def rows572 := Exported.decodeRows ExportedData.chunk572.1 ExportedData.chunk572.2
def codes572 := Codes.rows ExportedData.chunk572.1 ExportedData.chunk572.2
def program572 := Coded.program 97634 73220 codes572
theorem checked572 : Coded.checkRows 97634 73220 codes572 = true := by decide
theorem ordered572 : Ordered 97634 73220 program572 := Coded.checked_ordered checked572
theorem length572 : program572.length = 128 := by
  rw [program572, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct572 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows572, row.Sat w) ↔ Satisfies program572 w := by
  rw [rows572, Codes.rows_expand]
  exact Coded.checked_correct checked572 h1

def rows573 := Exported.decodeRows ExportedData.chunk573.1 ExportedData.chunk573.2
def codes573 := Codes.rows ExportedData.chunk573.1 ExportedData.chunk573.2
def program573 := Coded.program 97634 73348 codes573
theorem checked573 : Coded.checkRows 97634 73348 codes573 = true := by decide
theorem ordered573 : Ordered 97634 73348 program573 := Coded.checked_ordered checked573
theorem length573 : program573.length = 128 := by
  rw [program573, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct573 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows573, row.Sat w) ↔ Satisfies program573 w := by
  rw [rows573, Codes.rows_expand]
  exact Coded.checked_correct checked573 h1

def rows574 := Exported.decodeRows ExportedData.chunk574.1 ExportedData.chunk574.2
def codes574 := Codes.rows ExportedData.chunk574.1 ExportedData.chunk574.2
def program574 := Coded.program 97634 73476 codes574
theorem checked574 : Coded.checkRows 97634 73476 codes574 = true := by decide
theorem ordered574 : Ordered 97634 73476 program574 := Coded.checked_ordered checked574
theorem length574 : program574.length = 128 := by
  rw [program574, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct574 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows574, row.Sat w) ↔ Satisfies program574 w := by
  rw [rows574, Codes.rows_expand]
  exact Coded.checked_correct checked574 h1

def rows575 := Exported.decodeRows ExportedData.chunk575.1 ExportedData.chunk575.2
def codes575 := Codes.rows ExportedData.chunk575.1 ExportedData.chunk575.2
def program575 := Coded.program 97634 73604 codes575
theorem checked575 : Coded.checkRows 97634 73604 codes575 = true := by decide
theorem ordered575 : Ordered 97634 73604 program575 := Coded.checked_ordered checked575
theorem length575 : program575.length = 128 := by
  rw [program575, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct575 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows575, row.Sat w) ↔ Satisfies program575 w := by
  rw [rows575, Codes.rows_expand]
  exact Coded.checked_correct checked575 h1

def rows576 := Exported.decodeRows ExportedData.chunk576.1 ExportedData.chunk576.2
def codes576 := Codes.rows ExportedData.chunk576.1 ExportedData.chunk576.2
def program576 := Coded.program 97634 73732 codes576
theorem checked576 : Coded.checkRows 97634 73732 codes576 = true := by decide
theorem ordered576 : Ordered 97634 73732 program576 := Coded.checked_ordered checked576
theorem length576 : program576.length = 128 := by
  rw [program576, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct576 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows576, row.Sat w) ↔ Satisfies program576 w := by
  rw [rows576, Codes.rows_expand]
  exact Coded.checked_correct checked576 h1

def rows577 := Exported.decodeRows ExportedData.chunk577.1 ExportedData.chunk577.2
def codes577 := Codes.rows ExportedData.chunk577.1 ExportedData.chunk577.2
def program577 := Coded.program 97634 73860 codes577
theorem checked577 : Coded.checkRows 97634 73860 codes577 = true := by decide
theorem ordered577 : Ordered 97634 73860 program577 := Coded.checked_ordered checked577
theorem length577 : program577.length = 128 := by
  rw [program577, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct577 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows577, row.Sat w) ↔ Satisfies program577 w := by
  rw [rows577, Codes.rows_expand]
  exact Coded.checked_correct checked577 h1

def rows578 := Exported.decodeRows ExportedData.chunk578.1 ExportedData.chunk578.2
def codes578 := Codes.rows ExportedData.chunk578.1 ExportedData.chunk578.2
def program578 := Coded.program 97634 73988 codes578
theorem checked578 : Coded.checkRows 97634 73988 codes578 = true := by decide
theorem ordered578 : Ordered 97634 73988 program578 := Coded.checked_ordered checked578
theorem length578 : program578.length = 128 := by
  rw [program578, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct578 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows578, row.Sat w) ↔ Satisfies program578 w := by
  rw [rows578, Codes.rows_expand]
  exact Coded.checked_correct checked578 h1

def rows579 := Exported.decodeRows ExportedData.chunk579.1 ExportedData.chunk579.2
def codes579 := Codes.rows ExportedData.chunk579.1 ExportedData.chunk579.2
def program579 := Coded.program 97634 74116 codes579
theorem checked579 : Coded.checkRows 97634 74116 codes579 = true := by decide
theorem ordered579 : Ordered 97634 74116 program579 := Coded.checked_ordered checked579
theorem length579 : program579.length = 128 := by
  rw [program579, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct579 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows579, row.Sat w) ↔ Satisfies program579 w := by
  rw [rows579, Codes.rows_expand]
  exact Coded.checked_correct checked579 h1

end CircuitCorrectness.ProgramCertificates
