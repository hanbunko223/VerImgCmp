import CircuitCorrectness.ConcreteProgram
import CircuitCorrectness.ProgramCertificates.Group02

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows660 := Exported.decodeRows ExportedData.chunk660.1 ExportedData.chunk660.2
def codes660 := Codes.rows ExportedData.chunk660.1 ExportedData.chunk660.2
def program660 := Coded.program 97634 84484 codes660
theorem checked660 : Coded.checkRows 97634 84484 codes660 = true := by decide
theorem ordered660 : Ordered 97634 84484 program660 := Coded.checked_ordered checked660
theorem length660 : program660.length = 128 := by
  rw [program660, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct660 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows660, row.Sat w) ↔ Satisfies program660 w := by
  rw [rows660, Codes.rows_expand]
  exact Coded.checked_correct checked660 h1

def rows661 := Exported.decodeRows ExportedData.chunk661.1 ExportedData.chunk661.2
def codes661 := Codes.rows ExportedData.chunk661.1 ExportedData.chunk661.2
def program661 := Coded.program 97634 84612 codes661
theorem checked661 : Coded.checkRows 97634 84612 codes661 = true := by decide
theorem ordered661 : Ordered 97634 84612 program661 := Coded.checked_ordered checked661
theorem length661 : program661.length = 128 := by
  rw [program661, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct661 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows661, row.Sat w) ↔ Satisfies program661 w := by
  rw [rows661, Codes.rows_expand]
  exact Coded.checked_correct checked661 h1

def rows662 := Exported.decodeRows ExportedData.chunk662.1 ExportedData.chunk662.2
def codes662 := Codes.rows ExportedData.chunk662.1 ExportedData.chunk662.2
def program662 := Coded.program 97634 84740 codes662
theorem checked662 : Coded.checkRows 97634 84740 codes662 = true := by decide
theorem ordered662 : Ordered 97634 84740 program662 := Coded.checked_ordered checked662
theorem length662 : program662.length = 128 := by
  rw [program662, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct662 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows662, row.Sat w) ↔ Satisfies program662 w := by
  rw [rows662, Codes.rows_expand]
  exact Coded.checked_correct checked662 h1

def rows663 := Exported.decodeRows ExportedData.chunk663.1 ExportedData.chunk663.2
def codes663 := Codes.rows ExportedData.chunk663.1 ExportedData.chunk663.2
def program663 := Coded.program 97634 84868 codes663
theorem checked663 : Coded.checkRows 97634 84868 codes663 = true := by decide
theorem ordered663 : Ordered 97634 84868 program663 := Coded.checked_ordered checked663
theorem length663 : program663.length = 128 := by
  rw [program663, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct663 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows663, row.Sat w) ↔ Satisfies program663 w := by
  rw [rows663, Codes.rows_expand]
  exact Coded.checked_correct checked663 h1

def rows664 := Exported.decodeRows ExportedData.chunk664.1 ExportedData.chunk664.2
def codes664 := Codes.rows ExportedData.chunk664.1 ExportedData.chunk664.2
def program664 := Coded.program 97634 84996 codes664
theorem checked664 : Coded.checkRows 97634 84996 codes664 = true := by decide
theorem ordered664 : Ordered 97634 84996 program664 := Coded.checked_ordered checked664
theorem length664 : program664.length = 128 := by
  rw [program664, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct664 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows664, row.Sat w) ↔ Satisfies program664 w := by
  rw [rows664, Codes.rows_expand]
  exact Coded.checked_correct checked664 h1

def rows665 := Exported.decodeRows ExportedData.chunk665.1 ExportedData.chunk665.2
def codes665 := Codes.rows ExportedData.chunk665.1 ExportedData.chunk665.2
def program665 := Coded.program 97634 85124 codes665
theorem checked665 : Coded.checkRows 97634 85124 codes665 = true := by decide
theorem ordered665 : Ordered 97634 85124 program665 := Coded.checked_ordered checked665
theorem length665 : program665.length = 128 := by
  rw [program665, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct665 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows665, row.Sat w) ↔ Satisfies program665 w := by
  rw [rows665, Codes.rows_expand]
  exact Coded.checked_correct checked665 h1

def rows666 := Exported.decodeRows ExportedData.chunk666.1 ExportedData.chunk666.2
def codes666 := Codes.rows ExportedData.chunk666.1 ExportedData.chunk666.2
def program666 := Coded.program 97634 85252 codes666
theorem checked666 : Coded.checkRows 97634 85252 codes666 = true := by decide
theorem ordered666 : Ordered 97634 85252 program666 := Coded.checked_ordered checked666
theorem length666 : program666.length = 128 := by
  rw [program666, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct666 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows666, row.Sat w) ↔ Satisfies program666 w := by
  rw [rows666, Codes.rows_expand]
  exact Coded.checked_correct checked666 h1

def rows667 := Exported.decodeRows ExportedData.chunk667.1 ExportedData.chunk667.2
def codes667 := Codes.rows ExportedData.chunk667.1 ExportedData.chunk667.2
def program667 := Coded.program 97634 85380 codes667
theorem checked667 : Coded.checkRows 97634 85380 codes667 = true := by decide
theorem ordered667 : Ordered 97634 85380 program667 := Coded.checked_ordered checked667
theorem length667 : program667.length = 128 := by
  rw [program667, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct667 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows667, row.Sat w) ↔ Satisfies program667 w := by
  rw [rows667, Codes.rows_expand]
  exact Coded.checked_correct checked667 h1

def rows668 := Exported.decodeRows ExportedData.chunk668.1 ExportedData.chunk668.2
def codes668 := Codes.rows ExportedData.chunk668.1 ExportedData.chunk668.2
def program668 := Coded.program 97634 85508 codes668
theorem checked668 : Coded.checkRows 97634 85508 codes668 = true := by decide
theorem ordered668 : Ordered 97634 85508 program668 := Coded.checked_ordered checked668
theorem length668 : program668.length = 128 := by
  rw [program668, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct668 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows668, row.Sat w) ↔ Satisfies program668 w := by
  rw [rows668, Codes.rows_expand]
  exact Coded.checked_correct checked668 h1

def rows669 := Exported.decodeRows ExportedData.chunk669.1 ExportedData.chunk669.2
def codes669 := Codes.rows ExportedData.chunk669.1 ExportedData.chunk669.2
def program669 := Coded.program 97634 85636 codes669
theorem checked669 : Coded.checkRows 97634 85636 codes669 = true := by decide
theorem ordered669 : Ordered 97634 85636 program669 := Coded.checked_ordered checked669
theorem length669 : program669.length = 128 := by
  rw [program669, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct669 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows669, row.Sat w) ↔ Satisfies program669 w := by
  rw [rows669, Codes.rows_expand]
  exact Coded.checked_correct checked669 h1

def rows670 := Exported.decodeRows ExportedData.chunk670.1 ExportedData.chunk670.2
def codes670 := Codes.rows ExportedData.chunk670.1 ExportedData.chunk670.2
def program670 := Coded.program 97634 85764 codes670
theorem checked670 : Coded.checkRows 97634 85764 codes670 = true := by decide
theorem ordered670 : Ordered 97634 85764 program670 := Coded.checked_ordered checked670
theorem length670 : program670.length = 128 := by
  rw [program670, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct670 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows670, row.Sat w) ↔ Satisfies program670 w := by
  rw [rows670, Codes.rows_expand]
  exact Coded.checked_correct checked670 h1

def rows671 := Exported.decodeRows ExportedData.chunk671.1 ExportedData.chunk671.2
def codes671 := Codes.rows ExportedData.chunk671.1 ExportedData.chunk671.2
def program671 := Coded.program 97634 85892 codes671
theorem checked671 : Coded.checkRows 97634 85892 codes671 = true := by decide
theorem ordered671 : Ordered 97634 85892 program671 := Coded.checked_ordered checked671
theorem length671 : program671.length = 128 := by
  rw [program671, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct671 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows671, row.Sat w) ↔ Satisfies program671 w := by
  rw [rows671, Codes.rows_expand]
  exact Coded.checked_correct checked671 h1

def rows672 := Exported.decodeRows ExportedData.chunk672.1 ExportedData.chunk672.2
def codes672 := Codes.rows ExportedData.chunk672.1 ExportedData.chunk672.2
def program672 := Coded.program 97634 86020 codes672
theorem checked672 : Coded.checkRows 97634 86020 codes672 = true := by decide
theorem ordered672 : Ordered 97634 86020 program672 := Coded.checked_ordered checked672
theorem length672 : program672.length = 128 := by
  rw [program672, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct672 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows672, row.Sat w) ↔ Satisfies program672 w := by
  rw [rows672, Codes.rows_expand]
  exact Coded.checked_correct checked672 h1

def rows673 := Exported.decodeRows ExportedData.chunk673.1 ExportedData.chunk673.2
def codes673 := Codes.rows ExportedData.chunk673.1 ExportedData.chunk673.2
def program673 := Coded.program 97634 86148 codes673
theorem checked673 : Coded.checkRows 97634 86148 codes673 = true := by decide
theorem ordered673 : Ordered 97634 86148 program673 := Coded.checked_ordered checked673
theorem length673 : program673.length = 128 := by
  rw [program673, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct673 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows673, row.Sat w) ↔ Satisfies program673 w := by
  rw [rows673, Codes.rows_expand]
  exact Coded.checked_correct checked673 h1

def rows674 := Exported.decodeRows ExportedData.chunk674.1 ExportedData.chunk674.2
def codes674 := Codes.rows ExportedData.chunk674.1 ExportedData.chunk674.2
def program674 := Coded.program 97634 86276 codes674
theorem checked674 : Coded.checkRows 97634 86276 codes674 = true := by decide
theorem ordered674 : Ordered 97634 86276 program674 := Coded.checked_ordered checked674
theorem length674 : program674.length = 128 := by
  rw [program674, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct674 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows674, row.Sat w) ↔ Satisfies program674 w := by
  rw [rows674, Codes.rows_expand]
  exact Coded.checked_correct checked674 h1

def rows675 := Exported.decodeRows ExportedData.chunk675.1 ExportedData.chunk675.2
def codes675 := Codes.rows ExportedData.chunk675.1 ExportedData.chunk675.2
def program675 := Coded.program 97634 86404 codes675
theorem checked675 : Coded.checkRows 97634 86404 codes675 = true := by decide
theorem ordered675 : Ordered 97634 86404 program675 := Coded.checked_ordered checked675
theorem length675 : program675.length = 128 := by
  rw [program675, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct675 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows675, row.Sat w) ↔ Satisfies program675 w := by
  rw [rows675, Codes.rows_expand]
  exact Coded.checked_correct checked675 h1

def rows676 := Exported.decodeRows ExportedData.chunk676.1 ExportedData.chunk676.2
def codes676 := Codes.rows ExportedData.chunk676.1 ExportedData.chunk676.2
def program676 := Coded.program 97634 86532 codes676
theorem checked676 : Coded.checkRows 97634 86532 codes676 = true := by decide
theorem ordered676 : Ordered 97634 86532 program676 := Coded.checked_ordered checked676
theorem length676 : program676.length = 128 := by
  rw [program676, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct676 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows676, row.Sat w) ↔ Satisfies program676 w := by
  rw [rows676, Codes.rows_expand]
  exact Coded.checked_correct checked676 h1

def rows677 := Exported.decodeRows ExportedData.chunk677.1 ExportedData.chunk677.2
def codes677 := Codes.rows ExportedData.chunk677.1 ExportedData.chunk677.2
def program677 := Coded.program 97634 86660 codes677
theorem checked677 : Coded.checkRows 97634 86660 codes677 = true := by decide
theorem ordered677 : Ordered 97634 86660 program677 := Coded.checked_ordered checked677
theorem length677 : program677.length = 128 := by
  rw [program677, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct677 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows677, row.Sat w) ↔ Satisfies program677 w := by
  rw [rows677, Codes.rows_expand]
  exact Coded.checked_correct checked677 h1

def rows678 := Exported.decodeRows ExportedData.chunk678.1 ExportedData.chunk678.2
def codes678 := Codes.rows ExportedData.chunk678.1 ExportedData.chunk678.2
def program678 := Coded.program 97634 86788 codes678
theorem checked678 : Coded.checkRows 97634 86788 codes678 = true := by decide
theorem ordered678 : Ordered 97634 86788 program678 := Coded.checked_ordered checked678
theorem length678 : program678.length = 128 := by
  rw [program678, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct678 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows678, row.Sat w) ↔ Satisfies program678 w := by
  rw [rows678, Codes.rows_expand]
  exact Coded.checked_correct checked678 h1

def rows679 := Exported.decodeRows ExportedData.chunk679.1 ExportedData.chunk679.2
def codes679 := Codes.rows ExportedData.chunk679.1 ExportedData.chunk679.2
def program679 := Coded.program 97634 86916 codes679
theorem checked679 : Coded.checkRows 97634 86916 codes679 = true := by decide
theorem ordered679 : Ordered 97634 86916 program679 := Coded.checked_ordered checked679
theorem length679 : program679.length = 128 := by
  rw [program679, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct679 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows679, row.Sat w) ↔ Satisfies program679 w := by
  rw [rows679, Codes.rows_expand]
  exact Coded.checked_correct checked679 h1

end CircuitCorrectness.ProgramCertificates
