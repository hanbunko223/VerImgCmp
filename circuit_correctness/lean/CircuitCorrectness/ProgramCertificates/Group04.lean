import CircuitCorrectness.ConcreteProgram
import CircuitCorrectness.ProgramCertificates.Group00

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows620 := Exported.decodeRows ExportedData.chunk620.1 ExportedData.chunk620.2
def codes620 := Codes.rows ExportedData.chunk620.1 ExportedData.chunk620.2
def program620 := Coded.program 97634 79364 codes620
theorem checked620 : Coded.checkRows 97634 79364 codes620 = true := by decide
theorem ordered620 : Ordered 97634 79364 program620 := Coded.checked_ordered checked620
theorem length620 : program620.length = 128 := by
  rw [program620, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct620 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows620, row.Sat w) ↔ Satisfies program620 w := by
  rw [rows620, Codes.rows_expand]
  exact Coded.checked_correct checked620 h1

def rows621 := Exported.decodeRows ExportedData.chunk621.1 ExportedData.chunk621.2
def codes621 := Codes.rows ExportedData.chunk621.1 ExportedData.chunk621.2
def program621 := Coded.program 97634 79492 codes621
theorem checked621 : Coded.checkRows 97634 79492 codes621 = true := by decide
theorem ordered621 : Ordered 97634 79492 program621 := Coded.checked_ordered checked621
theorem length621 : program621.length = 128 := by
  rw [program621, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct621 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows621, row.Sat w) ↔ Satisfies program621 w := by
  rw [rows621, Codes.rows_expand]
  exact Coded.checked_correct checked621 h1

def rows622 := Exported.decodeRows ExportedData.chunk622.1 ExportedData.chunk622.2
def codes622 := Codes.rows ExportedData.chunk622.1 ExportedData.chunk622.2
def program622 := Coded.program 97634 79620 codes622
theorem checked622 : Coded.checkRows 97634 79620 codes622 = true := by decide
theorem ordered622 : Ordered 97634 79620 program622 := Coded.checked_ordered checked622
theorem length622 : program622.length = 128 := by
  rw [program622, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct622 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows622, row.Sat w) ↔ Satisfies program622 w := by
  rw [rows622, Codes.rows_expand]
  exact Coded.checked_correct checked622 h1

def rows623 := Exported.decodeRows ExportedData.chunk623.1 ExportedData.chunk623.2
def codes623 := Codes.rows ExportedData.chunk623.1 ExportedData.chunk623.2
def program623 := Coded.program 97634 79748 codes623
theorem checked623 : Coded.checkRows 97634 79748 codes623 = true := by decide
theorem ordered623 : Ordered 97634 79748 program623 := Coded.checked_ordered checked623
theorem length623 : program623.length = 128 := by
  rw [program623, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct623 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows623, row.Sat w) ↔ Satisfies program623 w := by
  rw [rows623, Codes.rows_expand]
  exact Coded.checked_correct checked623 h1

def rows624 := Exported.decodeRows ExportedData.chunk624.1 ExportedData.chunk624.2
def codes624 := Codes.rows ExportedData.chunk624.1 ExportedData.chunk624.2
def program624 := Coded.program 97634 79876 codes624
theorem checked624 : Coded.checkRows 97634 79876 codes624 = true := by decide
theorem ordered624 : Ordered 97634 79876 program624 := Coded.checked_ordered checked624
theorem length624 : program624.length = 128 := by
  rw [program624, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct624 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows624, row.Sat w) ↔ Satisfies program624 w := by
  rw [rows624, Codes.rows_expand]
  exact Coded.checked_correct checked624 h1

def rows625 := Exported.decodeRows ExportedData.chunk625.1 ExportedData.chunk625.2
def codes625 := Codes.rows ExportedData.chunk625.1 ExportedData.chunk625.2
def program625 := Coded.program 97634 80004 codes625
theorem checked625 : Coded.checkRows 97634 80004 codes625 = true := by decide
theorem ordered625 : Ordered 97634 80004 program625 := Coded.checked_ordered checked625
theorem length625 : program625.length = 128 := by
  rw [program625, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct625 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows625, row.Sat w) ↔ Satisfies program625 w := by
  rw [rows625, Codes.rows_expand]
  exact Coded.checked_correct checked625 h1

def rows626 := Exported.decodeRows ExportedData.chunk626.1 ExportedData.chunk626.2
def codes626 := Codes.rows ExportedData.chunk626.1 ExportedData.chunk626.2
def program626 := Coded.program 97634 80132 codes626
theorem checked626 : Coded.checkRows 97634 80132 codes626 = true := by decide
theorem ordered626 : Ordered 97634 80132 program626 := Coded.checked_ordered checked626
theorem length626 : program626.length = 128 := by
  rw [program626, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct626 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows626, row.Sat w) ↔ Satisfies program626 w := by
  rw [rows626, Codes.rows_expand]
  exact Coded.checked_correct checked626 h1

def rows627 := Exported.decodeRows ExportedData.chunk627.1 ExportedData.chunk627.2
def codes627 := Codes.rows ExportedData.chunk627.1 ExportedData.chunk627.2
def program627 := Coded.program 97634 80260 codes627
theorem checked627 : Coded.checkRows 97634 80260 codes627 = true := by decide
theorem ordered627 : Ordered 97634 80260 program627 := Coded.checked_ordered checked627
theorem length627 : program627.length = 128 := by
  rw [program627, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct627 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows627, row.Sat w) ↔ Satisfies program627 w := by
  rw [rows627, Codes.rows_expand]
  exact Coded.checked_correct checked627 h1

def rows628 := Exported.decodeRows ExportedData.chunk628.1 ExportedData.chunk628.2
def codes628 := Codes.rows ExportedData.chunk628.1 ExportedData.chunk628.2
def program628 := Coded.program 97634 80388 codes628
theorem checked628 : Coded.checkRows 97634 80388 codes628 = true := by decide
theorem ordered628 : Ordered 97634 80388 program628 := Coded.checked_ordered checked628
theorem length628 : program628.length = 128 := by
  rw [program628, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct628 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows628, row.Sat w) ↔ Satisfies program628 w := by
  rw [rows628, Codes.rows_expand]
  exact Coded.checked_correct checked628 h1

def rows629 := Exported.decodeRows ExportedData.chunk629.1 ExportedData.chunk629.2
def codes629 := Codes.rows ExportedData.chunk629.1 ExportedData.chunk629.2
def program629 := Coded.program 97634 80516 codes629
theorem checked629 : Coded.checkRows 97634 80516 codes629 = true := by decide
theorem ordered629 : Ordered 97634 80516 program629 := Coded.checked_ordered checked629
theorem length629 : program629.length = 128 := by
  rw [program629, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct629 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows629, row.Sat w) ↔ Satisfies program629 w := by
  rw [rows629, Codes.rows_expand]
  exact Coded.checked_correct checked629 h1

def rows630 := Exported.decodeRows ExportedData.chunk630.1 ExportedData.chunk630.2
def codes630 := Codes.rows ExportedData.chunk630.1 ExportedData.chunk630.2
def program630 := Coded.program 97634 80644 codes630
theorem checked630 : Coded.checkRows 97634 80644 codes630 = true := by decide
theorem ordered630 : Ordered 97634 80644 program630 := Coded.checked_ordered checked630
theorem length630 : program630.length = 128 := by
  rw [program630, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct630 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows630, row.Sat w) ↔ Satisfies program630 w := by
  rw [rows630, Codes.rows_expand]
  exact Coded.checked_correct checked630 h1

def rows631 := Exported.decodeRows ExportedData.chunk631.1 ExportedData.chunk631.2
def codes631 := Codes.rows ExportedData.chunk631.1 ExportedData.chunk631.2
def program631 := Coded.program 97634 80772 codes631
theorem checked631 : Coded.checkRows 97634 80772 codes631 = true := by decide
theorem ordered631 : Ordered 97634 80772 program631 := Coded.checked_ordered checked631
theorem length631 : program631.length = 128 := by
  rw [program631, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct631 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows631, row.Sat w) ↔ Satisfies program631 w := by
  rw [rows631, Codes.rows_expand]
  exact Coded.checked_correct checked631 h1

def rows632 := Exported.decodeRows ExportedData.chunk632.1 ExportedData.chunk632.2
def codes632 := Codes.rows ExportedData.chunk632.1 ExportedData.chunk632.2
def program632 := Coded.program 97634 80900 codes632
theorem checked632 : Coded.checkRows 97634 80900 codes632 = true := by decide
theorem ordered632 : Ordered 97634 80900 program632 := Coded.checked_ordered checked632
theorem length632 : program632.length = 128 := by
  rw [program632, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct632 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows632, row.Sat w) ↔ Satisfies program632 w := by
  rw [rows632, Codes.rows_expand]
  exact Coded.checked_correct checked632 h1

def rows633 := Exported.decodeRows ExportedData.chunk633.1 ExportedData.chunk633.2
def codes633 := Codes.rows ExportedData.chunk633.1 ExportedData.chunk633.2
def program633 := Coded.program 97634 81028 codes633
theorem checked633 : Coded.checkRows 97634 81028 codes633 = true := by decide
theorem ordered633 : Ordered 97634 81028 program633 := Coded.checked_ordered checked633
theorem length633 : program633.length = 128 := by
  rw [program633, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct633 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows633, row.Sat w) ↔ Satisfies program633 w := by
  rw [rows633, Codes.rows_expand]
  exact Coded.checked_correct checked633 h1

def rows634 := Exported.decodeRows ExportedData.chunk634.1 ExportedData.chunk634.2
def codes634 := Codes.rows ExportedData.chunk634.1 ExportedData.chunk634.2
def program634 := Coded.program 97634 81156 codes634
theorem checked634 : Coded.checkRows 97634 81156 codes634 = true := by decide
theorem ordered634 : Ordered 97634 81156 program634 := Coded.checked_ordered checked634
theorem length634 : program634.length = 128 := by
  rw [program634, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct634 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows634, row.Sat w) ↔ Satisfies program634 w := by
  rw [rows634, Codes.rows_expand]
  exact Coded.checked_correct checked634 h1

def rows635 := Exported.decodeRows ExportedData.chunk635.1 ExportedData.chunk635.2
def codes635 := Codes.rows ExportedData.chunk635.1 ExportedData.chunk635.2
def program635 := Coded.program 97634 81284 codes635
theorem checked635 : Coded.checkRows 97634 81284 codes635 = true := by decide
theorem ordered635 : Ordered 97634 81284 program635 := Coded.checked_ordered checked635
theorem length635 : program635.length = 128 := by
  rw [program635, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct635 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows635, row.Sat w) ↔ Satisfies program635 w := by
  rw [rows635, Codes.rows_expand]
  exact Coded.checked_correct checked635 h1

def rows636 := Exported.decodeRows ExportedData.chunk636.1 ExportedData.chunk636.2
def codes636 := Codes.rows ExportedData.chunk636.1 ExportedData.chunk636.2
def program636 := Coded.program 97634 81412 codes636
theorem checked636 : Coded.checkRows 97634 81412 codes636 = true := by decide
theorem ordered636 : Ordered 97634 81412 program636 := Coded.checked_ordered checked636
theorem length636 : program636.length = 128 := by
  rw [program636, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct636 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows636, row.Sat w) ↔ Satisfies program636 w := by
  rw [rows636, Codes.rows_expand]
  exact Coded.checked_correct checked636 h1

def rows637 := Exported.decodeRows ExportedData.chunk637.1 ExportedData.chunk637.2
def codes637 := Codes.rows ExportedData.chunk637.1 ExportedData.chunk637.2
def program637 := Coded.program 97634 81540 codes637
theorem checked637 : Coded.checkRows 97634 81540 codes637 = true := by decide
theorem ordered637 : Ordered 97634 81540 program637 := Coded.checked_ordered checked637
theorem length637 : program637.length = 128 := by
  rw [program637, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct637 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows637, row.Sat w) ↔ Satisfies program637 w := by
  rw [rows637, Codes.rows_expand]
  exact Coded.checked_correct checked637 h1

def rows638 := Exported.decodeRows ExportedData.chunk638.1 ExportedData.chunk638.2
def codes638 := Codes.rows ExportedData.chunk638.1 ExportedData.chunk638.2
def program638 := Coded.program 97634 81668 codes638
theorem checked638 : Coded.checkRows 97634 81668 codes638 = true := by decide
theorem ordered638 : Ordered 97634 81668 program638 := Coded.checked_ordered checked638
theorem length638 : program638.length = 128 := by
  rw [program638, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct638 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows638, row.Sat w) ↔ Satisfies program638 w := by
  rw [rows638, Codes.rows_expand]
  exact Coded.checked_correct checked638 h1

def rows639 := Exported.decodeRows ExportedData.chunk639.1 ExportedData.chunk639.2
def codes639 := Codes.rows ExportedData.chunk639.1 ExportedData.chunk639.2
def program639 := Coded.program 97634 81796 codes639
theorem checked639 : Coded.checkRows 97634 81796 codes639 = true := by decide
theorem ordered639 : Ordered 97634 81796 program639 := Coded.checked_ordered checked639
theorem length639 : program639.length = 128 := by
  rw [program639, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct639 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows639, row.Sat w) ↔ Satisfies program639 w := by
  rw [rows639, Codes.rows_expand]
  exact Coded.checked_correct checked639 h1

end CircuitCorrectness.ProgramCertificates
