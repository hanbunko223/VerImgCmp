import CircuitCorrectness.ConcreteProgram
import CircuitCorrectness.ProgramCertificates.Group01

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows640 := Exported.decodeRows ExportedData.chunk640.1 ExportedData.chunk640.2
def codes640 := Codes.rows ExportedData.chunk640.1 ExportedData.chunk640.2
def program640 := Coded.program 97634 81924 codes640
theorem checked640 : Coded.checkRows 97634 81924 codes640 = true := by decide
theorem ordered640 : Ordered 97634 81924 program640 := Coded.checked_ordered checked640
theorem length640 : program640.length = 128 := by
  rw [program640, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct640 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows640, row.Sat w) ↔ Satisfies program640 w := by
  rw [rows640, Codes.rows_expand]
  exact Coded.checked_correct checked640 h1

def rows641 := Exported.decodeRows ExportedData.chunk641.1 ExportedData.chunk641.2
def codes641 := Codes.rows ExportedData.chunk641.1 ExportedData.chunk641.2
def program641 := Coded.program 97634 82052 codes641
theorem checked641 : Coded.checkRows 97634 82052 codes641 = true := by decide
theorem ordered641 : Ordered 97634 82052 program641 := Coded.checked_ordered checked641
theorem length641 : program641.length = 128 := by
  rw [program641, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct641 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows641, row.Sat w) ↔ Satisfies program641 w := by
  rw [rows641, Codes.rows_expand]
  exact Coded.checked_correct checked641 h1

def rows642 := Exported.decodeRows ExportedData.chunk642.1 ExportedData.chunk642.2
def codes642 := Codes.rows ExportedData.chunk642.1 ExportedData.chunk642.2
def program642 := Coded.program 97634 82180 codes642
theorem checked642 : Coded.checkRows 97634 82180 codes642 = true := by decide
theorem ordered642 : Ordered 97634 82180 program642 := Coded.checked_ordered checked642
theorem length642 : program642.length = 128 := by
  rw [program642, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct642 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows642, row.Sat w) ↔ Satisfies program642 w := by
  rw [rows642, Codes.rows_expand]
  exact Coded.checked_correct checked642 h1

def rows643 := Exported.decodeRows ExportedData.chunk643.1 ExportedData.chunk643.2
def codes643 := Codes.rows ExportedData.chunk643.1 ExportedData.chunk643.2
def program643 := Coded.program 97634 82308 codes643
theorem checked643 : Coded.checkRows 97634 82308 codes643 = true := by decide
theorem ordered643 : Ordered 97634 82308 program643 := Coded.checked_ordered checked643
theorem length643 : program643.length = 128 := by
  rw [program643, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct643 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows643, row.Sat w) ↔ Satisfies program643 w := by
  rw [rows643, Codes.rows_expand]
  exact Coded.checked_correct checked643 h1

def rows644 := Exported.decodeRows ExportedData.chunk644.1 ExportedData.chunk644.2
def codes644 := Codes.rows ExportedData.chunk644.1 ExportedData.chunk644.2
def program644 := Coded.program 97634 82436 codes644
theorem checked644 : Coded.checkRows 97634 82436 codes644 = true := by decide
theorem ordered644 : Ordered 97634 82436 program644 := Coded.checked_ordered checked644
theorem length644 : program644.length = 128 := by
  rw [program644, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct644 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows644, row.Sat w) ↔ Satisfies program644 w := by
  rw [rows644, Codes.rows_expand]
  exact Coded.checked_correct checked644 h1

def rows645 := Exported.decodeRows ExportedData.chunk645.1 ExportedData.chunk645.2
def codes645 := Codes.rows ExportedData.chunk645.1 ExportedData.chunk645.2
def program645 := Coded.program 97634 82564 codes645
theorem checked645 : Coded.checkRows 97634 82564 codes645 = true := by decide
theorem ordered645 : Ordered 97634 82564 program645 := Coded.checked_ordered checked645
theorem length645 : program645.length = 128 := by
  rw [program645, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct645 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows645, row.Sat w) ↔ Satisfies program645 w := by
  rw [rows645, Codes.rows_expand]
  exact Coded.checked_correct checked645 h1

def rows646 := Exported.decodeRows ExportedData.chunk646.1 ExportedData.chunk646.2
def codes646 := Codes.rows ExportedData.chunk646.1 ExportedData.chunk646.2
def program646 := Coded.program 97634 82692 codes646
theorem checked646 : Coded.checkRows 97634 82692 codes646 = true := by decide
theorem ordered646 : Ordered 97634 82692 program646 := Coded.checked_ordered checked646
theorem length646 : program646.length = 128 := by
  rw [program646, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct646 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows646, row.Sat w) ↔ Satisfies program646 w := by
  rw [rows646, Codes.rows_expand]
  exact Coded.checked_correct checked646 h1

def rows647 := Exported.decodeRows ExportedData.chunk647.1 ExportedData.chunk647.2
def codes647 := Codes.rows ExportedData.chunk647.1 ExportedData.chunk647.2
def program647 := Coded.program 97634 82820 codes647
theorem checked647 : Coded.checkRows 97634 82820 codes647 = true := by decide
theorem ordered647 : Ordered 97634 82820 program647 := Coded.checked_ordered checked647
theorem length647 : program647.length = 128 := by
  rw [program647, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct647 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows647, row.Sat w) ↔ Satisfies program647 w := by
  rw [rows647, Codes.rows_expand]
  exact Coded.checked_correct checked647 h1

def rows648 := Exported.decodeRows ExportedData.chunk648.1 ExportedData.chunk648.2
def codes648 := Codes.rows ExportedData.chunk648.1 ExportedData.chunk648.2
def program648 := Coded.program 97634 82948 codes648
theorem checked648 : Coded.checkRows 97634 82948 codes648 = true := by decide
theorem ordered648 : Ordered 97634 82948 program648 := Coded.checked_ordered checked648
theorem length648 : program648.length = 128 := by
  rw [program648, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct648 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows648, row.Sat w) ↔ Satisfies program648 w := by
  rw [rows648, Codes.rows_expand]
  exact Coded.checked_correct checked648 h1

def rows649 := Exported.decodeRows ExportedData.chunk649.1 ExportedData.chunk649.2
def codes649 := Codes.rows ExportedData.chunk649.1 ExportedData.chunk649.2
def program649 := Coded.program 97634 83076 codes649
theorem checked649 : Coded.checkRows 97634 83076 codes649 = true := by decide
theorem ordered649 : Ordered 97634 83076 program649 := Coded.checked_ordered checked649
theorem length649 : program649.length = 128 := by
  rw [program649, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct649 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows649, row.Sat w) ↔ Satisfies program649 w := by
  rw [rows649, Codes.rows_expand]
  exact Coded.checked_correct checked649 h1

def rows650 := Exported.decodeRows ExportedData.chunk650.1 ExportedData.chunk650.2
def codes650 := Codes.rows ExportedData.chunk650.1 ExportedData.chunk650.2
def program650 := Coded.program 97634 83204 codes650
theorem checked650 : Coded.checkRows 97634 83204 codes650 = true := by decide
theorem ordered650 : Ordered 97634 83204 program650 := Coded.checked_ordered checked650
theorem length650 : program650.length = 128 := by
  rw [program650, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct650 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows650, row.Sat w) ↔ Satisfies program650 w := by
  rw [rows650, Codes.rows_expand]
  exact Coded.checked_correct checked650 h1

def rows651 := Exported.decodeRows ExportedData.chunk651.1 ExportedData.chunk651.2
def codes651 := Codes.rows ExportedData.chunk651.1 ExportedData.chunk651.2
def program651 := Coded.program 97634 83332 codes651
theorem checked651 : Coded.checkRows 97634 83332 codes651 = true := by decide
theorem ordered651 : Ordered 97634 83332 program651 := Coded.checked_ordered checked651
theorem length651 : program651.length = 128 := by
  rw [program651, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct651 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows651, row.Sat w) ↔ Satisfies program651 w := by
  rw [rows651, Codes.rows_expand]
  exact Coded.checked_correct checked651 h1

def rows652 := Exported.decodeRows ExportedData.chunk652.1 ExportedData.chunk652.2
def codes652 := Codes.rows ExportedData.chunk652.1 ExportedData.chunk652.2
def program652 := Coded.program 97634 83460 codes652
theorem checked652 : Coded.checkRows 97634 83460 codes652 = true := by decide
theorem ordered652 : Ordered 97634 83460 program652 := Coded.checked_ordered checked652
theorem length652 : program652.length = 128 := by
  rw [program652, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct652 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows652, row.Sat w) ↔ Satisfies program652 w := by
  rw [rows652, Codes.rows_expand]
  exact Coded.checked_correct checked652 h1

def rows653 := Exported.decodeRows ExportedData.chunk653.1 ExportedData.chunk653.2
def codes653 := Codes.rows ExportedData.chunk653.1 ExportedData.chunk653.2
def program653 := Coded.program 97634 83588 codes653
theorem checked653 : Coded.checkRows 97634 83588 codes653 = true := by decide
theorem ordered653 : Ordered 97634 83588 program653 := Coded.checked_ordered checked653
theorem length653 : program653.length = 128 := by
  rw [program653, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct653 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows653, row.Sat w) ↔ Satisfies program653 w := by
  rw [rows653, Codes.rows_expand]
  exact Coded.checked_correct checked653 h1

def rows654 := Exported.decodeRows ExportedData.chunk654.1 ExportedData.chunk654.2
def codes654 := Codes.rows ExportedData.chunk654.1 ExportedData.chunk654.2
def program654 := Coded.program 97634 83716 codes654
theorem checked654 : Coded.checkRows 97634 83716 codes654 = true := by decide
theorem ordered654 : Ordered 97634 83716 program654 := Coded.checked_ordered checked654
theorem length654 : program654.length = 128 := by
  rw [program654, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct654 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows654, row.Sat w) ↔ Satisfies program654 w := by
  rw [rows654, Codes.rows_expand]
  exact Coded.checked_correct checked654 h1

def rows655 := Exported.decodeRows ExportedData.chunk655.1 ExportedData.chunk655.2
def codes655 := Codes.rows ExportedData.chunk655.1 ExportedData.chunk655.2
def program655 := Coded.program 97634 83844 codes655
theorem checked655 : Coded.checkRows 97634 83844 codes655 = true := by decide
theorem ordered655 : Ordered 97634 83844 program655 := Coded.checked_ordered checked655
theorem length655 : program655.length = 128 := by
  rw [program655, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct655 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows655, row.Sat w) ↔ Satisfies program655 w := by
  rw [rows655, Codes.rows_expand]
  exact Coded.checked_correct checked655 h1

def rows656 := Exported.decodeRows ExportedData.chunk656.1 ExportedData.chunk656.2
def codes656 := Codes.rows ExportedData.chunk656.1 ExportedData.chunk656.2
def program656 := Coded.program 97634 83972 codes656
theorem checked656 : Coded.checkRows 97634 83972 codes656 = true := by decide
theorem ordered656 : Ordered 97634 83972 program656 := Coded.checked_ordered checked656
theorem length656 : program656.length = 128 := by
  rw [program656, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct656 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows656, row.Sat w) ↔ Satisfies program656 w := by
  rw [rows656, Codes.rows_expand]
  exact Coded.checked_correct checked656 h1

def rows657 := Exported.decodeRows ExportedData.chunk657.1 ExportedData.chunk657.2
def codes657 := Codes.rows ExportedData.chunk657.1 ExportedData.chunk657.2
def program657 := Coded.program 97634 84100 codes657
theorem checked657 : Coded.checkRows 97634 84100 codes657 = true := by decide
theorem ordered657 : Ordered 97634 84100 program657 := Coded.checked_ordered checked657
theorem length657 : program657.length = 128 := by
  rw [program657, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct657 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows657, row.Sat w) ↔ Satisfies program657 w := by
  rw [rows657, Codes.rows_expand]
  exact Coded.checked_correct checked657 h1

def rows658 := Exported.decodeRows ExportedData.chunk658.1 ExportedData.chunk658.2
def codes658 := Codes.rows ExportedData.chunk658.1 ExportedData.chunk658.2
def program658 := Coded.program 97634 84228 codes658
theorem checked658 : Coded.checkRows 97634 84228 codes658 = true := by decide
theorem ordered658 : Ordered 97634 84228 program658 := Coded.checked_ordered checked658
theorem length658 : program658.length = 128 := by
  rw [program658, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct658 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows658, row.Sat w) ↔ Satisfies program658 w := by
  rw [rows658, Codes.rows_expand]
  exact Coded.checked_correct checked658 h1

def rows659 := Exported.decodeRows ExportedData.chunk659.1 ExportedData.chunk659.2
def codes659 := Codes.rows ExportedData.chunk659.1 ExportedData.chunk659.2
def program659 := Coded.program 97634 84356 codes659
theorem checked659 : Coded.checkRows 97634 84356 codes659 = true := by decide
theorem ordered659 : Ordered 97634 84356 program659 := Coded.checked_ordered checked659
theorem length659 : program659.length = 128 := by
  rw [program659, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct659 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows659, row.Sat w) ↔ Satisfies program659 w := by
  rw [rows659, Codes.rows_expand]
  exact Coded.checked_correct checked659 h1

end CircuitCorrectness.ProgramCertificates
