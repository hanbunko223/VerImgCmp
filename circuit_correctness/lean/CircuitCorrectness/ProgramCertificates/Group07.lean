import CircuitCorrectness.ConcreteProgram
import CircuitCorrectness.ProgramCertificates.Group03

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows680 := Exported.decodeRows ExportedData.chunk680.1 ExportedData.chunk680.2
def codes680 := Codes.rows ExportedData.chunk680.1 ExportedData.chunk680.2
def program680 := Coded.program 97634 87044 codes680
theorem checked680 : Coded.checkRows 97634 87044 codes680 = true := by decide
theorem ordered680 : Ordered 97634 87044 program680 := Coded.checked_ordered checked680
theorem length680 : program680.length = 128 := by
  rw [program680, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct680 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows680, row.Sat w) ↔ Satisfies program680 w := by
  rw [rows680, Codes.rows_expand]
  exact Coded.checked_correct checked680 h1

def rows681 := Exported.decodeRows ExportedData.chunk681.1 ExportedData.chunk681.2
def codes681 := Codes.rows ExportedData.chunk681.1 ExportedData.chunk681.2
def program681 := Coded.program 97634 87172 codes681
theorem checked681 : Coded.checkRows 97634 87172 codes681 = true := by decide
theorem ordered681 : Ordered 97634 87172 program681 := Coded.checked_ordered checked681
theorem length681 : program681.length = 128 := by
  rw [program681, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct681 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows681, row.Sat w) ↔ Satisfies program681 w := by
  rw [rows681, Codes.rows_expand]
  exact Coded.checked_correct checked681 h1

def rows682 := Exported.decodeRows ExportedData.chunk682.1 ExportedData.chunk682.2
def codes682 := Codes.rows ExportedData.chunk682.1 ExportedData.chunk682.2
def program682 := Coded.program 97634 87300 codes682
theorem checked682 : Coded.checkRows 97634 87300 codes682 = true := by decide
theorem ordered682 : Ordered 97634 87300 program682 := Coded.checked_ordered checked682
theorem length682 : program682.length = 128 := by
  rw [program682, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct682 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows682, row.Sat w) ↔ Satisfies program682 w := by
  rw [rows682, Codes.rows_expand]
  exact Coded.checked_correct checked682 h1

def rows683 := Exported.decodeRows ExportedData.chunk683.1 ExportedData.chunk683.2
def codes683 := Codes.rows ExportedData.chunk683.1 ExportedData.chunk683.2
def program683 := Coded.program 97634 87428 codes683
theorem checked683 : Coded.checkRows 97634 87428 codes683 = true := by decide
theorem ordered683 : Ordered 97634 87428 program683 := Coded.checked_ordered checked683
theorem length683 : program683.length = 128 := by
  rw [program683, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct683 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows683, row.Sat w) ↔ Satisfies program683 w := by
  rw [rows683, Codes.rows_expand]
  exact Coded.checked_correct checked683 h1

def rows684 := Exported.decodeRows ExportedData.chunk684.1 ExportedData.chunk684.2
def codes684 := Codes.rows ExportedData.chunk684.1 ExportedData.chunk684.2
def program684 := Coded.program 97634 87556 codes684
theorem checked684 : Coded.checkRows 97634 87556 codes684 = true := by decide
theorem ordered684 : Ordered 97634 87556 program684 := Coded.checked_ordered checked684
theorem length684 : program684.length = 128 := by
  rw [program684, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct684 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows684, row.Sat w) ↔ Satisfies program684 w := by
  rw [rows684, Codes.rows_expand]
  exact Coded.checked_correct checked684 h1

def rows685 := Exported.decodeRows ExportedData.chunk685.1 ExportedData.chunk685.2
def codes685 := Codes.rows ExportedData.chunk685.1 ExportedData.chunk685.2
def program685 := Coded.program 97634 87684 codes685
theorem checked685 : Coded.checkRows 97634 87684 codes685 = true := by decide
theorem ordered685 : Ordered 97634 87684 program685 := Coded.checked_ordered checked685
theorem length685 : program685.length = 128 := by
  rw [program685, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct685 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows685, row.Sat w) ↔ Satisfies program685 w := by
  rw [rows685, Codes.rows_expand]
  exact Coded.checked_correct checked685 h1

def rows686 := Exported.decodeRows ExportedData.chunk686.1 ExportedData.chunk686.2
def codes686 := Codes.rows ExportedData.chunk686.1 ExportedData.chunk686.2
def program686 := Coded.program 97634 87812 codes686
theorem checked686 : Coded.checkRows 97634 87812 codes686 = true := by decide
theorem ordered686 : Ordered 97634 87812 program686 := Coded.checked_ordered checked686
theorem length686 : program686.length = 128 := by
  rw [program686, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct686 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows686, row.Sat w) ↔ Satisfies program686 w := by
  rw [rows686, Codes.rows_expand]
  exact Coded.checked_correct checked686 h1

def rows687 := Exported.decodeRows ExportedData.chunk687.1 ExportedData.chunk687.2
def codes687 := Codes.rows ExportedData.chunk687.1 ExportedData.chunk687.2
def program687 := Coded.program 97634 87940 codes687
theorem checked687 : Coded.checkRows 97634 87940 codes687 = true := by decide
theorem ordered687 : Ordered 97634 87940 program687 := Coded.checked_ordered checked687
theorem length687 : program687.length = 128 := by
  rw [program687, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct687 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows687, row.Sat w) ↔ Satisfies program687 w := by
  rw [rows687, Codes.rows_expand]
  exact Coded.checked_correct checked687 h1

def rows688 := Exported.decodeRows ExportedData.chunk688.1 ExportedData.chunk688.2
def codes688 := Codes.rows ExportedData.chunk688.1 ExportedData.chunk688.2
def program688 := Coded.program 97634 88068 codes688
theorem checked688 : Coded.checkRows 97634 88068 codes688 = true := by decide
theorem ordered688 : Ordered 97634 88068 program688 := Coded.checked_ordered checked688
theorem length688 : program688.length = 128 := by
  rw [program688, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct688 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows688, row.Sat w) ↔ Satisfies program688 w := by
  rw [rows688, Codes.rows_expand]
  exact Coded.checked_correct checked688 h1

def rows689 := Exported.decodeRows ExportedData.chunk689.1 ExportedData.chunk689.2
def codes689 := Codes.rows ExportedData.chunk689.1 ExportedData.chunk689.2
def program689 := Coded.program 97634 88196 codes689
theorem checked689 : Coded.checkRows 97634 88196 codes689 = true := by decide
theorem ordered689 : Ordered 97634 88196 program689 := Coded.checked_ordered checked689
theorem length689 : program689.length = 128 := by
  rw [program689, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct689 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows689, row.Sat w) ↔ Satisfies program689 w := by
  rw [rows689, Codes.rows_expand]
  exact Coded.checked_correct checked689 h1

def rows690 := Exported.decodeRows ExportedData.chunk690.1 ExportedData.chunk690.2
def codes690 := Codes.rows ExportedData.chunk690.1 ExportedData.chunk690.2
def program690 := Coded.program 97634 88324 codes690
theorem checked690 : Coded.checkRows 97634 88324 codes690 = true := by decide
theorem ordered690 : Ordered 97634 88324 program690 := Coded.checked_ordered checked690
theorem length690 : program690.length = 128 := by
  rw [program690, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct690 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows690, row.Sat w) ↔ Satisfies program690 w := by
  rw [rows690, Codes.rows_expand]
  exact Coded.checked_correct checked690 h1

def rows691 := Exported.decodeRows ExportedData.chunk691.1 ExportedData.chunk691.2
def codes691 := Codes.rows ExportedData.chunk691.1 ExportedData.chunk691.2
def program691 := Coded.program 97634 88452 codes691
theorem checked691 : Coded.checkRows 97634 88452 codes691 = true := by decide
theorem ordered691 : Ordered 97634 88452 program691 := Coded.checked_ordered checked691
theorem length691 : program691.length = 128 := by
  rw [program691, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct691 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows691, row.Sat w) ↔ Satisfies program691 w := by
  rw [rows691, Codes.rows_expand]
  exact Coded.checked_correct checked691 h1

def rows692 := Exported.decodeRows ExportedData.chunk692.1 ExportedData.chunk692.2
def codes692 := Codes.rows ExportedData.chunk692.1 ExportedData.chunk692.2
def program692 := Coded.program 97634 88580 codes692
theorem checked692 : Coded.checkRows 97634 88580 codes692 = true := by decide
theorem ordered692 : Ordered 97634 88580 program692 := Coded.checked_ordered checked692
theorem length692 : program692.length = 128 := by
  rw [program692, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct692 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows692, row.Sat w) ↔ Satisfies program692 w := by
  rw [rows692, Codes.rows_expand]
  exact Coded.checked_correct checked692 h1

def rows693 := Exported.decodeRows ExportedData.chunk693.1 ExportedData.chunk693.2
def codes693 := Codes.rows ExportedData.chunk693.1 ExportedData.chunk693.2
def program693 := Coded.program 97634 88708 codes693
theorem checked693 : Coded.checkRows 97634 88708 codes693 = true := by decide
theorem ordered693 : Ordered 97634 88708 program693 := Coded.checked_ordered checked693
theorem length693 : program693.length = 128 := by
  rw [program693, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct693 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows693, row.Sat w) ↔ Satisfies program693 w := by
  rw [rows693, Codes.rows_expand]
  exact Coded.checked_correct checked693 h1

def rows694 := Exported.decodeRows ExportedData.chunk694.1 ExportedData.chunk694.2
def codes694 := Codes.rows ExportedData.chunk694.1 ExportedData.chunk694.2
def program694 := Coded.program 97634 88836 codes694
theorem checked694 : Coded.checkRows 97634 88836 codes694 = true := by decide
theorem ordered694 : Ordered 97634 88836 program694 := Coded.checked_ordered checked694
theorem length694 : program694.length = 128 := by
  rw [program694, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct694 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows694, row.Sat w) ↔ Satisfies program694 w := by
  rw [rows694, Codes.rows_expand]
  exact Coded.checked_correct checked694 h1

def rows695 := Exported.decodeRows ExportedData.chunk695.1 ExportedData.chunk695.2
def codes695 := Codes.rows ExportedData.chunk695.1 ExportedData.chunk695.2
def program695 := Coded.program 97634 88964 codes695
theorem checked695 : Coded.checkRows 97634 88964 codes695 = true := by decide
theorem ordered695 : Ordered 97634 88964 program695 := Coded.checked_ordered checked695
theorem length695 : program695.length = 128 := by
  rw [program695, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct695 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows695, row.Sat w) ↔ Satisfies program695 w := by
  rw [rows695, Codes.rows_expand]
  exact Coded.checked_correct checked695 h1

def rows696 := Exported.decodeRows ExportedData.chunk696.1 ExportedData.chunk696.2
def codes696 := Codes.rows ExportedData.chunk696.1 ExportedData.chunk696.2
def program696 := Coded.program 97634 89092 codes696
theorem checked696 : Coded.checkRows 97634 89092 codes696 = true := by decide
theorem ordered696 : Ordered 97634 89092 program696 := Coded.checked_ordered checked696
theorem length696 : program696.length = 128 := by
  rw [program696, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct696 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows696, row.Sat w) ↔ Satisfies program696 w := by
  rw [rows696, Codes.rows_expand]
  exact Coded.checked_correct checked696 h1

def rows697 := Exported.decodeRows ExportedData.chunk697.1 ExportedData.chunk697.2
def codes697 := Codes.rows ExportedData.chunk697.1 ExportedData.chunk697.2
def program697 := Coded.program 97634 89220 codes697
theorem checked697 : Coded.checkRows 97634 89220 codes697 = true := by decide
theorem ordered697 : Ordered 97634 89220 program697 := Coded.checked_ordered checked697
theorem length697 : program697.length = 128 := by
  rw [program697, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct697 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows697, row.Sat w) ↔ Satisfies program697 w := by
  rw [rows697, Codes.rows_expand]
  exact Coded.checked_correct checked697 h1

def rows698 := Exported.decodeRows ExportedData.chunk698.1 ExportedData.chunk698.2
def codes698 := Codes.rows ExportedData.chunk698.1 ExportedData.chunk698.2
def program698 := Coded.program 97634 89348 codes698
theorem checked698 : Coded.checkRows 97634 89348 codes698 = true := by decide
theorem ordered698 : Ordered 97634 89348 program698 := Coded.checked_ordered checked698
theorem length698 : program698.length = 128 := by
  rw [program698, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct698 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows698, row.Sat w) ↔ Satisfies program698 w := by
  rw [rows698, Codes.rows_expand]
  exact Coded.checked_correct checked698 h1

def rows699 := Exported.decodeRows ExportedData.chunk699.1 ExportedData.chunk699.2
def codes699 := Codes.rows ExportedData.chunk699.1 ExportedData.chunk699.2
def program699 := Coded.program 97634 89476 codes699
theorem checked699 : Coded.checkRows 97634 89476 codes699 = true := by decide
theorem ordered699 : Ordered 97634 89476 program699 := Coded.checked_ordered checked699
theorem length699 : program699.length = 128 := by
  rw [program699, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct699 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows699, row.Sat w) ↔ Satisfies program699 w := by
  rw [rows699, Codes.rows_expand]
  exact Coded.checked_correct checked699 h1

end CircuitCorrectness.ProgramCertificates
