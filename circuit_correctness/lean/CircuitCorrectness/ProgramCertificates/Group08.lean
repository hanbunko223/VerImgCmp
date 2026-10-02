import CircuitCorrectness.ConcreteProgram
import CircuitCorrectness.ProgramCertificates.Group04

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows700 := Exported.decodeRows ExportedData.chunk700.1 ExportedData.chunk700.2
def codes700 := Codes.rows ExportedData.chunk700.1 ExportedData.chunk700.2
def program700 := Coded.program 97634 89604 codes700
theorem checked700 : Coded.checkRows 97634 89604 codes700 = true := by decide
theorem ordered700 : Ordered 97634 89604 program700 := Coded.checked_ordered checked700
theorem length700 : program700.length = 128 := by
  rw [program700, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct700 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows700, row.Sat w) ↔ Satisfies program700 w := by
  rw [rows700, Codes.rows_expand]
  exact Coded.checked_correct checked700 h1

def rows701 := Exported.decodeRows ExportedData.chunk701.1 ExportedData.chunk701.2
def codes701 := Codes.rows ExportedData.chunk701.1 ExportedData.chunk701.2
def program701 := Coded.program 97634 89732 codes701
theorem checked701 : Coded.checkRows 97634 89732 codes701 = true := by decide
theorem ordered701 : Ordered 97634 89732 program701 := Coded.checked_ordered checked701
theorem length701 : program701.length = 128 := by
  rw [program701, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct701 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows701, row.Sat w) ↔ Satisfies program701 w := by
  rw [rows701, Codes.rows_expand]
  exact Coded.checked_correct checked701 h1

def rows702 := Exported.decodeRows ExportedData.chunk702.1 ExportedData.chunk702.2
def codes702 := Codes.rows ExportedData.chunk702.1 ExportedData.chunk702.2
def program702 := Coded.program 97634 89860 codes702
theorem checked702 : Coded.checkRows 97634 89860 codes702 = true := by decide
theorem ordered702 : Ordered 97634 89860 program702 := Coded.checked_ordered checked702
theorem length702 : program702.length = 128 := by
  rw [program702, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct702 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows702, row.Sat w) ↔ Satisfies program702 w := by
  rw [rows702, Codes.rows_expand]
  exact Coded.checked_correct checked702 h1

def rows703 := Exported.decodeRows ExportedData.chunk703.1 ExportedData.chunk703.2
def codes703 := Codes.rows ExportedData.chunk703.1 ExportedData.chunk703.2
def program703 := Coded.program 97634 89988 codes703
theorem checked703 : Coded.checkRows 97634 89988 codes703 = true := by decide
theorem ordered703 : Ordered 97634 89988 program703 := Coded.checked_ordered checked703
theorem length703 : program703.length = 128 := by
  rw [program703, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct703 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows703, row.Sat w) ↔ Satisfies program703 w := by
  rw [rows703, Codes.rows_expand]
  exact Coded.checked_correct checked703 h1

def rows704 := Exported.decodeRows ExportedData.chunk704.1 ExportedData.chunk704.2
def codes704 := Codes.rows ExportedData.chunk704.1 ExportedData.chunk704.2
def program704 := Coded.program 97634 90116 codes704
theorem checked704 : Coded.checkRows 97634 90116 codes704 = true := by decide
theorem ordered704 : Ordered 97634 90116 program704 := Coded.checked_ordered checked704
theorem length704 : program704.length = 128 := by
  rw [program704, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct704 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows704, row.Sat w) ↔ Satisfies program704 w := by
  rw [rows704, Codes.rows_expand]
  exact Coded.checked_correct checked704 h1

def rows705 := Exported.decodeRows ExportedData.chunk705.1 ExportedData.chunk705.2
def codes705 := Codes.rows ExportedData.chunk705.1 ExportedData.chunk705.2
def program705 := Coded.program 97634 90244 codes705
theorem checked705 : Coded.checkRows 97634 90244 codes705 = true := by decide
theorem ordered705 : Ordered 97634 90244 program705 := Coded.checked_ordered checked705
theorem length705 : program705.length = 128 := by
  rw [program705, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct705 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows705, row.Sat w) ↔ Satisfies program705 w := by
  rw [rows705, Codes.rows_expand]
  exact Coded.checked_correct checked705 h1

def rows706 := Exported.decodeRows ExportedData.chunk706.1 ExportedData.chunk706.2
def codes706 := Codes.rows ExportedData.chunk706.1 ExportedData.chunk706.2
def program706 := Coded.program 97634 90372 codes706
theorem checked706 : Coded.checkRows 97634 90372 codes706 = true := by decide
theorem ordered706 : Ordered 97634 90372 program706 := Coded.checked_ordered checked706
theorem length706 : program706.length = 128 := by
  rw [program706, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct706 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows706, row.Sat w) ↔ Satisfies program706 w := by
  rw [rows706, Codes.rows_expand]
  exact Coded.checked_correct checked706 h1

def rows707 := Exported.decodeRows ExportedData.chunk707.1 ExportedData.chunk707.2
def codes707 := Codes.rows ExportedData.chunk707.1 ExportedData.chunk707.2
def program707 := Coded.program 97634 90500 codes707
theorem checked707 : Coded.checkRows 97634 90500 codes707 = true := by decide
theorem ordered707 : Ordered 97634 90500 program707 := Coded.checked_ordered checked707
theorem length707 : program707.length = 128 := by
  rw [program707, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct707 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows707, row.Sat w) ↔ Satisfies program707 w := by
  rw [rows707, Codes.rows_expand]
  exact Coded.checked_correct checked707 h1

def rows708 := Exported.decodeRows ExportedData.chunk708.1 ExportedData.chunk708.2
def codes708 := Codes.rows ExportedData.chunk708.1 ExportedData.chunk708.2
def program708 := Coded.program 97634 90628 codes708
theorem checked708 : Coded.checkRows 97634 90628 codes708 = true := by decide
theorem ordered708 : Ordered 97634 90628 program708 := Coded.checked_ordered checked708
theorem length708 : program708.length = 128 := by
  rw [program708, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct708 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows708, row.Sat w) ↔ Satisfies program708 w := by
  rw [rows708, Codes.rows_expand]
  exact Coded.checked_correct checked708 h1

def rows709 := Exported.decodeRows ExportedData.chunk709.1 ExportedData.chunk709.2
def codes709 := Codes.rows ExportedData.chunk709.1 ExportedData.chunk709.2
def program709 := Coded.program 97634 90756 codes709
theorem checked709 : Coded.checkRows 97634 90756 codes709 = true := by decide
theorem ordered709 : Ordered 97634 90756 program709 := Coded.checked_ordered checked709
theorem length709 : program709.length = 128 := by
  rw [program709, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct709 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows709, row.Sat w) ↔ Satisfies program709 w := by
  rw [rows709, Codes.rows_expand]
  exact Coded.checked_correct checked709 h1

def rows710 := Exported.decodeRows ExportedData.chunk710.1 ExportedData.chunk710.2
def codes710 := Codes.rows ExportedData.chunk710.1 ExportedData.chunk710.2
def program710 := Coded.program 97634 90884 codes710
theorem checked710 : Coded.checkRows 97634 90884 codes710 = true := by decide
theorem ordered710 : Ordered 97634 90884 program710 := Coded.checked_ordered checked710
theorem length710 : program710.length = 128 := by
  rw [program710, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct710 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows710, row.Sat w) ↔ Satisfies program710 w := by
  rw [rows710, Codes.rows_expand]
  exact Coded.checked_correct checked710 h1

def rows711 := Exported.decodeRows ExportedData.chunk711.1 ExportedData.chunk711.2
def codes711 := Codes.rows ExportedData.chunk711.1 ExportedData.chunk711.2
def program711 := Coded.program 97634 91012 codes711
theorem checked711 : Coded.checkRows 97634 91012 codes711 = true := by decide
theorem ordered711 : Ordered 97634 91012 program711 := Coded.checked_ordered checked711
theorem length711 : program711.length = 128 := by
  rw [program711, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct711 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows711, row.Sat w) ↔ Satisfies program711 w := by
  rw [rows711, Codes.rows_expand]
  exact Coded.checked_correct checked711 h1

def rows712 := Exported.decodeRows ExportedData.chunk712.1 ExportedData.chunk712.2
def codes712 := Codes.rows ExportedData.chunk712.1 ExportedData.chunk712.2
def program712 := Coded.program 97634 91140 codes712
theorem checked712 : Coded.checkRows 97634 91140 codes712 = true := by decide
theorem ordered712 : Ordered 97634 91140 program712 := Coded.checked_ordered checked712
theorem length712 : program712.length = 128 := by
  rw [program712, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct712 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows712, row.Sat w) ↔ Satisfies program712 w := by
  rw [rows712, Codes.rows_expand]
  exact Coded.checked_correct checked712 h1

def rows713 := Exported.decodeRows ExportedData.chunk713.1 ExportedData.chunk713.2
def codes713 := Codes.rows ExportedData.chunk713.1 ExportedData.chunk713.2
def program713 := Coded.program 97634 91268 codes713
theorem checked713 : Coded.checkRows 97634 91268 codes713 = true := by decide
theorem ordered713 : Ordered 97634 91268 program713 := Coded.checked_ordered checked713
theorem length713 : program713.length = 128 := by
  rw [program713, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct713 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows713, row.Sat w) ↔ Satisfies program713 w := by
  rw [rows713, Codes.rows_expand]
  exact Coded.checked_correct checked713 h1

def rows714 := Exported.decodeRows ExportedData.chunk714.1 ExportedData.chunk714.2
def codes714 := Codes.rows ExportedData.chunk714.1 ExportedData.chunk714.2
def program714 := Coded.program 97634 91396 codes714
theorem checked714 : Coded.checkRows 97634 91396 codes714 = true := by decide
theorem ordered714 : Ordered 97634 91396 program714 := Coded.checked_ordered checked714
theorem length714 : program714.length = 128 := by
  rw [program714, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct714 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows714, row.Sat w) ↔ Satisfies program714 w := by
  rw [rows714, Codes.rows_expand]
  exact Coded.checked_correct checked714 h1

def rows715 := Exported.decodeRows ExportedData.chunk715.1 ExportedData.chunk715.2
def codes715 := Codes.rows ExportedData.chunk715.1 ExportedData.chunk715.2
def program715 := Coded.program 97634 91524 codes715
theorem checked715 : Coded.checkRows 97634 91524 codes715 = true := by decide
theorem ordered715 : Ordered 97634 91524 program715 := Coded.checked_ordered checked715
theorem length715 : program715.length = 128 := by
  rw [program715, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct715 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows715, row.Sat w) ↔ Satisfies program715 w := by
  rw [rows715, Codes.rows_expand]
  exact Coded.checked_correct checked715 h1

def rows716 := Exported.decodeRows ExportedData.chunk716.1 ExportedData.chunk716.2
def codes716 := Codes.rows ExportedData.chunk716.1 ExportedData.chunk716.2
def program716 := Coded.program 97634 91652 codes716
theorem checked716 : Coded.checkRows 97634 91652 codes716 = true := by decide
theorem ordered716 : Ordered 97634 91652 program716 := Coded.checked_ordered checked716
theorem length716 : program716.length = 128 := by
  rw [program716, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct716 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows716, row.Sat w) ↔ Satisfies program716 w := by
  rw [rows716, Codes.rows_expand]
  exact Coded.checked_correct checked716 h1

def rows717 := Exported.decodeRows ExportedData.chunk717.1 ExportedData.chunk717.2
def codes717 := Codes.rows ExportedData.chunk717.1 ExportedData.chunk717.2
def program717 := Coded.program 97634 91780 codes717
theorem checked717 : Coded.checkRows 97634 91780 codes717 = true := by decide
theorem ordered717 : Ordered 97634 91780 program717 := Coded.checked_ordered checked717
theorem length717 : program717.length = 128 := by
  rw [program717, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct717 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows717, row.Sat w) ↔ Satisfies program717 w := by
  rw [rows717, Codes.rows_expand]
  exact Coded.checked_correct checked717 h1

def rows718 := Exported.decodeRows ExportedData.chunk718.1 ExportedData.chunk718.2
def codes718 := Codes.rows ExportedData.chunk718.1 ExportedData.chunk718.2
def program718 := Coded.program 97634 91908 codes718
theorem checked718 : Coded.checkRows 97634 91908 codes718 = true := by decide
theorem ordered718 : Ordered 97634 91908 program718 := Coded.checked_ordered checked718
theorem length718 : program718.length = 128 := by
  rw [program718, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct718 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows718, row.Sat w) ↔ Satisfies program718 w := by
  rw [rows718, Codes.rows_expand]
  exact Coded.checked_correct checked718 h1

def rows719 := Exported.decodeRows ExportedData.chunk719.1 ExportedData.chunk719.2
def codes719 := Codes.rows ExportedData.chunk719.1 ExportedData.chunk719.2
def program719 := Coded.program 97634 92036 codes719
theorem checked719 : Coded.checkRows 97634 92036 codes719 = true := by decide
theorem ordered719 : Ordered 97634 92036 program719 := Coded.checked_ordered checked719
theorem length719 : program719.length = 128 := by
  rw [program719, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct719 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows719, row.Sat w) ↔ Satisfies program719 w := by
  rw [rows719, Codes.rows_expand]
  exact Coded.checked_correct checked719 h1

end CircuitCorrectness.ProgramCertificates
