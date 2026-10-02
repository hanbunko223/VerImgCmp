import CircuitCorrectness.ConcreteProgram
import CircuitCorrectness.ProgramCertificates.Group07

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows760 := Exported.decodeRows ExportedData.chunk760.1 ExportedData.chunk760.2
def codes760 := Codes.rows ExportedData.chunk760.1 ExportedData.chunk760.2
def program760 := Coded.program 97634 97284 codes760
theorem checked760 : Coded.checkRows 97634 97284 codes760 = true := by decide
theorem ordered760 : Ordered 97634 97284 program760 := Coded.checked_ordered checked760
theorem length760 : program760.length = 128 := by
  rw [program760, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct760 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows760, row.Sat w) ↔ Satisfies program760 w := by
  rw [rows760, Codes.rows_expand]
  exact Coded.checked_correct checked760 h1

def rows761 := Exported.decodeRows ExportedData.chunk761.1 ExportedData.chunk761.2
def codes761 := Codes.rows ExportedData.chunk761.1 ExportedData.chunk761.2
def program761 := Coded.program 97634 97412 codes761
theorem checked761 : Coded.checkRows 97634 97412 codes761 = true := by decide
theorem ordered761 : Ordered 97634 97412 program761 := Coded.checked_ordered checked761
theorem length761 : program761.length = 128 := by
  rw [program761, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct761 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows761, row.Sat w) ↔ Satisfies program761 w := by
  rw [rows761, Codes.rows_expand]
  exact Coded.checked_correct checked761 h1

def rows762 := Exported.decodeRows ExportedData.chunk762.1 ExportedData.chunk762.2
def codes762 := Codes.rows ExportedData.chunk762.1 ExportedData.chunk762.2
def program762 := Coded.program 97634 97540 codes762
theorem checked762 : Coded.checkRows 97634 97540 codes762 = true := by decide
theorem ordered762 : Ordered 97634 97540 program762 := Coded.checked_ordered checked762
theorem length762 : program762.length = 94 := by
  rw [program762, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct762 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows762, row.Sat w) ↔ Satisfies program762 w := by
  rw [rows762, Codes.rows_expand]
  exact Coded.checked_correct checked762 h1

end CircuitCorrectness.ProgramCertificates
