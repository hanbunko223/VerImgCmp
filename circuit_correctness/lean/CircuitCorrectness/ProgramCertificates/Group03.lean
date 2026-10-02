import CircuitCorrectness.ConcreteProgram

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows600 := Exported.decodeRows ExportedData.chunk600.1 ExportedData.chunk600.2
def codes600 := Codes.rows ExportedData.chunk600.1 ExportedData.chunk600.2
def program600 := Coded.program 97634 76804 codes600
theorem checked600 : Coded.checkRows 97634 76804 codes600 = true := by decide
theorem ordered600 : Ordered 97634 76804 program600 := Coded.checked_ordered checked600
theorem length600 : program600.length = 128 := by
  rw [program600, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct600 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows600, row.Sat w) ↔ Satisfies program600 w := by
  rw [rows600, Codes.rows_expand]
  exact Coded.checked_correct checked600 h1

def rows601 := Exported.decodeRows ExportedData.chunk601.1 ExportedData.chunk601.2
def codes601 := Codes.rows ExportedData.chunk601.1 ExportedData.chunk601.2
def program601 := Coded.program 97634 76932 codes601
theorem checked601 : Coded.checkRows 97634 76932 codes601 = true := by decide
theorem ordered601 : Ordered 97634 76932 program601 := Coded.checked_ordered checked601
theorem length601 : program601.length = 128 := by
  rw [program601, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct601 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows601, row.Sat w) ↔ Satisfies program601 w := by
  rw [rows601, Codes.rows_expand]
  exact Coded.checked_correct checked601 h1

def rows602 := Exported.decodeRows ExportedData.chunk602.1 ExportedData.chunk602.2
def codes602 := Codes.rows ExportedData.chunk602.1 ExportedData.chunk602.2
def program602 := Coded.program 97634 77060 codes602
theorem checked602 : Coded.checkRows 97634 77060 codes602 = true := by decide
theorem ordered602 : Ordered 97634 77060 program602 := Coded.checked_ordered checked602
theorem length602 : program602.length = 128 := by
  rw [program602, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct602 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows602, row.Sat w) ↔ Satisfies program602 w := by
  rw [rows602, Codes.rows_expand]
  exact Coded.checked_correct checked602 h1

def rows603 := Exported.decodeRows ExportedData.chunk603.1 ExportedData.chunk603.2
def codes603 := Codes.rows ExportedData.chunk603.1 ExportedData.chunk603.2
def program603 := Coded.program 97634 77188 codes603
theorem checked603 : Coded.checkRows 97634 77188 codes603 = true := by decide
theorem ordered603 : Ordered 97634 77188 program603 := Coded.checked_ordered checked603
theorem length603 : program603.length = 128 := by
  rw [program603, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct603 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows603, row.Sat w) ↔ Satisfies program603 w := by
  rw [rows603, Codes.rows_expand]
  exact Coded.checked_correct checked603 h1

def rows604 := Exported.decodeRows ExportedData.chunk604.1 ExportedData.chunk604.2
def codes604 := Codes.rows ExportedData.chunk604.1 ExportedData.chunk604.2
def program604 := Coded.program 97634 77316 codes604
theorem checked604 : Coded.checkRows 97634 77316 codes604 = true := by decide
theorem ordered604 : Ordered 97634 77316 program604 := Coded.checked_ordered checked604
theorem length604 : program604.length = 128 := by
  rw [program604, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct604 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows604, row.Sat w) ↔ Satisfies program604 w := by
  rw [rows604, Codes.rows_expand]
  exact Coded.checked_correct checked604 h1

def rows605 := Exported.decodeRows ExportedData.chunk605.1 ExportedData.chunk605.2
def codes605 := Codes.rows ExportedData.chunk605.1 ExportedData.chunk605.2
def program605 := Coded.program 97634 77444 codes605
theorem checked605 : Coded.checkRows 97634 77444 codes605 = true := by decide
theorem ordered605 : Ordered 97634 77444 program605 := Coded.checked_ordered checked605
theorem length605 : program605.length = 128 := by
  rw [program605, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct605 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows605, row.Sat w) ↔ Satisfies program605 w := by
  rw [rows605, Codes.rows_expand]
  exact Coded.checked_correct checked605 h1

def rows606 := Exported.decodeRows ExportedData.chunk606.1 ExportedData.chunk606.2
def codes606 := Codes.rows ExportedData.chunk606.1 ExportedData.chunk606.2
def program606 := Coded.program 97634 77572 codes606
theorem checked606 : Coded.checkRows 97634 77572 codes606 = true := by decide
theorem ordered606 : Ordered 97634 77572 program606 := Coded.checked_ordered checked606
theorem length606 : program606.length = 128 := by
  rw [program606, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct606 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows606, row.Sat w) ↔ Satisfies program606 w := by
  rw [rows606, Codes.rows_expand]
  exact Coded.checked_correct checked606 h1

def rows607 := Exported.decodeRows ExportedData.chunk607.1 ExportedData.chunk607.2
def codes607 := Codes.rows ExportedData.chunk607.1 ExportedData.chunk607.2
def program607 := Coded.program 97634 77700 codes607
theorem checked607 : Coded.checkRows 97634 77700 codes607 = true := by decide
theorem ordered607 : Ordered 97634 77700 program607 := Coded.checked_ordered checked607
theorem length607 : program607.length = 128 := by
  rw [program607, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct607 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows607, row.Sat w) ↔ Satisfies program607 w := by
  rw [rows607, Codes.rows_expand]
  exact Coded.checked_correct checked607 h1

def rows608 := Exported.decodeRows ExportedData.chunk608.1 ExportedData.chunk608.2
def codes608 := Codes.rows ExportedData.chunk608.1 ExportedData.chunk608.2
def program608 := Coded.program 97634 77828 codes608
theorem checked608 : Coded.checkRows 97634 77828 codes608 = true := by decide
theorem ordered608 : Ordered 97634 77828 program608 := Coded.checked_ordered checked608
theorem length608 : program608.length = 128 := by
  rw [program608, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct608 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows608, row.Sat w) ↔ Satisfies program608 w := by
  rw [rows608, Codes.rows_expand]
  exact Coded.checked_correct checked608 h1

def rows609 := Exported.decodeRows ExportedData.chunk609.1 ExportedData.chunk609.2
def codes609 := Codes.rows ExportedData.chunk609.1 ExportedData.chunk609.2
def program609 := Coded.program 97634 77956 codes609
theorem checked609 : Coded.checkRows 97634 77956 codes609 = true := by decide
theorem ordered609 : Ordered 97634 77956 program609 := Coded.checked_ordered checked609
theorem length609 : program609.length = 128 := by
  rw [program609, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct609 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows609, row.Sat w) ↔ Satisfies program609 w := by
  rw [rows609, Codes.rows_expand]
  exact Coded.checked_correct checked609 h1

def rows610 := Exported.decodeRows ExportedData.chunk610.1 ExportedData.chunk610.2
def codes610 := Codes.rows ExportedData.chunk610.1 ExportedData.chunk610.2
def program610 := Coded.program 97634 78084 codes610
theorem checked610 : Coded.checkRows 97634 78084 codes610 = true := by decide
theorem ordered610 : Ordered 97634 78084 program610 := Coded.checked_ordered checked610
theorem length610 : program610.length = 128 := by
  rw [program610, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct610 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows610, row.Sat w) ↔ Satisfies program610 w := by
  rw [rows610, Codes.rows_expand]
  exact Coded.checked_correct checked610 h1

def rows611 := Exported.decodeRows ExportedData.chunk611.1 ExportedData.chunk611.2
def codes611 := Codes.rows ExportedData.chunk611.1 ExportedData.chunk611.2
def program611 := Coded.program 97634 78212 codes611
theorem checked611 : Coded.checkRows 97634 78212 codes611 = true := by decide
theorem ordered611 : Ordered 97634 78212 program611 := Coded.checked_ordered checked611
theorem length611 : program611.length = 128 := by
  rw [program611, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct611 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows611, row.Sat w) ↔ Satisfies program611 w := by
  rw [rows611, Codes.rows_expand]
  exact Coded.checked_correct checked611 h1

def rows612 := Exported.decodeRows ExportedData.chunk612.1 ExportedData.chunk612.2
def codes612 := Codes.rows ExportedData.chunk612.1 ExportedData.chunk612.2
def program612 := Coded.program 97634 78340 codes612
theorem checked612 : Coded.checkRows 97634 78340 codes612 = true := by decide
theorem ordered612 : Ordered 97634 78340 program612 := Coded.checked_ordered checked612
theorem length612 : program612.length = 128 := by
  rw [program612, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct612 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows612, row.Sat w) ↔ Satisfies program612 w := by
  rw [rows612, Codes.rows_expand]
  exact Coded.checked_correct checked612 h1

def rows613 := Exported.decodeRows ExportedData.chunk613.1 ExportedData.chunk613.2
def codes613 := Codes.rows ExportedData.chunk613.1 ExportedData.chunk613.2
def program613 := Coded.program 97634 78468 codes613
theorem checked613 : Coded.checkRows 97634 78468 codes613 = true := by decide
theorem ordered613 : Ordered 97634 78468 program613 := Coded.checked_ordered checked613
theorem length613 : program613.length = 128 := by
  rw [program613, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct613 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows613, row.Sat w) ↔ Satisfies program613 w := by
  rw [rows613, Codes.rows_expand]
  exact Coded.checked_correct checked613 h1

def rows614 := Exported.decodeRows ExportedData.chunk614.1 ExportedData.chunk614.2
def codes614 := Codes.rows ExportedData.chunk614.1 ExportedData.chunk614.2
def program614 := Coded.program 97634 78596 codes614
theorem checked614 : Coded.checkRows 97634 78596 codes614 = true := by decide
theorem ordered614 : Ordered 97634 78596 program614 := Coded.checked_ordered checked614
theorem length614 : program614.length = 128 := by
  rw [program614, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct614 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows614, row.Sat w) ↔ Satisfies program614 w := by
  rw [rows614, Codes.rows_expand]
  exact Coded.checked_correct checked614 h1

def rows615 := Exported.decodeRows ExportedData.chunk615.1 ExportedData.chunk615.2
def codes615 := Codes.rows ExportedData.chunk615.1 ExportedData.chunk615.2
def program615 := Coded.program 97634 78724 codes615
theorem checked615 : Coded.checkRows 97634 78724 codes615 = true := by decide
theorem ordered615 : Ordered 97634 78724 program615 := Coded.checked_ordered checked615
theorem length615 : program615.length = 128 := by
  rw [program615, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct615 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows615, row.Sat w) ↔ Satisfies program615 w := by
  rw [rows615, Codes.rows_expand]
  exact Coded.checked_correct checked615 h1

def rows616 := Exported.decodeRows ExportedData.chunk616.1 ExportedData.chunk616.2
def codes616 := Codes.rows ExportedData.chunk616.1 ExportedData.chunk616.2
def program616 := Coded.program 97634 78852 codes616
theorem checked616 : Coded.checkRows 97634 78852 codes616 = true := by decide
theorem ordered616 : Ordered 97634 78852 program616 := Coded.checked_ordered checked616
theorem length616 : program616.length = 128 := by
  rw [program616, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct616 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows616, row.Sat w) ↔ Satisfies program616 w := by
  rw [rows616, Codes.rows_expand]
  exact Coded.checked_correct checked616 h1

def rows617 := Exported.decodeRows ExportedData.chunk617.1 ExportedData.chunk617.2
def codes617 := Codes.rows ExportedData.chunk617.1 ExportedData.chunk617.2
def program617 := Coded.program 97634 78980 codes617
theorem checked617 : Coded.checkRows 97634 78980 codes617 = true := by decide
theorem ordered617 : Ordered 97634 78980 program617 := Coded.checked_ordered checked617
theorem length617 : program617.length = 128 := by
  rw [program617, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct617 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows617, row.Sat w) ↔ Satisfies program617 w := by
  rw [rows617, Codes.rows_expand]
  exact Coded.checked_correct checked617 h1

def rows618 := Exported.decodeRows ExportedData.chunk618.1 ExportedData.chunk618.2
def codes618 := Codes.rows ExportedData.chunk618.1 ExportedData.chunk618.2
def program618 := Coded.program 97634 79108 codes618
theorem checked618 : Coded.checkRows 97634 79108 codes618 = true := by decide
theorem ordered618 : Ordered 97634 79108 program618 := Coded.checked_ordered checked618
theorem length618 : program618.length = 128 := by
  rw [program618, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct618 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows618, row.Sat w) ↔ Satisfies program618 w := by
  rw [rows618, Codes.rows_expand]
  exact Coded.checked_correct checked618 h1

def rows619 := Exported.decodeRows ExportedData.chunk619.1 ExportedData.chunk619.2
def codes619 := Codes.rows ExportedData.chunk619.1 ExportedData.chunk619.2
def program619 := Coded.program 97634 79236 codes619
theorem checked619 : Coded.checkRows 97634 79236 codes619 = true := by decide
theorem ordered619 : Ordered 97634 79236 program619 := Coded.checked_ordered checked619
theorem length619 : program619.length = 128 := by
  rw [program619, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct619 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows619, row.Sat w) ↔ Satisfies program619 w := by
  rw [rows619, Codes.rows_expand]
  exact Coded.checked_correct checked619 h1

end CircuitCorrectness.ProgramCertificates
