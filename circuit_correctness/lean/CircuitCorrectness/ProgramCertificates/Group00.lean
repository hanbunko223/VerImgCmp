import CircuitCorrectness.ConcreteProgram

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows540 := Exported.decodeRows ExportedData.chunk540.1 ExportedData.chunk540.2
def codes540 := Codes.rows ExportedData.chunk540.1 ExportedData.chunk540.2
def program540 := Coded.program 97634 69124 codes540
theorem checked540 : Coded.checkRows 97634 69124 codes540 = true := by decide
theorem ordered540 : Ordered 97634 69124 program540 := Coded.checked_ordered checked540
theorem length540 : program540.length = 128 := by
  rw [program540, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct540 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows540, row.Sat w) ↔ Satisfies program540 w := by
  rw [rows540, Codes.rows_expand]
  exact Coded.checked_correct checked540 h1

def rows541 := Exported.decodeRows ExportedData.chunk541.1 ExportedData.chunk541.2
def codes541 := Codes.rows ExportedData.chunk541.1 ExportedData.chunk541.2
def program541 := Coded.program 97634 69252 codes541
theorem checked541 : Coded.checkRows 97634 69252 codes541 = true := by decide
theorem ordered541 : Ordered 97634 69252 program541 := Coded.checked_ordered checked541
theorem length541 : program541.length = 128 := by
  rw [program541, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct541 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows541, row.Sat w) ↔ Satisfies program541 w := by
  rw [rows541, Codes.rows_expand]
  exact Coded.checked_correct checked541 h1

def rows542 := Exported.decodeRows ExportedData.chunk542.1 ExportedData.chunk542.2
def codes542 := Codes.rows ExportedData.chunk542.1 ExportedData.chunk542.2
def program542 := Coded.program 97634 69380 codes542
theorem checked542 : Coded.checkRows 97634 69380 codes542 = true := by decide
theorem ordered542 : Ordered 97634 69380 program542 := Coded.checked_ordered checked542
theorem length542 : program542.length = 128 := by
  rw [program542, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct542 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows542, row.Sat w) ↔ Satisfies program542 w := by
  rw [rows542, Codes.rows_expand]
  exact Coded.checked_correct checked542 h1

def rows543 := Exported.decodeRows ExportedData.chunk543.1 ExportedData.chunk543.2
def codes543 := Codes.rows ExportedData.chunk543.1 ExportedData.chunk543.2
def program543 := Coded.program 97634 69508 codes543
theorem checked543 : Coded.checkRows 97634 69508 codes543 = true := by decide
theorem ordered543 : Ordered 97634 69508 program543 := Coded.checked_ordered checked543
theorem length543 : program543.length = 128 := by
  rw [program543, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct543 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows543, row.Sat w) ↔ Satisfies program543 w := by
  rw [rows543, Codes.rows_expand]
  exact Coded.checked_correct checked543 h1

def rows544 := Exported.decodeRows ExportedData.chunk544.1 ExportedData.chunk544.2
def codes544 := Codes.rows ExportedData.chunk544.1 ExportedData.chunk544.2
def program544 := Coded.program 97634 69636 codes544
theorem checked544 : Coded.checkRows 97634 69636 codes544 = true := by decide
theorem ordered544 : Ordered 97634 69636 program544 := Coded.checked_ordered checked544
theorem length544 : program544.length = 128 := by
  rw [program544, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct544 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows544, row.Sat w) ↔ Satisfies program544 w := by
  rw [rows544, Codes.rows_expand]
  exact Coded.checked_correct checked544 h1

def rows545 := Exported.decodeRows ExportedData.chunk545.1 ExportedData.chunk545.2
def codes545 := Codes.rows ExportedData.chunk545.1 ExportedData.chunk545.2
def program545 := Coded.program 97634 69764 codes545
theorem checked545 : Coded.checkRows 97634 69764 codes545 = true := by decide
theorem ordered545 : Ordered 97634 69764 program545 := Coded.checked_ordered checked545
theorem length545 : program545.length = 128 := by
  rw [program545, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct545 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows545, row.Sat w) ↔ Satisfies program545 w := by
  rw [rows545, Codes.rows_expand]
  exact Coded.checked_correct checked545 h1

def rows546 := Exported.decodeRows ExportedData.chunk546.1 ExportedData.chunk546.2
def codes546 := Codes.rows ExportedData.chunk546.1 ExportedData.chunk546.2
def program546 := Coded.program 97634 69892 codes546
theorem checked546 : Coded.checkRows 97634 69892 codes546 = true := by decide
theorem ordered546 : Ordered 97634 69892 program546 := Coded.checked_ordered checked546
theorem length546 : program546.length = 128 := by
  rw [program546, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct546 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows546, row.Sat w) ↔ Satisfies program546 w := by
  rw [rows546, Codes.rows_expand]
  exact Coded.checked_correct checked546 h1

def rows547 := Exported.decodeRows ExportedData.chunk547.1 ExportedData.chunk547.2
def codes547 := Codes.rows ExportedData.chunk547.1 ExportedData.chunk547.2
def program547 := Coded.program 97634 70020 codes547
theorem checked547 : Coded.checkRows 97634 70020 codes547 = true := by decide
theorem ordered547 : Ordered 97634 70020 program547 := Coded.checked_ordered checked547
theorem length547 : program547.length = 128 := by
  rw [program547, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct547 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows547, row.Sat w) ↔ Satisfies program547 w := by
  rw [rows547, Codes.rows_expand]
  exact Coded.checked_correct checked547 h1

def rows548 := Exported.decodeRows ExportedData.chunk548.1 ExportedData.chunk548.2
def codes548 := Codes.rows ExportedData.chunk548.1 ExportedData.chunk548.2
def program548 := Coded.program 97634 70148 codes548
theorem checked548 : Coded.checkRows 97634 70148 codes548 = true := by decide
theorem ordered548 : Ordered 97634 70148 program548 := Coded.checked_ordered checked548
theorem length548 : program548.length = 128 := by
  rw [program548, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct548 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows548, row.Sat w) ↔ Satisfies program548 w := by
  rw [rows548, Codes.rows_expand]
  exact Coded.checked_correct checked548 h1

def rows549 := Exported.decodeRows ExportedData.chunk549.1 ExportedData.chunk549.2
def codes549 := Codes.rows ExportedData.chunk549.1 ExportedData.chunk549.2
def program549 := Coded.program 97634 70276 codes549
theorem checked549 : Coded.checkRows 97634 70276 codes549 = true := by decide
theorem ordered549 : Ordered 97634 70276 program549 := Coded.checked_ordered checked549
theorem length549 : program549.length = 128 := by
  rw [program549, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct549 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows549, row.Sat w) ↔ Satisfies program549 w := by
  rw [rows549, Codes.rows_expand]
  exact Coded.checked_correct checked549 h1

def rows550 := Exported.decodeRows ExportedData.chunk550.1 ExportedData.chunk550.2
def codes550 := Codes.rows ExportedData.chunk550.1 ExportedData.chunk550.2
def program550 := Coded.program 97634 70404 codes550
theorem checked550 : Coded.checkRows 97634 70404 codes550 = true := by decide
theorem ordered550 : Ordered 97634 70404 program550 := Coded.checked_ordered checked550
theorem length550 : program550.length = 128 := by
  rw [program550, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct550 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows550, row.Sat w) ↔ Satisfies program550 w := by
  rw [rows550, Codes.rows_expand]
  exact Coded.checked_correct checked550 h1

def rows551 := Exported.decodeRows ExportedData.chunk551.1 ExportedData.chunk551.2
def codes551 := Codes.rows ExportedData.chunk551.1 ExportedData.chunk551.2
def program551 := Coded.program 97634 70532 codes551
theorem checked551 : Coded.checkRows 97634 70532 codes551 = true := by decide
theorem ordered551 : Ordered 97634 70532 program551 := Coded.checked_ordered checked551
theorem length551 : program551.length = 128 := by
  rw [program551, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct551 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows551, row.Sat w) ↔ Satisfies program551 w := by
  rw [rows551, Codes.rows_expand]
  exact Coded.checked_correct checked551 h1

def rows552 := Exported.decodeRows ExportedData.chunk552.1 ExportedData.chunk552.2
def codes552 := Codes.rows ExportedData.chunk552.1 ExportedData.chunk552.2
def program552 := Coded.program 97634 70660 codes552
theorem checked552 : Coded.checkRows 97634 70660 codes552 = true := by decide
theorem ordered552 : Ordered 97634 70660 program552 := Coded.checked_ordered checked552
theorem length552 : program552.length = 128 := by
  rw [program552, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct552 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows552, row.Sat w) ↔ Satisfies program552 w := by
  rw [rows552, Codes.rows_expand]
  exact Coded.checked_correct checked552 h1

def rows553 := Exported.decodeRows ExportedData.chunk553.1 ExportedData.chunk553.2
def codes553 := Codes.rows ExportedData.chunk553.1 ExportedData.chunk553.2
def program553 := Coded.program 97634 70788 codes553
theorem checked553 : Coded.checkRows 97634 70788 codes553 = true := by decide
theorem ordered553 : Ordered 97634 70788 program553 := Coded.checked_ordered checked553
theorem length553 : program553.length = 128 := by
  rw [program553, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct553 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows553, row.Sat w) ↔ Satisfies program553 w := by
  rw [rows553, Codes.rows_expand]
  exact Coded.checked_correct checked553 h1

def rows554 := Exported.decodeRows ExportedData.chunk554.1 ExportedData.chunk554.2
def codes554 := Codes.rows ExportedData.chunk554.1 ExportedData.chunk554.2
def program554 := Coded.program 97634 70916 codes554
theorem checked554 : Coded.checkRows 97634 70916 codes554 = true := by decide
theorem ordered554 : Ordered 97634 70916 program554 := Coded.checked_ordered checked554
theorem length554 : program554.length = 128 := by
  rw [program554, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct554 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows554, row.Sat w) ↔ Satisfies program554 w := by
  rw [rows554, Codes.rows_expand]
  exact Coded.checked_correct checked554 h1

def rows555 := Exported.decodeRows ExportedData.chunk555.1 ExportedData.chunk555.2
def codes555 := Codes.rows ExportedData.chunk555.1 ExportedData.chunk555.2
def program555 := Coded.program 97634 71044 codes555
theorem checked555 : Coded.checkRows 97634 71044 codes555 = true := by decide
theorem ordered555 : Ordered 97634 71044 program555 := Coded.checked_ordered checked555
theorem length555 : program555.length = 128 := by
  rw [program555, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct555 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows555, row.Sat w) ↔ Satisfies program555 w := by
  rw [rows555, Codes.rows_expand]
  exact Coded.checked_correct checked555 h1

def rows556 := Exported.decodeRows ExportedData.chunk556.1 ExportedData.chunk556.2
def codes556 := Codes.rows ExportedData.chunk556.1 ExportedData.chunk556.2
def program556 := Coded.program 97634 71172 codes556
theorem checked556 : Coded.checkRows 97634 71172 codes556 = true := by decide
theorem ordered556 : Ordered 97634 71172 program556 := Coded.checked_ordered checked556
theorem length556 : program556.length = 128 := by
  rw [program556, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct556 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows556, row.Sat w) ↔ Satisfies program556 w := by
  rw [rows556, Codes.rows_expand]
  exact Coded.checked_correct checked556 h1

def rows557 := Exported.decodeRows ExportedData.chunk557.1 ExportedData.chunk557.2
def codes557 := Codes.rows ExportedData.chunk557.1 ExportedData.chunk557.2
def program557 := Coded.program 97634 71300 codes557
theorem checked557 : Coded.checkRows 97634 71300 codes557 = true := by decide
theorem ordered557 : Ordered 97634 71300 program557 := Coded.checked_ordered checked557
theorem length557 : program557.length = 128 := by
  rw [program557, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct557 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows557, row.Sat w) ↔ Satisfies program557 w := by
  rw [rows557, Codes.rows_expand]
  exact Coded.checked_correct checked557 h1

def rows558 := Exported.decodeRows ExportedData.chunk558.1 ExportedData.chunk558.2
def codes558 := Codes.rows ExportedData.chunk558.1 ExportedData.chunk558.2
def program558 := Coded.program 97634 71428 codes558
theorem checked558 : Coded.checkRows 97634 71428 codes558 = true := by decide
theorem ordered558 : Ordered 97634 71428 program558 := Coded.checked_ordered checked558
theorem length558 : program558.length = 128 := by
  rw [program558, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct558 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows558, row.Sat w) ↔ Satisfies program558 w := by
  rw [rows558, Codes.rows_expand]
  exact Coded.checked_correct checked558 h1

def rows559 := Exported.decodeRows ExportedData.chunk559.1 ExportedData.chunk559.2
def codes559 := Codes.rows ExportedData.chunk559.1 ExportedData.chunk559.2
def program559 := Coded.program 97634 71556 codes559
theorem checked559 : Coded.checkRows 97634 71556 codes559 = true := by decide
theorem ordered559 : Ordered 97634 71556 program559 := Coded.checked_ordered checked559
theorem length559 : program559.length = 128 := by
  rw [program559, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct559 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows559, row.Sat w) ↔ Satisfies program559 w := by
  rw [rows559, Codes.rows_expand]
  exact Coded.checked_correct checked559 h1

end CircuitCorrectness.ProgramCertificates
