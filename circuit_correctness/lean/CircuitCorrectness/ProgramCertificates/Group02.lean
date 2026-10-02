import CircuitCorrectness.ConcreteProgram

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine ConcreteBytes

def rows580 := Exported.decodeRows ExportedData.chunk580.1 ExportedData.chunk580.2
def codes580 := Codes.rows ExportedData.chunk580.1 ExportedData.chunk580.2
def program580 := Coded.program 97634 74244 codes580
theorem checked580 : Coded.checkRows 97634 74244 codes580 = true := by decide
theorem ordered580 : Ordered 97634 74244 program580 := Coded.checked_ordered checked580
theorem length580 : program580.length = 128 := by
  rw [program580, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct580 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows580, row.Sat w) ↔ Satisfies program580 w := by
  rw [rows580, Codes.rows_expand]
  exact Coded.checked_correct checked580 h1

def rows581 := Exported.decodeRows ExportedData.chunk581.1 ExportedData.chunk581.2
def codes581 := Codes.rows ExportedData.chunk581.1 ExportedData.chunk581.2
def program581 := Coded.program 97634 74372 codes581
theorem checked581 : Coded.checkRows 97634 74372 codes581 = true := by decide
theorem ordered581 : Ordered 97634 74372 program581 := Coded.checked_ordered checked581
theorem length581 : program581.length = 128 := by
  rw [program581, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct581 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows581, row.Sat w) ↔ Satisfies program581 w := by
  rw [rows581, Codes.rows_expand]
  exact Coded.checked_correct checked581 h1

def rows582 := Exported.decodeRows ExportedData.chunk582.1 ExportedData.chunk582.2
def codes582 := Codes.rows ExportedData.chunk582.1 ExportedData.chunk582.2
def program582 := Coded.program 97634 74500 codes582
theorem checked582 : Coded.checkRows 97634 74500 codes582 = true := by decide
theorem ordered582 : Ordered 97634 74500 program582 := Coded.checked_ordered checked582
theorem length582 : program582.length = 128 := by
  rw [program582, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct582 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows582, row.Sat w) ↔ Satisfies program582 w := by
  rw [rows582, Codes.rows_expand]
  exact Coded.checked_correct checked582 h1

def rows583 := Exported.decodeRows ExportedData.chunk583.1 ExportedData.chunk583.2
def codes583 := Codes.rows ExportedData.chunk583.1 ExportedData.chunk583.2
def program583 := Coded.program 97634 74628 codes583
theorem checked583 : Coded.checkRows 97634 74628 codes583 = true := by decide
theorem ordered583 : Ordered 97634 74628 program583 := Coded.checked_ordered checked583
theorem length583 : program583.length = 128 := by
  rw [program583, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct583 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows583, row.Sat w) ↔ Satisfies program583 w := by
  rw [rows583, Codes.rows_expand]
  exact Coded.checked_correct checked583 h1

def rows584 := Exported.decodeRows ExportedData.chunk584.1 ExportedData.chunk584.2
def codes584 := Codes.rows ExportedData.chunk584.1 ExportedData.chunk584.2
def program584 := Coded.program 97634 74756 codes584
theorem checked584 : Coded.checkRows 97634 74756 codes584 = true := by decide
theorem ordered584 : Ordered 97634 74756 program584 := Coded.checked_ordered checked584
theorem length584 : program584.length = 128 := by
  rw [program584, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct584 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows584, row.Sat w) ↔ Satisfies program584 w := by
  rw [rows584, Codes.rows_expand]
  exact Coded.checked_correct checked584 h1

def rows585 := Exported.decodeRows ExportedData.chunk585.1 ExportedData.chunk585.2
def codes585 := Codes.rows ExportedData.chunk585.1 ExportedData.chunk585.2
def program585 := Coded.program 97634 74884 codes585
theorem checked585 : Coded.checkRows 97634 74884 codes585 = true := by decide
theorem ordered585 : Ordered 97634 74884 program585 := Coded.checked_ordered checked585
theorem length585 : program585.length = 128 := by
  rw [program585, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct585 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows585, row.Sat w) ↔ Satisfies program585 w := by
  rw [rows585, Codes.rows_expand]
  exact Coded.checked_correct checked585 h1

def rows586 := Exported.decodeRows ExportedData.chunk586.1 ExportedData.chunk586.2
def codes586 := Codes.rows ExportedData.chunk586.1 ExportedData.chunk586.2
def program586 := Coded.program 97634 75012 codes586
theorem checked586 : Coded.checkRows 97634 75012 codes586 = true := by decide
theorem ordered586 : Ordered 97634 75012 program586 := Coded.checked_ordered checked586
theorem length586 : program586.length = 128 := by
  rw [program586, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct586 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows586, row.Sat w) ↔ Satisfies program586 w := by
  rw [rows586, Codes.rows_expand]
  exact Coded.checked_correct checked586 h1

def rows587 := Exported.decodeRows ExportedData.chunk587.1 ExportedData.chunk587.2
def codes587 := Codes.rows ExportedData.chunk587.1 ExportedData.chunk587.2
def program587 := Coded.program 97634 75140 codes587
theorem checked587 : Coded.checkRows 97634 75140 codes587 = true := by decide
theorem ordered587 : Ordered 97634 75140 program587 := Coded.checked_ordered checked587
theorem length587 : program587.length = 128 := by
  rw [program587, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct587 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows587, row.Sat w) ↔ Satisfies program587 w := by
  rw [rows587, Codes.rows_expand]
  exact Coded.checked_correct checked587 h1

def rows588 := Exported.decodeRows ExportedData.chunk588.1 ExportedData.chunk588.2
def codes588 := Codes.rows ExportedData.chunk588.1 ExportedData.chunk588.2
def program588 := Coded.program 97634 75268 codes588
theorem checked588 : Coded.checkRows 97634 75268 codes588 = true := by decide
theorem ordered588 : Ordered 97634 75268 program588 := Coded.checked_ordered checked588
theorem length588 : program588.length = 128 := by
  rw [program588, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct588 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows588, row.Sat w) ↔ Satisfies program588 w := by
  rw [rows588, Codes.rows_expand]
  exact Coded.checked_correct checked588 h1

def rows589 := Exported.decodeRows ExportedData.chunk589.1 ExportedData.chunk589.2
def codes589 := Codes.rows ExportedData.chunk589.1 ExportedData.chunk589.2
def program589 := Coded.program 97634 75396 codes589
theorem checked589 : Coded.checkRows 97634 75396 codes589 = true := by decide
theorem ordered589 : Ordered 97634 75396 program589 := Coded.checked_ordered checked589
theorem length589 : program589.length = 128 := by
  rw [program589, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct589 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows589, row.Sat w) ↔ Satisfies program589 w := by
  rw [rows589, Codes.rows_expand]
  exact Coded.checked_correct checked589 h1

def rows590 := Exported.decodeRows ExportedData.chunk590.1 ExportedData.chunk590.2
def codes590 := Codes.rows ExportedData.chunk590.1 ExportedData.chunk590.2
def program590 := Coded.program 97634 75524 codes590
theorem checked590 : Coded.checkRows 97634 75524 codes590 = true := by decide
theorem ordered590 : Ordered 97634 75524 program590 := Coded.checked_ordered checked590
theorem length590 : program590.length = 128 := by
  rw [program590, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct590 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows590, row.Sat w) ↔ Satisfies program590 w := by
  rw [rows590, Codes.rows_expand]
  exact Coded.checked_correct checked590 h1

def rows591 := Exported.decodeRows ExportedData.chunk591.1 ExportedData.chunk591.2
def codes591 := Codes.rows ExportedData.chunk591.1 ExportedData.chunk591.2
def program591 := Coded.program 97634 75652 codes591
theorem checked591 : Coded.checkRows 97634 75652 codes591 = true := by decide
theorem ordered591 : Ordered 97634 75652 program591 := Coded.checked_ordered checked591
theorem length591 : program591.length = 128 := by
  rw [program591, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct591 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows591, row.Sat w) ↔ Satisfies program591 w := by
  rw [rows591, Codes.rows_expand]
  exact Coded.checked_correct checked591 h1

def rows592 := Exported.decodeRows ExportedData.chunk592.1 ExportedData.chunk592.2
def codes592 := Codes.rows ExportedData.chunk592.1 ExportedData.chunk592.2
def program592 := Coded.program 97634 75780 codes592
theorem checked592 : Coded.checkRows 97634 75780 codes592 = true := by decide
theorem ordered592 : Ordered 97634 75780 program592 := Coded.checked_ordered checked592
theorem length592 : program592.length = 128 := by
  rw [program592, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct592 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows592, row.Sat w) ↔ Satisfies program592 w := by
  rw [rows592, Codes.rows_expand]
  exact Coded.checked_correct checked592 h1

def rows593 := Exported.decodeRows ExportedData.chunk593.1 ExportedData.chunk593.2
def codes593 := Codes.rows ExportedData.chunk593.1 ExportedData.chunk593.2
def program593 := Coded.program 97634 75908 codes593
theorem checked593 : Coded.checkRows 97634 75908 codes593 = true := by decide
theorem ordered593 : Ordered 97634 75908 program593 := Coded.checked_ordered checked593
theorem length593 : program593.length = 128 := by
  rw [program593, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct593 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows593, row.Sat w) ↔ Satisfies program593 w := by
  rw [rows593, Codes.rows_expand]
  exact Coded.checked_correct checked593 h1

def rows594 := Exported.decodeRows ExportedData.chunk594.1 ExportedData.chunk594.2
def codes594 := Codes.rows ExportedData.chunk594.1 ExportedData.chunk594.2
def program594 := Coded.program 97634 76036 codes594
theorem checked594 : Coded.checkRows 97634 76036 codes594 = true := by decide
theorem ordered594 : Ordered 97634 76036 program594 := Coded.checked_ordered checked594
theorem length594 : program594.length = 128 := by
  rw [program594, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct594 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows594, row.Sat w) ↔ Satisfies program594 w := by
  rw [rows594, Codes.rows_expand]
  exact Coded.checked_correct checked594 h1

def rows595 := Exported.decodeRows ExportedData.chunk595.1 ExportedData.chunk595.2
def codes595 := Codes.rows ExportedData.chunk595.1 ExportedData.chunk595.2
def program595 := Coded.program 97634 76164 codes595
theorem checked595 : Coded.checkRows 97634 76164 codes595 = true := by decide
theorem ordered595 : Ordered 97634 76164 program595 := Coded.checked_ordered checked595
theorem length595 : program595.length = 128 := by
  rw [program595, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct595 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows595, row.Sat w) ↔ Satisfies program595 w := by
  rw [rows595, Codes.rows_expand]
  exact Coded.checked_correct checked595 h1

def rows596 := Exported.decodeRows ExportedData.chunk596.1 ExportedData.chunk596.2
def codes596 := Codes.rows ExportedData.chunk596.1 ExportedData.chunk596.2
def program596 := Coded.program 97634 76292 codes596
theorem checked596 : Coded.checkRows 97634 76292 codes596 = true := by decide
theorem ordered596 : Ordered 97634 76292 program596 := Coded.checked_ordered checked596
theorem length596 : program596.length = 128 := by
  rw [program596, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct596 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows596, row.Sat w) ↔ Satisfies program596 w := by
  rw [rows596, Codes.rows_expand]
  exact Coded.checked_correct checked596 h1

def rows597 := Exported.decodeRows ExportedData.chunk597.1 ExportedData.chunk597.2
def codes597 := Codes.rows ExportedData.chunk597.1 ExportedData.chunk597.2
def program597 := Coded.program 97634 76420 codes597
theorem checked597 : Coded.checkRows 97634 76420 codes597 = true := by decide
theorem ordered597 : Ordered 97634 76420 program597 := Coded.checked_ordered checked597
theorem length597 : program597.length = 128 := by
  rw [program597, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct597 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows597, row.Sat w) ↔ Satisfies program597 w := by
  rw [rows597, Codes.rows_expand]
  exact Coded.checked_correct checked597 h1

def rows598 := Exported.decodeRows ExportedData.chunk598.1 ExportedData.chunk598.2
def codes598 := Codes.rows ExportedData.chunk598.1 ExportedData.chunk598.2
def program598 := Coded.program 97634 76548 codes598
theorem checked598 : Coded.checkRows 97634 76548 codes598 = true := by decide
theorem ordered598 : Ordered 97634 76548 program598 := Coded.checked_ordered checked598
theorem length598 : program598.length = 128 := by
  rw [program598, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct598 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows598, row.Sat w) ↔ Satisfies program598 w := by
  rw [rows598, Codes.rows_expand]
  exact Coded.checked_correct checked598 h1

def rows599 := Exported.decodeRows ExportedData.chunk599.1 ExportedData.chunk599.2
def codes599 := Codes.rows ExportedData.chunk599.1 ExportedData.chunk599.2
def program599 := Coded.program 97634 76676 codes599
theorem checked599 : Coded.checkRows 97634 76676 codes599 = true := by decide
theorem ordered599 : Ordered 97634 76676 program599 := Coded.checked_ordered checked599
theorem length599 : program599.length = 128 := by
  rw [program599, Coded.program_length]
  exact Coded.codes_length _ _
theorem correct599 {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ rows599, row.Sat w) ↔ Satisfies program599 w := by
  rw [rows599, Codes.rows_expand]
  exact Coded.checked_correct checked599 h1

end CircuitCorrectness.ProgramCertificates
