import CircuitCorrectness.ProgramCertificates.Group08
import CircuitCorrectness.ProgramCertificates.Group09
import CircuitCorrectness.ProgramCertificates.Group10
import CircuitCorrectness.ProgramCertificates.Group11

set_option maxRecDepth 100000
set_option maxHeartbeats 10000000
namespace CircuitCorrectness.ProgramCertificates
open ConcreteProgram StraightLine

def rowChunks : List (List Row) := [rows540, rows541, rows542, rows543, rows544, rows545, rows546, rows547, rows548, rows549, rows550, rows551, rows552, rows553, rows554, rows555, rows556, rows557, rows558, rows559, rows560, rows561, rows562, rows563, rows564, rows565, rows566, rows567, rows568, rows569, rows570, rows571, rows572, rows573, rows574, rows575, rows576, rows577, rows578, rows579, rows580, rows581, rows582, rows583, rows584, rows585, rows586, rows587, rows588, rows589, rows590, rows591, rows592, rows593, rows594, rows595, rows596, rows597, rows598, rows599, rows600, rows601, rows602, rows603, rows604, rows605, rows606, rows607, rows608, rows609, rows610, rows611, rows612, rows613, rows614, rows615, rows616, rows617, rows618, rows619, rows620, rows621, rows622, rows623, rows624, rows625, rows626, rows627, rows628, rows629, rows630, rows631, rows632, rows633, rows634, rows635, rows636, rows637, rows638, rows639, rows640, rows641, rows642, rows643, rows644, rows645, rows646, rows647, rows648, rows649, rows650, rows651, rows652, rows653, rows654, rows655, rows656, rows657, rows658, rows659, rows660, rows661, rows662, rows663, rows664, rows665, rows666, rows667, rows668, rows669, rows670, rows671, rows672, rows673, rows674, rows675, rows676, rows677, rows678, rows679, rows680, rows681, rows682, rows683, rows684, rows685, rows686, rows687, rows688, rows689, rows690, rows691, rows692, rows693, rows694, rows695, rows696, rows697, rows698, rows699, rows700, rows701, rows702, rows703, rows704, rows705, rows706, rows707, rows708, rows709, rows710, rows711, rows712, rows713, rows714, rows715, rows716, rows717, rows718, rows719, rows720, rows721, rows722, rows723, rows724, rows725, rows726, rows727, rows728, rows729, rows730, rows731, rows732, rows733, rows734, rows735, rows736, rows737, rows738, rows739, rows740, rows741, rows742, rows743, rows744, rows745, rows746, rows747, rows748, rows749, rows750, rows751, rows752, rows753, rows754, rows755, rows756, rows757, rows758, rows759, rows760, rows761, rows762]
def programChunks : List Program := [program540, program541, program542, program543, program544, program545, program546, program547, program548, program549, program550, program551, program552, program553, program554, program555, program556, program557, program558, program559, program560, program561, program562, program563, program564, program565, program566, program567, program568, program569, program570, program571, program572, program573, program574, program575, program576, program577, program578, program579, program580, program581, program582, program583, program584, program585, program586, program587, program588, program589, program590, program591, program592, program593, program594, program595, program596, program597, program598, program599, program600, program601, program602, program603, program604, program605, program606, program607, program608, program609, program610, program611, program612, program613, program614, program615, program616, program617, program618, program619, program620, program621, program622, program623, program624, program625, program626, program627, program628, program629, program630, program631, program632, program633, program634, program635, program636, program637, program638, program639, program640, program641, program642, program643, program644, program645, program646, program647, program648, program649, program650, program651, program652, program653, program654, program655, program656, program657, program658, program659, program660, program661, program662, program663, program664, program665, program666, program667, program668, program669, program670, program671, program672, program673, program674, program675, program676, program677, program678, program679, program680, program681, program682, program683, program684, program685, program686, program687, program688, program689, program690, program691, program692, program693, program694, program695, program696, program697, program698, program699, program700, program701, program702, program703, program704, program705, program706, program707, program708, program709, program710, program711, program712, program713, program714, program715, program716, program717, program718, program719, program720, program721, program722, program723, program724, program725, program726, program727, program728, program729, program730, program731, program732, program733, program734, program735, program736, program737, program738, program739, program740, program741, program742, program743, program744, program745, program746, program747, program748, program749, program750, program751, program752, program753, program754, program755, program756, program757, program758, program759, program760, program761, program762]
def program : Program := programChunks.flatten

theorem rowChunks_eq : rowChunks =
    (ExportedData.chunks.toList.drop 540).map
      (fun chunk => Exported.decodeRows chunk.1 chunk.2) := by rfl

theorem actual_suffix : Exported.rows.drop 69120 = rowChunks.flatten := by
  rw [rowChunks_eq, ← List.flatMap_def]
  have hl : ((ExportedData.chunks.toList.take 540).flatMap
      (fun chunk => Exported.decodeRows chunk.1 chunk.2)).length = 69120 := by
    simp only [List.length_flatMap, Exported.decodeRows_length]
    decide
  have hs := List.take_append_drop 540 ExportedData.chunks.toList
  unfold Exported.rows
  conv_lhs => rw [← hs, List.flatMap_append]
  exact List.drop_left' hl

theorem chunks_correct : List.Forall₂
    (fun rs ps => ∀ w : Assignment, w 97634 = 1 →
      ((∀ row ∈ rs, row.Sat w) ↔ Satisfies ps w)) rowChunks programChunks := by
  unfold rowChunks programChunks
  refine .cons (fun w h1 => correct540 h1) ?_
  refine .cons (fun w h1 => correct541 h1) ?_
  refine .cons (fun w h1 => correct542 h1) ?_
  refine .cons (fun w h1 => correct543 h1) ?_
  refine .cons (fun w h1 => correct544 h1) ?_
  refine .cons (fun w h1 => correct545 h1) ?_
  refine .cons (fun w h1 => correct546 h1) ?_
  refine .cons (fun w h1 => correct547 h1) ?_
  refine .cons (fun w h1 => correct548 h1) ?_
  refine .cons (fun w h1 => correct549 h1) ?_
  refine .cons (fun w h1 => correct550 h1) ?_
  refine .cons (fun w h1 => correct551 h1) ?_
  refine .cons (fun w h1 => correct552 h1) ?_
  refine .cons (fun w h1 => correct553 h1) ?_
  refine .cons (fun w h1 => correct554 h1) ?_
  refine .cons (fun w h1 => correct555 h1) ?_
  refine .cons (fun w h1 => correct556 h1) ?_
  refine .cons (fun w h1 => correct557 h1) ?_
  refine .cons (fun w h1 => correct558 h1) ?_
  refine .cons (fun w h1 => correct559 h1) ?_
  refine .cons (fun w h1 => correct560 h1) ?_
  refine .cons (fun w h1 => correct561 h1) ?_
  refine .cons (fun w h1 => correct562 h1) ?_
  refine .cons (fun w h1 => correct563 h1) ?_
  refine .cons (fun w h1 => correct564 h1) ?_
  refine .cons (fun w h1 => correct565 h1) ?_
  refine .cons (fun w h1 => correct566 h1) ?_
  refine .cons (fun w h1 => correct567 h1) ?_
  refine .cons (fun w h1 => correct568 h1) ?_
  refine .cons (fun w h1 => correct569 h1) ?_
  refine .cons (fun w h1 => correct570 h1) ?_
  refine .cons (fun w h1 => correct571 h1) ?_
  refine .cons (fun w h1 => correct572 h1) ?_
  refine .cons (fun w h1 => correct573 h1) ?_
  refine .cons (fun w h1 => correct574 h1) ?_
  refine .cons (fun w h1 => correct575 h1) ?_
  refine .cons (fun w h1 => correct576 h1) ?_
  refine .cons (fun w h1 => correct577 h1) ?_
  refine .cons (fun w h1 => correct578 h1) ?_
  refine .cons (fun w h1 => correct579 h1) ?_
  refine .cons (fun w h1 => correct580 h1) ?_
  refine .cons (fun w h1 => correct581 h1) ?_
  refine .cons (fun w h1 => correct582 h1) ?_
  refine .cons (fun w h1 => correct583 h1) ?_
  refine .cons (fun w h1 => correct584 h1) ?_
  refine .cons (fun w h1 => correct585 h1) ?_
  refine .cons (fun w h1 => correct586 h1) ?_
  refine .cons (fun w h1 => correct587 h1) ?_
  refine .cons (fun w h1 => correct588 h1) ?_
  refine .cons (fun w h1 => correct589 h1) ?_
  refine .cons (fun w h1 => correct590 h1) ?_
  refine .cons (fun w h1 => correct591 h1) ?_
  refine .cons (fun w h1 => correct592 h1) ?_
  refine .cons (fun w h1 => correct593 h1) ?_
  refine .cons (fun w h1 => correct594 h1) ?_
  refine .cons (fun w h1 => correct595 h1) ?_
  refine .cons (fun w h1 => correct596 h1) ?_
  refine .cons (fun w h1 => correct597 h1) ?_
  refine .cons (fun w h1 => correct598 h1) ?_
  refine .cons (fun w h1 => correct599 h1) ?_
  refine .cons (fun w h1 => correct600 h1) ?_
  refine .cons (fun w h1 => correct601 h1) ?_
  refine .cons (fun w h1 => correct602 h1) ?_
  refine .cons (fun w h1 => correct603 h1) ?_
  refine .cons (fun w h1 => correct604 h1) ?_
  refine .cons (fun w h1 => correct605 h1) ?_
  refine .cons (fun w h1 => correct606 h1) ?_
  refine .cons (fun w h1 => correct607 h1) ?_
  refine .cons (fun w h1 => correct608 h1) ?_
  refine .cons (fun w h1 => correct609 h1) ?_
  refine .cons (fun w h1 => correct610 h1) ?_
  refine .cons (fun w h1 => correct611 h1) ?_
  refine .cons (fun w h1 => correct612 h1) ?_
  refine .cons (fun w h1 => correct613 h1) ?_
  refine .cons (fun w h1 => correct614 h1) ?_
  refine .cons (fun w h1 => correct615 h1) ?_
  refine .cons (fun w h1 => correct616 h1) ?_
  refine .cons (fun w h1 => correct617 h1) ?_
  refine .cons (fun w h1 => correct618 h1) ?_
  refine .cons (fun w h1 => correct619 h1) ?_
  refine .cons (fun w h1 => correct620 h1) ?_
  refine .cons (fun w h1 => correct621 h1) ?_
  refine .cons (fun w h1 => correct622 h1) ?_
  refine .cons (fun w h1 => correct623 h1) ?_
  refine .cons (fun w h1 => correct624 h1) ?_
  refine .cons (fun w h1 => correct625 h1) ?_
  refine .cons (fun w h1 => correct626 h1) ?_
  refine .cons (fun w h1 => correct627 h1) ?_
  refine .cons (fun w h1 => correct628 h1) ?_
  refine .cons (fun w h1 => correct629 h1) ?_
  refine .cons (fun w h1 => correct630 h1) ?_
  refine .cons (fun w h1 => correct631 h1) ?_
  refine .cons (fun w h1 => correct632 h1) ?_
  refine .cons (fun w h1 => correct633 h1) ?_
  refine .cons (fun w h1 => correct634 h1) ?_
  refine .cons (fun w h1 => correct635 h1) ?_
  refine .cons (fun w h1 => correct636 h1) ?_
  refine .cons (fun w h1 => correct637 h1) ?_
  refine .cons (fun w h1 => correct638 h1) ?_
  refine .cons (fun w h1 => correct639 h1) ?_
  refine .cons (fun w h1 => correct640 h1) ?_
  refine .cons (fun w h1 => correct641 h1) ?_
  refine .cons (fun w h1 => correct642 h1) ?_
  refine .cons (fun w h1 => correct643 h1) ?_
  refine .cons (fun w h1 => correct644 h1) ?_
  refine .cons (fun w h1 => correct645 h1) ?_
  refine .cons (fun w h1 => correct646 h1) ?_
  refine .cons (fun w h1 => correct647 h1) ?_
  refine .cons (fun w h1 => correct648 h1) ?_
  refine .cons (fun w h1 => correct649 h1) ?_
  refine .cons (fun w h1 => correct650 h1) ?_
  refine .cons (fun w h1 => correct651 h1) ?_
  refine .cons (fun w h1 => correct652 h1) ?_
  refine .cons (fun w h1 => correct653 h1) ?_
  refine .cons (fun w h1 => correct654 h1) ?_
  refine .cons (fun w h1 => correct655 h1) ?_
  refine .cons (fun w h1 => correct656 h1) ?_
  refine .cons (fun w h1 => correct657 h1) ?_
  refine .cons (fun w h1 => correct658 h1) ?_
  refine .cons (fun w h1 => correct659 h1) ?_
  refine .cons (fun w h1 => correct660 h1) ?_
  refine .cons (fun w h1 => correct661 h1) ?_
  refine .cons (fun w h1 => correct662 h1) ?_
  refine .cons (fun w h1 => correct663 h1) ?_
  refine .cons (fun w h1 => correct664 h1) ?_
  refine .cons (fun w h1 => correct665 h1) ?_
  refine .cons (fun w h1 => correct666 h1) ?_
  refine .cons (fun w h1 => correct667 h1) ?_
  refine .cons (fun w h1 => correct668 h1) ?_
  refine .cons (fun w h1 => correct669 h1) ?_
  refine .cons (fun w h1 => correct670 h1) ?_
  refine .cons (fun w h1 => correct671 h1) ?_
  refine .cons (fun w h1 => correct672 h1) ?_
  refine .cons (fun w h1 => correct673 h1) ?_
  refine .cons (fun w h1 => correct674 h1) ?_
  refine .cons (fun w h1 => correct675 h1) ?_
  refine .cons (fun w h1 => correct676 h1) ?_
  refine .cons (fun w h1 => correct677 h1) ?_
  refine .cons (fun w h1 => correct678 h1) ?_
  refine .cons (fun w h1 => correct679 h1) ?_
  refine .cons (fun w h1 => correct680 h1) ?_
  refine .cons (fun w h1 => correct681 h1) ?_
  refine .cons (fun w h1 => correct682 h1) ?_
  refine .cons (fun w h1 => correct683 h1) ?_
  refine .cons (fun w h1 => correct684 h1) ?_
  refine .cons (fun w h1 => correct685 h1) ?_
  refine .cons (fun w h1 => correct686 h1) ?_
  refine .cons (fun w h1 => correct687 h1) ?_
  refine .cons (fun w h1 => correct688 h1) ?_
  refine .cons (fun w h1 => correct689 h1) ?_
  refine .cons (fun w h1 => correct690 h1) ?_
  refine .cons (fun w h1 => correct691 h1) ?_
  refine .cons (fun w h1 => correct692 h1) ?_
  refine .cons (fun w h1 => correct693 h1) ?_
  refine .cons (fun w h1 => correct694 h1) ?_
  refine .cons (fun w h1 => correct695 h1) ?_
  refine .cons (fun w h1 => correct696 h1) ?_
  refine .cons (fun w h1 => correct697 h1) ?_
  refine .cons (fun w h1 => correct698 h1) ?_
  refine .cons (fun w h1 => correct699 h1) ?_
  refine .cons (fun w h1 => correct700 h1) ?_
  refine .cons (fun w h1 => correct701 h1) ?_
  refine .cons (fun w h1 => correct702 h1) ?_
  refine .cons (fun w h1 => correct703 h1) ?_
  refine .cons (fun w h1 => correct704 h1) ?_
  refine .cons (fun w h1 => correct705 h1) ?_
  refine .cons (fun w h1 => correct706 h1) ?_
  refine .cons (fun w h1 => correct707 h1) ?_
  refine .cons (fun w h1 => correct708 h1) ?_
  refine .cons (fun w h1 => correct709 h1) ?_
  refine .cons (fun w h1 => correct710 h1) ?_
  refine .cons (fun w h1 => correct711 h1) ?_
  refine .cons (fun w h1 => correct712 h1) ?_
  refine .cons (fun w h1 => correct713 h1) ?_
  refine .cons (fun w h1 => correct714 h1) ?_
  refine .cons (fun w h1 => correct715 h1) ?_
  refine .cons (fun w h1 => correct716 h1) ?_
  refine .cons (fun w h1 => correct717 h1) ?_
  refine .cons (fun w h1 => correct718 h1) ?_
  refine .cons (fun w h1 => correct719 h1) ?_
  refine .cons (fun w h1 => correct720 h1) ?_
  refine .cons (fun w h1 => correct721 h1) ?_
  refine .cons (fun w h1 => correct722 h1) ?_
  refine .cons (fun w h1 => correct723 h1) ?_
  refine .cons (fun w h1 => correct724 h1) ?_
  refine .cons (fun w h1 => correct725 h1) ?_
  refine .cons (fun w h1 => correct726 h1) ?_
  refine .cons (fun w h1 => correct727 h1) ?_
  refine .cons (fun w h1 => correct728 h1) ?_
  refine .cons (fun w h1 => correct729 h1) ?_
  refine .cons (fun w h1 => correct730 h1) ?_
  refine .cons (fun w h1 => correct731 h1) ?_
  refine .cons (fun w h1 => correct732 h1) ?_
  refine .cons (fun w h1 => correct733 h1) ?_
  refine .cons (fun w h1 => correct734 h1) ?_
  refine .cons (fun w h1 => correct735 h1) ?_
  refine .cons (fun w h1 => correct736 h1) ?_
  refine .cons (fun w h1 => correct737 h1) ?_
  refine .cons (fun w h1 => correct738 h1) ?_
  refine .cons (fun w h1 => correct739 h1) ?_
  refine .cons (fun w h1 => correct740 h1) ?_
  refine .cons (fun w h1 => correct741 h1) ?_
  refine .cons (fun w h1 => correct742 h1) ?_
  refine .cons (fun w h1 => correct743 h1) ?_
  refine .cons (fun w h1 => correct744 h1) ?_
  refine .cons (fun w h1 => correct745 h1) ?_
  refine .cons (fun w h1 => correct746 h1) ?_
  refine .cons (fun w h1 => correct747 h1) ?_
  refine .cons (fun w h1 => correct748 h1) ?_
  refine .cons (fun w h1 => correct749 h1) ?_
  refine .cons (fun w h1 => correct750 h1) ?_
  refine .cons (fun w h1 => correct751 h1) ?_
  refine .cons (fun w h1 => correct752 h1) ?_
  refine .cons (fun w h1 => correct753 h1) ?_
  refine .cons (fun w h1 => correct754 h1) ?_
  refine .cons (fun w h1 => correct755 h1) ?_
  refine .cons (fun w h1 => correct756 h1) ?_
  refine .cons (fun w h1 => correct757 h1) ?_
  refine .cons (fun w h1 => correct758 h1) ?_
  refine .cons (fun w h1 => correct759 h1) ?_
  refine .cons (fun w h1 => correct760 h1) ?_
  refine .cons (fun w h1 => correct761 h1) ?_
  refine .cons (fun w h1 => correct762 h1) ?_
  exact .nil

theorem correct {w : Assignment} (h1 : w 97634 = 1) :
    (∀ row ∈ Exported.rows.drop 69120, row.Sat w) ↔ Satisfies program w := by
  rw [actual_suffix]
  exact flatten_correct chunks_correct h1

theorem length : program.length = 28510 := by
  simp only [program, programChunks, List.length_flatten, List.map_cons,
    List.map_nil, List.sum_cons, List.sum_nil,
    length540, length541, length542, length543, length544, length545, length546, length547, length548, length549, length550, length551, length552, length553, length554, length555, length556, length557, length558, length559, length560, length561, length562, length563, length564, length565, length566, length567, length568, length569, length570, length571, length572, length573, length574, length575, length576, length577, length578, length579, length580, length581, length582, length583, length584, length585, length586, length587, length588, length589, length590, length591, length592, length593, length594, length595, length596, length597, length598, length599, length600, length601, length602, length603, length604, length605, length606, length607, length608, length609, length610, length611, length612, length613, length614, length615, length616, length617, length618, length619, length620, length621, length622, length623, length624, length625, length626, length627, length628, length629, length630, length631, length632, length633, length634, length635, length636, length637, length638, length639, length640, length641, length642, length643, length644, length645, length646, length647, length648, length649, length650, length651, length652, length653, length654, length655, length656, length657, length658, length659, length660, length661, length662, length663, length664, length665, length666, length667, length668, length669, length670, length671, length672, length673, length674, length675, length676, length677, length678, length679, length680, length681, length682, length683, length684, length685, length686, length687, length688, length689, length690, length691, length692, length693, length694, length695, length696, length697, length698, length699, length700, length701, length702, length703, length704, length705, length706, length707, length708, length709, length710, length711, length712, length713, length714, length715, length716, length717, length718, length719, length720, length721, length722, length723, length724, length725, length726, length727, length728, length729, length730, length731, length732, length733, length734, length735, length736, length737, length738, length739, length740, length741, length742, length743, length744, length745, length746, length747, length748, length749, length750, length751, length752, length753, length754, length755, length756, length757, length758, length759, length760, length761, length762]
  rfl

theorem ordered : Ordered 97634 69124 program := by
  unfold program programChunks
  simp only [List.flatten_cons, List.flatten_nil]
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered540, ?_⟩
  rw [length540]
  change Ordered 97634 69252 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered541, ?_⟩
  rw [length541]
  change Ordered 97634 69380 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered542, ?_⟩
  rw [length542]
  change Ordered 97634 69508 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered543, ?_⟩
  rw [length543]
  change Ordered 97634 69636 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered544, ?_⟩
  rw [length544]
  change Ordered 97634 69764 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered545, ?_⟩
  rw [length545]
  change Ordered 97634 69892 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered546, ?_⟩
  rw [length546]
  change Ordered 97634 70020 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered547, ?_⟩
  rw [length547]
  change Ordered 97634 70148 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered548, ?_⟩
  rw [length548]
  change Ordered 97634 70276 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered549, ?_⟩
  rw [length549]
  change Ordered 97634 70404 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered550, ?_⟩
  rw [length550]
  change Ordered 97634 70532 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered551, ?_⟩
  rw [length551]
  change Ordered 97634 70660 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered552, ?_⟩
  rw [length552]
  change Ordered 97634 70788 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered553, ?_⟩
  rw [length553]
  change Ordered 97634 70916 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered554, ?_⟩
  rw [length554]
  change Ordered 97634 71044 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered555, ?_⟩
  rw [length555]
  change Ordered 97634 71172 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered556, ?_⟩
  rw [length556]
  change Ordered 97634 71300 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered557, ?_⟩
  rw [length557]
  change Ordered 97634 71428 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered558, ?_⟩
  rw [length558]
  change Ordered 97634 71556 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered559, ?_⟩
  rw [length559]
  change Ordered 97634 71684 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered560, ?_⟩
  rw [length560]
  change Ordered 97634 71812 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered561, ?_⟩
  rw [length561]
  change Ordered 97634 71940 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered562, ?_⟩
  rw [length562]
  change Ordered 97634 72068 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered563, ?_⟩
  rw [length563]
  change Ordered 97634 72196 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered564, ?_⟩
  rw [length564]
  change Ordered 97634 72324 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered565, ?_⟩
  rw [length565]
  change Ordered 97634 72452 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered566, ?_⟩
  rw [length566]
  change Ordered 97634 72580 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered567, ?_⟩
  rw [length567]
  change Ordered 97634 72708 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered568, ?_⟩
  rw [length568]
  change Ordered 97634 72836 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered569, ?_⟩
  rw [length569]
  change Ordered 97634 72964 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered570, ?_⟩
  rw [length570]
  change Ordered 97634 73092 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered571, ?_⟩
  rw [length571]
  change Ordered 97634 73220 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered572, ?_⟩
  rw [length572]
  change Ordered 97634 73348 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered573, ?_⟩
  rw [length573]
  change Ordered 97634 73476 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered574, ?_⟩
  rw [length574]
  change Ordered 97634 73604 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered575, ?_⟩
  rw [length575]
  change Ordered 97634 73732 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered576, ?_⟩
  rw [length576]
  change Ordered 97634 73860 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered577, ?_⟩
  rw [length577]
  change Ordered 97634 73988 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered578, ?_⟩
  rw [length578]
  change Ordered 97634 74116 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered579, ?_⟩
  rw [length579]
  change Ordered 97634 74244 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered580, ?_⟩
  rw [length580]
  change Ordered 97634 74372 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered581, ?_⟩
  rw [length581]
  change Ordered 97634 74500 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered582, ?_⟩
  rw [length582]
  change Ordered 97634 74628 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered583, ?_⟩
  rw [length583]
  change Ordered 97634 74756 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered584, ?_⟩
  rw [length584]
  change Ordered 97634 74884 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered585, ?_⟩
  rw [length585]
  change Ordered 97634 75012 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered586, ?_⟩
  rw [length586]
  change Ordered 97634 75140 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered587, ?_⟩
  rw [length587]
  change Ordered 97634 75268 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered588, ?_⟩
  rw [length588]
  change Ordered 97634 75396 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered589, ?_⟩
  rw [length589]
  change Ordered 97634 75524 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered590, ?_⟩
  rw [length590]
  change Ordered 97634 75652 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered591, ?_⟩
  rw [length591]
  change Ordered 97634 75780 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered592, ?_⟩
  rw [length592]
  change Ordered 97634 75908 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered593, ?_⟩
  rw [length593]
  change Ordered 97634 76036 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered594, ?_⟩
  rw [length594]
  change Ordered 97634 76164 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered595, ?_⟩
  rw [length595]
  change Ordered 97634 76292 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered596, ?_⟩
  rw [length596]
  change Ordered 97634 76420 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered597, ?_⟩
  rw [length597]
  change Ordered 97634 76548 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered598, ?_⟩
  rw [length598]
  change Ordered 97634 76676 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered599, ?_⟩
  rw [length599]
  change Ordered 97634 76804 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered600, ?_⟩
  rw [length600]
  change Ordered 97634 76932 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered601, ?_⟩
  rw [length601]
  change Ordered 97634 77060 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered602, ?_⟩
  rw [length602]
  change Ordered 97634 77188 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered603, ?_⟩
  rw [length603]
  change Ordered 97634 77316 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered604, ?_⟩
  rw [length604]
  change Ordered 97634 77444 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered605, ?_⟩
  rw [length605]
  change Ordered 97634 77572 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered606, ?_⟩
  rw [length606]
  change Ordered 97634 77700 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered607, ?_⟩
  rw [length607]
  change Ordered 97634 77828 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered608, ?_⟩
  rw [length608]
  change Ordered 97634 77956 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered609, ?_⟩
  rw [length609]
  change Ordered 97634 78084 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered610, ?_⟩
  rw [length610]
  change Ordered 97634 78212 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered611, ?_⟩
  rw [length611]
  change Ordered 97634 78340 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered612, ?_⟩
  rw [length612]
  change Ordered 97634 78468 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered613, ?_⟩
  rw [length613]
  change Ordered 97634 78596 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered614, ?_⟩
  rw [length614]
  change Ordered 97634 78724 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered615, ?_⟩
  rw [length615]
  change Ordered 97634 78852 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered616, ?_⟩
  rw [length616]
  change Ordered 97634 78980 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered617, ?_⟩
  rw [length617]
  change Ordered 97634 79108 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered618, ?_⟩
  rw [length618]
  change Ordered 97634 79236 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered619, ?_⟩
  rw [length619]
  change Ordered 97634 79364 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered620, ?_⟩
  rw [length620]
  change Ordered 97634 79492 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered621, ?_⟩
  rw [length621]
  change Ordered 97634 79620 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered622, ?_⟩
  rw [length622]
  change Ordered 97634 79748 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered623, ?_⟩
  rw [length623]
  change Ordered 97634 79876 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered624, ?_⟩
  rw [length624]
  change Ordered 97634 80004 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered625, ?_⟩
  rw [length625]
  change Ordered 97634 80132 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered626, ?_⟩
  rw [length626]
  change Ordered 97634 80260 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered627, ?_⟩
  rw [length627]
  change Ordered 97634 80388 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered628, ?_⟩
  rw [length628]
  change Ordered 97634 80516 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered629, ?_⟩
  rw [length629]
  change Ordered 97634 80644 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered630, ?_⟩
  rw [length630]
  change Ordered 97634 80772 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered631, ?_⟩
  rw [length631]
  change Ordered 97634 80900 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered632, ?_⟩
  rw [length632]
  change Ordered 97634 81028 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered633, ?_⟩
  rw [length633]
  change Ordered 97634 81156 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered634, ?_⟩
  rw [length634]
  change Ordered 97634 81284 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered635, ?_⟩
  rw [length635]
  change Ordered 97634 81412 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered636, ?_⟩
  rw [length636]
  change Ordered 97634 81540 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered637, ?_⟩
  rw [length637]
  change Ordered 97634 81668 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered638, ?_⟩
  rw [length638]
  change Ordered 97634 81796 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered639, ?_⟩
  rw [length639]
  change Ordered 97634 81924 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered640, ?_⟩
  rw [length640]
  change Ordered 97634 82052 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered641, ?_⟩
  rw [length641]
  change Ordered 97634 82180 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered642, ?_⟩
  rw [length642]
  change Ordered 97634 82308 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered643, ?_⟩
  rw [length643]
  change Ordered 97634 82436 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered644, ?_⟩
  rw [length644]
  change Ordered 97634 82564 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered645, ?_⟩
  rw [length645]
  change Ordered 97634 82692 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered646, ?_⟩
  rw [length646]
  change Ordered 97634 82820 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered647, ?_⟩
  rw [length647]
  change Ordered 97634 82948 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered648, ?_⟩
  rw [length648]
  change Ordered 97634 83076 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered649, ?_⟩
  rw [length649]
  change Ordered 97634 83204 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered650, ?_⟩
  rw [length650]
  change Ordered 97634 83332 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered651, ?_⟩
  rw [length651]
  change Ordered 97634 83460 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered652, ?_⟩
  rw [length652]
  change Ordered 97634 83588 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered653, ?_⟩
  rw [length653]
  change Ordered 97634 83716 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered654, ?_⟩
  rw [length654]
  change Ordered 97634 83844 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered655, ?_⟩
  rw [length655]
  change Ordered 97634 83972 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered656, ?_⟩
  rw [length656]
  change Ordered 97634 84100 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered657, ?_⟩
  rw [length657]
  change Ordered 97634 84228 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered658, ?_⟩
  rw [length658]
  change Ordered 97634 84356 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered659, ?_⟩
  rw [length659]
  change Ordered 97634 84484 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered660, ?_⟩
  rw [length660]
  change Ordered 97634 84612 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered661, ?_⟩
  rw [length661]
  change Ordered 97634 84740 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered662, ?_⟩
  rw [length662]
  change Ordered 97634 84868 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered663, ?_⟩
  rw [length663]
  change Ordered 97634 84996 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered664, ?_⟩
  rw [length664]
  change Ordered 97634 85124 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered665, ?_⟩
  rw [length665]
  change Ordered 97634 85252 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered666, ?_⟩
  rw [length666]
  change Ordered 97634 85380 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered667, ?_⟩
  rw [length667]
  change Ordered 97634 85508 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered668, ?_⟩
  rw [length668]
  change Ordered 97634 85636 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered669, ?_⟩
  rw [length669]
  change Ordered 97634 85764 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered670, ?_⟩
  rw [length670]
  change Ordered 97634 85892 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered671, ?_⟩
  rw [length671]
  change Ordered 97634 86020 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered672, ?_⟩
  rw [length672]
  change Ordered 97634 86148 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered673, ?_⟩
  rw [length673]
  change Ordered 97634 86276 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered674, ?_⟩
  rw [length674]
  change Ordered 97634 86404 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered675, ?_⟩
  rw [length675]
  change Ordered 97634 86532 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered676, ?_⟩
  rw [length676]
  change Ordered 97634 86660 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered677, ?_⟩
  rw [length677]
  change Ordered 97634 86788 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered678, ?_⟩
  rw [length678]
  change Ordered 97634 86916 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered679, ?_⟩
  rw [length679]
  change Ordered 97634 87044 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered680, ?_⟩
  rw [length680]
  change Ordered 97634 87172 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered681, ?_⟩
  rw [length681]
  change Ordered 97634 87300 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered682, ?_⟩
  rw [length682]
  change Ordered 97634 87428 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered683, ?_⟩
  rw [length683]
  change Ordered 97634 87556 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered684, ?_⟩
  rw [length684]
  change Ordered 97634 87684 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered685, ?_⟩
  rw [length685]
  change Ordered 97634 87812 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered686, ?_⟩
  rw [length686]
  change Ordered 97634 87940 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered687, ?_⟩
  rw [length687]
  change Ordered 97634 88068 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered688, ?_⟩
  rw [length688]
  change Ordered 97634 88196 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered689, ?_⟩
  rw [length689]
  change Ordered 97634 88324 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered690, ?_⟩
  rw [length690]
  change Ordered 97634 88452 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered691, ?_⟩
  rw [length691]
  change Ordered 97634 88580 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered692, ?_⟩
  rw [length692]
  change Ordered 97634 88708 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered693, ?_⟩
  rw [length693]
  change Ordered 97634 88836 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered694, ?_⟩
  rw [length694]
  change Ordered 97634 88964 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered695, ?_⟩
  rw [length695]
  change Ordered 97634 89092 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered696, ?_⟩
  rw [length696]
  change Ordered 97634 89220 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered697, ?_⟩
  rw [length697]
  change Ordered 97634 89348 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered698, ?_⟩
  rw [length698]
  change Ordered 97634 89476 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered699, ?_⟩
  rw [length699]
  change Ordered 97634 89604 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered700, ?_⟩
  rw [length700]
  change Ordered 97634 89732 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered701, ?_⟩
  rw [length701]
  change Ordered 97634 89860 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered702, ?_⟩
  rw [length702]
  change Ordered 97634 89988 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered703, ?_⟩
  rw [length703]
  change Ordered 97634 90116 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered704, ?_⟩
  rw [length704]
  change Ordered 97634 90244 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered705, ?_⟩
  rw [length705]
  change Ordered 97634 90372 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered706, ?_⟩
  rw [length706]
  change Ordered 97634 90500 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered707, ?_⟩
  rw [length707]
  change Ordered 97634 90628 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered708, ?_⟩
  rw [length708]
  change Ordered 97634 90756 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered709, ?_⟩
  rw [length709]
  change Ordered 97634 90884 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered710, ?_⟩
  rw [length710]
  change Ordered 97634 91012 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered711, ?_⟩
  rw [length711]
  change Ordered 97634 91140 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered712, ?_⟩
  rw [length712]
  change Ordered 97634 91268 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered713, ?_⟩
  rw [length713]
  change Ordered 97634 91396 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered714, ?_⟩
  rw [length714]
  change Ordered 97634 91524 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered715, ?_⟩
  rw [length715]
  change Ordered 97634 91652 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered716, ?_⟩
  rw [length716]
  change Ordered 97634 91780 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered717, ?_⟩
  rw [length717]
  change Ordered 97634 91908 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered718, ?_⟩
  rw [length718]
  change Ordered 97634 92036 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered719, ?_⟩
  rw [length719]
  change Ordered 97634 92164 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered720, ?_⟩
  rw [length720]
  change Ordered 97634 92292 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered721, ?_⟩
  rw [length721]
  change Ordered 97634 92420 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered722, ?_⟩
  rw [length722]
  change Ordered 97634 92548 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered723, ?_⟩
  rw [length723]
  change Ordered 97634 92676 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered724, ?_⟩
  rw [length724]
  change Ordered 97634 92804 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered725, ?_⟩
  rw [length725]
  change Ordered 97634 92932 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered726, ?_⟩
  rw [length726]
  change Ordered 97634 93060 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered727, ?_⟩
  rw [length727]
  change Ordered 97634 93188 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered728, ?_⟩
  rw [length728]
  change Ordered 97634 93316 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered729, ?_⟩
  rw [length729]
  change Ordered 97634 93444 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered730, ?_⟩
  rw [length730]
  change Ordered 97634 93572 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered731, ?_⟩
  rw [length731]
  change Ordered 97634 93700 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered732, ?_⟩
  rw [length732]
  change Ordered 97634 93828 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered733, ?_⟩
  rw [length733]
  change Ordered 97634 93956 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered734, ?_⟩
  rw [length734]
  change Ordered 97634 94084 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered735, ?_⟩
  rw [length735]
  change Ordered 97634 94212 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered736, ?_⟩
  rw [length736]
  change Ordered 97634 94340 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered737, ?_⟩
  rw [length737]
  change Ordered 97634 94468 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered738, ?_⟩
  rw [length738]
  change Ordered 97634 94596 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered739, ?_⟩
  rw [length739]
  change Ordered 97634 94724 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered740, ?_⟩
  rw [length740]
  change Ordered 97634 94852 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered741, ?_⟩
  rw [length741]
  change Ordered 97634 94980 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered742, ?_⟩
  rw [length742]
  change Ordered 97634 95108 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered743, ?_⟩
  rw [length743]
  change Ordered 97634 95236 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered744, ?_⟩
  rw [length744]
  change Ordered 97634 95364 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered745, ?_⟩
  rw [length745]
  change Ordered 97634 95492 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered746, ?_⟩
  rw [length746]
  change Ordered 97634 95620 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered747, ?_⟩
  rw [length747]
  change Ordered 97634 95748 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered748, ?_⟩
  rw [length748]
  change Ordered 97634 95876 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered749, ?_⟩
  rw [length749]
  change Ordered 97634 96004 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered750, ?_⟩
  rw [length750]
  change Ordered 97634 96132 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered751, ?_⟩
  rw [length751]
  change Ordered 97634 96260 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered752, ?_⟩
  rw [length752]
  change Ordered 97634 96388 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered753, ?_⟩
  rw [length753]
  change Ordered 97634 96516 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered754, ?_⟩
  rw [length754]
  change Ordered 97634 96644 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered755, ?_⟩
  rw [length755]
  change Ordered 97634 96772 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered756, ?_⟩
  rw [length756]
  change Ordered 97634 96900 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered757, ?_⟩
  rw [length757]
  change Ordered 97634 97028 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered758, ?_⟩
  rw [length758]
  change Ordered 97634 97156 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered759, ?_⟩
  rw [length759]
  change Ordered 97634 97284 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered760, ?_⟩
  rw [length760]
  change Ordered 97634 97412 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered761, ?_⟩
  rw [length761]
  change Ordered 97634 97540 _
  apply (ordered_append _ _ _ _).mpr
  refine ⟨ordered762, ?_⟩
  rw [length762]
  change Ordered 97634 97634 _
  trivial

theorem wellFormed : WellFormed (Frontier 69124 97634) program :=
  ordered_wellFormed ordered

theorem knownAfter : KnownAfter (Frontier 69124 97634) program = Frontier 97634 97634 := by
  rw [ordered_knownAfter ordered, length]

/-- A mathematical assignment satisfying every post-byte exported row. -/
theorem execution_satisfies (seed : Assignment) (h1 : seed 97634 = 1) :
    (∀ row ∈ Exported.rows.drop 69120, row.Sat (run program seed)) := by
  apply (correct ?_).mpr (run_satisfies wellFormed)
  exact (run_preserves_known wellFormed 97634 (Or.inr rfl)).trans h1

/-- Every already allocated wire, including the constant-one wire, is preserved. -/
theorem execution_preserves (seed : Assignment) :
    Agree (Frontier 69124 97634) (run program seed) seed :=
  run_preserves_known wellFormed

end CircuitCorrectness.ProgramCertificates
