import CircuitCorrectness.Pratt

set_option maxRecDepth 4096
set_option maxHeartbeats 4000000
namespace CircuitCorrectness
namespace Primality

theorem prime_2 : Nat.Prime 2 := Nat.prime_two

theorem prime_3 : Nat.Prime 3 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)

theorem prime_7 : Nat.Prime 7 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)

theorem prime_5 : Nat.Prime 5 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)

theorem prime_61 : Nat.Prime 61 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 1, 5 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)

theorem prime_1709 : Nat.Prime 1709 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 7 ^ 1, 61 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 61 1 _ prime_61 (by reduce_mod_char; decide) (by norm_num)

theorem prime_11 : Nat.Prime 11 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 5 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)

theorem prime_23 : Nat.Prime 23 := by
  refine PrattCertificate'.out ⟨5, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 11 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 11 1 _ prime_11 (by reduce_mod_char; decide) (by norm_num)

theorem prime_1381 : Nat.Prime 1381 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 1, 5 ^ 1, 23 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 23 1 _ prime_23 (by reduce_mod_char; decide) (by norm_num)

theorem prime_24859 : Nat.Prime 24859 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 2, 1381 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 1381 1 _ prime_1381 (by reduce_mod_char; decide) (by norm_num)

theorem prime_13 : Nat.Prime 13 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)

theorem prime_89 : Nat.Prime 89 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 3, 11 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 3 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 11 1 _ prime_11 (by reduce_mod_char; decide) (by norm_num)

theorem prime_179 : Nat.Prime 179 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 89 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 89 1 _ prime_89 (by reduce_mod_char; decide) (by norm_num)

theorem prime_359 : Nat.Prime 359 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 179 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 179 1 _ prime_179 (by reduce_mod_char; decide) (by norm_num)

theorem prime_17 : Nat.Prime 17 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 4] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl
  · exact .prime 2 4 _ prime_2 (by reduce_mod_char; decide) (by norm_num)

theorem prime_19 : Nat.Prime 19 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 2] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)

theorem prime_53 : Nat.Prime 53 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 13 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 13 1 _ prime_13 (by reduce_mod_char; decide) (by norm_num)

theorem prime_68477 : Nat.Prime 68477 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 17 ^ 1, 19 ^ 1, 53 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 17 1 _ prime_17 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 19 1 _ prime_19 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 53 1 _ prime_53 (by reduce_mod_char; decide) (by norm_num)

theorem prime_958679 : Nat.Prime 958679 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 7 ^ 1, 68477 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 68477 1 _ prime_68477 (by reduce_mod_char; decide) (by norm_num)

theorem prime_4129989133 : Nat.Prime 4129989133 := by
  refine PrattCertificate'.out ⟨5, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 1, 359 ^ 1, 958679 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 359 1 _ prime_359 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 958679 1 _ prime_958679 (by reduce_mod_char; decide) (by norm_num)

theorem prime_71 : Nat.Prime 71 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 5 ^ 1, 7 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)

theorem prime_29 : Nat.Prime 29 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 7 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)

theorem prime_241 : Nat.Prime 241 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 4, 3 ^ 1, 5 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 4 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)

theorem prime_125803 : Nat.Prime 125803 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 2, 29 ^ 1, 241 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 29 1 _ prime_29 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 241 1 _ prime_241 (by reduce_mod_char; decide) (by norm_num)

theorem prime_2012849 : Nat.Prime 2012849 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 4, 125803 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 4 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 125803 1 _ prime_125803 (by reduce_mod_char; decide) (by norm_num)

theorem prime_4025699 : Nat.Prime 4025699 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 2012849 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 2012849 1 _ prime_2012849 (by reduce_mod_char; decide) (by norm_num)

theorem prime_80513981 : Nat.Prime 80513981 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 5 ^ 1, 4025699 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 4025699 1 _ prime_4025699 (by reduce_mod_char; decide) (by norm_num)

theorem prime_5247740253619 : Nat.Prime 5247740253619 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 3, 17 ^ 1, 71 ^ 1, 80513981 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 3 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 17 1 _ prime_17 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 71 1 _ prime_71 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 80513981 1 _ prime_80513981 (by reduce_mod_char; decide) (by norm_num)

theorem prime_1690502597179744445941507 : Nat.Prime 1690502597179744445941507 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 1, 13 ^ 1, 4129989133 ^ 1, 5247740253619 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 13 1 _ prime_13 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 4129989133 1 _ prime_4129989133 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5247740253619 1 _ prime_5247740253619 (by reduce_mod_char; decide) (by norm_num)

theorem prime_43 : Nat.Prime 43 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 1, 7 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)

theorem prime_173 : Nat.Prime 173 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 43 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 43 1 _ prime_43 (by reduce_mod_char; decide) (by norm_num)

theorem prime_294793 : Nat.Prime 294793 := by
  refine PrattCertificate'.out ⟨10, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 3, 3 ^ 1, 71 ^ 1, 173 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 3 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 71 1 _ prime_71 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 173 1 _ prime_173 (by reduce_mod_char; decide) (by norm_num)

theorem prime_59 : Nat.Prime 59 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 29 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 29 1 _ prime_29 (by reduce_mod_char; decide) (by norm_num)

theorem prime_827 : Nat.Prime 827 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 7 ^ 1, 59 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 59 1 _ prime_59 (by reduce_mod_char; decide) (by norm_num)

theorem prime_2557 : Nat.Prime 2557 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 2, 71 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 71 1 _ prime_71 (by reduce_mod_char; decide) (by norm_num)

theorem prime_4229279 : Nat.Prime 4229279 := by
  refine PrattCertificate'.out ⟨13, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 827 ^ 1, 2557 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 827 1 _ prime_827 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 2557 1 _ prime_2557 (by reduce_mod_char; decide) (by norm_num)

theorem prime_6091 : Nat.Prime 6091 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 1, 5 ^ 1, 7 ^ 1, 29 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 29 1 _ prime_29 (by reduce_mod_char; decide) (by norm_num)

theorem prime_5701177 : Nat.Prime 5701177 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 3, 3 ^ 2, 13 ^ 1, 6091 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 3 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 13 1 _ prime_13 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 6091 1 _ prime_6091 (by reduce_mod_char; decide) (by norm_num)

theorem prime_399082391 : Nat.Prime 399082391 := by
  refine PrattCertificate'.out ⟨7, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 5 ^ 1, 7 ^ 1, 5701177 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5 1 _ prime_5 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5701177 1 _ prime_5701177 (by reduce_mod_char; decide) (by norm_num)

theorem prime_757 : Nat.Prime 757 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 3, 7 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 3 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 7 1 _ prime_7 (by reduce_mod_char; decide) (by norm_num)

theorem prime_3037 : Nat.Prime 3037 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3 ^ 1, 11 ^ 1, 23 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 1 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 11 1 _ prime_11 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 23 1 _ prime_23 (by reduce_mod_char; decide) (by norm_num)

theorem prime_12149 : Nat.Prime 12149 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 3037 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3037 1 _ prime_3037 (by reduce_mod_char; decide) (by norm_num)

theorem prime_31649 : Nat.Prime 31649 := by
  refine PrattCertificate'.out ⟨3, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 5, 23 ^ 1, 43 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact .prime 2 5 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 23 1 _ prime_23 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 43 1 _ prime_43 (by reduce_mod_char; decide) (by norm_num)

theorem prime_5239247429827 : Nat.Prime 5239247429827 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 1, 3 ^ 2, 757 ^ 1, 12149 ^ 1, 31649 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl
  · exact .prime 2 1 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 757 1 _ prime_757 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 12149 1 _ prime_12149 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 31649 1 _ prime_31649 (by reduce_mod_char; decide) (by norm_num)

theorem prime_10427374428728808478656897599072717 : Nat.Prime 10427374428728808478656897599072717 := by
  refine PrattCertificate'.out ⟨2, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 2, 294793 ^ 1, 4229279 ^ 1, 399082391 ^ 1, 5239247429827 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl
  · exact .prime 2 2 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 294793 1 _ prime_294793 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 4229279 1 _ prime_4229279 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 399082391 1 _ prime_399082391 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 5239247429827 1 _ prime_5239247429827 (by reduce_mod_char; decide) (by norm_num)

theorem prime_28948022309329048855892746252171976963363056481941647379679742748393362948097 : Nat.Prime 28948022309329048855892746252171976963363056481941647379679742748393362948097 := by
  refine PrattCertificate'.out ⟨5, (by reduce_mod_char), ?_⟩
  refine .split [2 ^ 32, 3 ^ 2, 1709 ^ 1, 24859 ^ 1, 1690502597179744445941507 ^ 1, 10427374428728808478656897599072717 ^ 1] (fun r hr => ?_) (by norm_num)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
  · exact .prime 2 32 _ prime_2 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 3 2 _ prime_3 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 1709 1 _ prime_1709 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 24859 1 _ prime_24859 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 1690502597179744445941507 1 _ prime_1690502597179744445941507 (by reduce_mod_char; decide) (by norm_num)
  · exact .prime 10427374428728808478656897599072717 1 _ prime_10427374428728808478656897599072717 (by reduce_mod_char; decide) (by norm_num)

end Primality

def modulus : Nat := 28948022309329048855892746252171976963363056481941647379679742748393362948097
theorem modulus_prime : Nat.Prime modulus := Primality.prime_28948022309329048855892746252171976963363056481941647379679742748393362948097
instance : Fact modulus.Prime := ⟨modulus_prime⟩
abbrev F := ZMod modulus
theorem chunk_fits : 2 ^ 240 < modulus := by decide
theorem signed_coefficients_fit : 2 * 1554357600 < modulus := by decide
end CircuitCorrectness
