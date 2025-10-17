theorem succ_add_succ (a b : Nat) : Nat.succ a + Nat.succ b = Nat.succ (Nat.succ (a + b)) := by
  rw [Nat.succ_add]
  rfl

theorem succ_add_succ_2 (a b : Nat) : Nat.succ a + Nat.succ b = Nat.succ (Nat.succ (a + b)) := by
  sorry
