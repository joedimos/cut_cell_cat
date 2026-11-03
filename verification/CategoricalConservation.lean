import Std

namespace CategoricalConservation

abbrev Ledger (ι : Type) := ι → Int

/-- The extensive state square factors through the face and source defects. -/
theorem balance_defect {ι φ ψ : Type}
    (D : Ledger ψ → Ledger ι) (Q : Ledger φ → Ledger ψ)
    (fineFlux : Ledger φ) (coarseFlux : Ledger ψ)
    (fine₀ fine₁ coarse₀ coarse₁ fineSrc coarseSrc : Ledger ι)
    (hadd : ∀ q r i, D (fun f => q f - r f) i = D q i - D r i)
    (hf : ∀ i, fine₁ i = fine₀ i + D (Q fineFlux) i + fineSrc i)
    (hc : ∀ i, coarse₁ i = coarse₀ i + D coarseFlux i + coarseSrc i)
    (h₀ : fine₀ = coarse₀) :
    ∀ i, fine₁ i - coarse₁ i =
      D (fun f => Q fineFlux f - coarseFlux f) i + (fineSrc i - coarseSrc i) := by
  intro i
  rw [hadd, hf i, hc i, h₀]
  omega

/-- On a two-cell component a repair exists exactly when its defect sums to zero. -/
theorem two_cell_correction (a b : Int) :
    (∃ q : Int, -q = a ∧ q = b) ↔ a + b = 0 := by
  constructor
  · rintro ⟨q, h₁, h₂⟩
    omega
  · intro h
    exact ⟨b, by omega, rfl⟩

/-- A closed interface cannot repair a nonzero component defect. -/
theorem closed_cut_obstruction :
    ¬ ∃ q : Int, q = 0 ∧ -q = (1 : Int) ∧ q = -1 := by
  rintro ⟨q, h, h₁, _⟩
  omega

/-- With opposing outward traces a global oriented atom exists uniquely. -/
theorem glue_atom (left right : Int) :
    (∃ q : Int, (q = left ∧ -q = right) ∧
      ∀ q' : Int, (q' = left ∧ -q' = right) → q' = q) ↔ left + right = 0 := by
  constructor
  · rintro ⟨q, ⟨hl, hr⟩, _⟩
    omega
  · intro h
    refine ⟨left, ⟨rfl, by omega⟩, ?_⟩
    intro q hq
    exact hq.1

/-- Aggregated flux agreement can conceal two opposing atom mismatches. -/
theorem temporal_alias :
    ((1 : Int) + 0) + (0 + (-1)) = 0 ∧
      (1 : Int) + 0 ≠ 0 ∧ (0 : Int) + (-1) ≠ 0 := by decide

/-- A closed three-cell chain aggregated into {0,1}|{2}. -/
def aggregate (m : Int × Int × Int) : Int × Int := (m.1 + m.2.1, m.2.2)
def fineBoundary (q : Int × Int) : Int × Int × Int := (-q.1, q.1 - q.2, q.2)
def coarseBoundary (q : Int) : Int × Int := (-q, q)

theorem three_cell_chain_map (q : Int × Int) :
    aggregate (fineBoundary q) = coarseBoundary q.2 := by
  apply Prod.ext
  · change -q.1 + (q.1 - q.2) = -q.2
    omega
  · rfl

/-- A counterclockwise square with edges 01, 12, 23, 30. -/
def squareEdgeBoundary (q : Int × Int × Int × Int) : Int × Int × Int × Int :=
  (-q.1 + q.2.2.2, q.1 - q.2.1, q.2.1 - q.2.2.1, q.2.2.1 - q.2.2.2)

theorem square_boundary_cycle (s : Int) :
    squareEdgeBoundary (s, s, s, s) = (0, 0, 0, 0) := by
  simp [squareEdgeBoundary] <;> omega

/-- Boundary relations compose by shared interface witnesses. -/
def compose {α β γ : Type} (r : α → β → Prop) (s : β → γ → Prop) (a : α) (c : γ) :=
  ∃ b, r a b ∧ s b c

theorem compose_assoc {α β γ δ : Type}
    (r : α → β → Prop) (s : β → γ → Prop) (t : γ → δ → Prop) (a : α) (d : δ) :
    compose (compose r s) t a d ↔ compose r (compose s t) a d := by
  constructor
  · rintro ⟨c, ⟨b, hr, hs⟩, ht⟩
    exact ⟨b, hr, c, hs, ht⟩
  · rintro ⟨b, hr, c, hs, ht⟩
    exact ⟨c, ⟨b, hr, hs⟩, ht⟩

/-- Scaling a satisfied geometric balance preserves a uniform scalar. -/
theorem constant_scalar_two_cells (v₀ v₁ q c : Int) (h : v₁ = v₀ - q) :
    c * v₁ = c * v₀ - c * q := by
  rw [h, Int.mul_sub]

end CategoricalConservation
