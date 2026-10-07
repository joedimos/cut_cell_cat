# Category theory through cells, patches, interfaces, and dynamics

Version 0.4 adds an executable mathematical layer. Its organizing example is a
partition of fine cells into coarse blocks. The same partition yields exact set
maps, an image/inverse-image adjunction, a saturation monad, a kernel pair,
volume-weighted restriction, and a chain map for mass balance. Fields on patches
supply a presheaf and a gluing problem. Interfaces compose by pushout.

This is a broad, concrete route through category theory, not an implementation
of every categorical framework. The distinctions in the coverage table are part
of the mathematical contract. In particular, numerical stability, categorical
laws, and Lean proof certificates are different kinds of evidence.

## Run the connected example

```sh
python -m pip install .
cut-cell-category
cut-cell-category --output category-report.json
python -m unittest discover -s tests -p 'test_categor*.py' -v
```

From a checkout use `python -m cutcell.category.showcase`. The command fails on
failed checks and reports an intentional counterexample: conservative
restriction generally does **not** commute with diffusion. In the four-cell
example, the Frobenius norm of that commutator is approximately 12.6491.

## Coverage and evidence

| Topic | Executable construction | Scope / evidence |
|---|---|---|
| Categories, duality | `FiniteCategory`, `opposite` | Every composable triple and identity checked exactly |
| Functors, natural transformations | `Functor`, `NaturalTransformation` | Endpoints, identity/composition preservation, every naturality square |
| 2-category structure | Vertical `then`, `horizontal` | Interchange and identity examples in tests |
| Representability, Yoneda | `representable`, `yoneda_lift`, `yoneda_evaluate` | Both inverse laws; all transformations enumerated for an S3 action |
| Initial / terminal objects | Empty and singleton finite sets | Unique maps, including empty edge cases |
| Limits / colimits | Products, coproducts, equalizers, coequalizers, pullbacks, pushouts | Concrete FinSet constructions; exhaustive small-instance uniqueness tests |
| Cartesian closed structure | `exponential`, `curry`, `uncurry` | Both inverse laws of the exponential adjunction, including empty sets |
| Cospans, monoidal composition | `Cospan.then`, `Cospan.tensor` | Interface gluing and disjoint union; associativity up to apex isomorphism |
| Adjunctions | `GaloisConnection`, `image_adjunction` | Every pair satisfies the adjunction equivalence in finite posets |
| Monads / comonads | `adj.monad`, `adj.comonad` | Idempotent closure/interior examples, not a general monad library |
| Kan extensions | `left_kan`, `right_kan` | Pointwise finite-poset formulas; universal inequalities tested exhaustively |
| Presheaves / sheaves | `section_presheaf`, `glue_sections` | Finite-valued fields on the discrete space of cells; overlap agreement and coverage |
| Chain complexes / homology | `closed_diffusion_diagram` | Conducting dual graph; H0 components, H1=0 for the bounded 1-D path |
| Chain maps / weighted adjoints | `Coarsening` | Exact integer incidence square; floating weighted identities; dynamics counterexample |
| General enriched, higher, derived, topos, operadic theories | Extension map below | Not implemented or certified |

The exact API is in `cutcell/category/`. Dense numerical audit matrices are in
`cutcell/categorical_numerics.py`; they do not replace the solver's efficient
stencil implementation. Category validation is cubic in the number of arrows.
Powersets, function sets, and universal-property enumeration are exponential.
Use these APIs for small mathematical examples, not full simulation-size sites.

## 1. Objects, arrows, and variance

`f.then(g)` means **g composed with f**. `FiniteCategory.composition[(f,g)]`
uses the same convention. A category constructor rejects missing composites,
extra noncomposable pairs, incorrectly typed identities, and nonassociative
multiplication. A finite monoid is allowed as a one-object category; it need not
be a poset or a group. The tests use the noncommutative group S3 so reversing
composition cannot accidentally pass as it would in an abelian example.

Finite-set objects use distinct, hashable labels with an explicit enumeration.
Different enumerations of a carrier are isomorphic but are different encoded
objects. Maps are total image tuples; equality is exact. Input dictionaries are
copied into read-only mappings. User-supplied labels must themselves be immutable.
The numerical layer uses floating-point arrays and is explicitly separate.

A functor F maps each arrow a:X→Y to F(a):F(X)→F(Y), preserves identities,
and satisfies F(gf)=F(g)F(f). A transformation α:F⇒G has components
α_X:F(X)→G(X), with G(f)α_X=α_Y F(f). Vertical composition is pointwise.
For β:H⇒K, horizontal composition has component β_(GX) H(α_X); naturality
makes this equal to K(α_X) β_(FX). `horizontal` and its tests use this formula.

### Yoneda, with an actual inverse

The implementation uses the **covariant** representable h_A=C(A,-). For a
finite-set-valued F:C→FinSet,

\[
\operatorname{Nat}(h_A,F)\cong F(A),\qquad
\alpha\mapsto\alpha_A(1_A),\qquad
x\mapsto[h\mapsto F(h)(x)].
\]

Proof: functoriality proves naturality of the displayed family. Evaluation of
its A component at 1_A gives x by preservation of identities. Conversely,
naturality of α at h:A→X gives α_X(h)=F(h)(α_A(1_A)), recovering every
component. This is a mathematical proof of the formula; exhaustive finite tests
check the code's realization. It is not a Lean proof.

For contravariant fields use the opposite category. The showcase applies Yoneda
to the patch restriction functor on P(cells)^op, so representable arrows out of a
full patch correspond to its restrictions. Variance is not silently switched.

## 2. Universal constructions must factor uniquely

`product(A,B)` returns A×B and two projections. `pair(f,g)` supplies the unique
mediator x↦(f(x),g(x)). Dually, `coproduct` tags elements by their summand,
preventing collisions between equal labels, and `copair` is the unique map
specified separately on the two summands.

For f:A→C and g:B→C, the pullback contains exactly pairs (a,b) satisfying
f(a)=g(b). A compatible cone u:X→A, v:X→B factors uniquely as x↦(u(x),v(x)).
`pullback_lift` rejects a noncommuting cone. For a fine-to-coarse partition p,
A×_B A is its **kernel pair**: pairs of cells in the same block.

The pushout of f:S→A and g:S→B is the tagged union A+B modulo the equivalence
relation generated by f(s)~g(s). Transitive closure is essential. A commuting
cocone u:A→X, v:B→X gives a map constant on each generating pair, hence on
each equivalence class. This defines a unique quotient map; `pushout_descend`
constructs it. A representative is chosen internally, but the result is
independent of that choice by the cocone equation.

Equalizers retain elements where parallel maps agree. Coequalizers quotient the
target by the generated equivalence relation. Tests enumerate **all** candidate
mediators for small cones/cocones and check exactly one factorization; checking
commutation alone would be insufficient to establish a universal property.

Finite sets are cartesian closed: B^A is the finite set of all functions A→B.
The evaluation map and `curry`/`uncurry` implement
Hom(X×A,B) ≅ Hom(X,B^A). The inverse equations follow by evaluation at every
(x,a), including the empty-function cases. These are finite constructions;
FinSet does not admit arbitrary infinite limits or colimits.

## 3. Interfaces and monoidal composition

A cospan A→N←B describes two boundary interfaces mapped into an apex. Sequential
composition glues the common interface by pushout; parallel composition uses
tagged disjoint union. This is useful for specifying how separate cell patches
are connected before attaching geometry or transport laws.

The implemented objects are **plain FinSet cospans**. Repeated pushouts produce
different nested quotient labels under different parenthesizations. Those
apices are canonically isomorphic by their universal property, not literally
equal. One obtains the usual category by taking isomorphism classes, or retains
apex maps and associators in a bicategorical treatment. The current API computes
representatives; it does not implement the full coherence machinery. Tests look
for boundary-preserving apex bijections rather than asserting tuple equality.

Plain interface gluing does not by itself glue conductances, preserve physical
volumes, or enforce a diffusion law. A future structured/decorated cospan layer
must specify the structure functor or decoration, its compatibility with
pushouts, and the numerical semantics before claiming compositional dynamics.

## 4. A partition produces an adjunction and a monad

For p:I→J, let L:P(I)→P(J) be direct image and R:P(J)→P(I) inverse image.
Then L(U)⊆V iff U⊆R(V): both sides say every element of U maps into V.
`image_adjunction` checks this equivalence for every pair of subsets.

The unit is U⊆RL(U), and the counit LR(V)⊆V. Applying these inequalities
and monotonicity in both directions gives RLR=R and LRL=L, the triangle
identities in a poset. The endofunctor T=RL is extensive and idempotent:
T²=T. It saturates a set of cells to all cells in every block it touches.
Its multiplication T²→T is the equality-induced identity arrow; associativity
and unit laws follow because parallel arrows in a poset are unique.
The corresponding comonad LR is an idempotent interior operator; if p is not
surjective it removes coarse labels outside the image.

Algebras for this closure monad are exactly fixed subsets T(U)=U: an algebra
arrow T(U)⊆U combines with extensivity to give equality. For the partition
(0,1)↦left, (2,3)↦right, the algebras are precisely unions of these two blocks.
This is a concrete monad example, not a claim that every monad is idempotent.
Direct image preserves unions but generally not intersections; the tests
include two different cells mapping to the same block as a counterexample.

## 5. Kan extensions organize missing levels

For monotone K:C→D and F:C→E, provided the indicated finite bounds exist,

\[
(\mathrm{Lan}_K F)(d)=\bigvee_{Kc\le d}F(c),\qquad
(\mathrm{Ran}_K F)(d)=\bigwedge_{d\le Kc}F(c).
\]

These constructions are monotone: the left indexing sets grow with d, and the
right indexing sets shrink. The join property gives Lan_K F≤H iff F≤HK:
use each term F(c)≤H(Kc)≤H(d) in one direction, and take d=Kc in the other.
Dually, H≤Ran_K F iff HK≤F. These are the universal properties in the thin
functor categories, verified for every monotone H in the small test cases.
An empty indexing set needs a bottom (left) or top (right); the API fails if
the needed bound does not exist. General Cat-valued Kan extensions, derived
functors, and coend formulas are outside this implementation.

## 6. Local data and sheaf gluing

Let I be the finite discrete space of cells and V a finite value set. Assign
S(U)=V^U to every patch U⊆I and restrict functions along inclusions. Identity
and composite restrictions are identical as functions, so S is a presheaf.
Compatible sections on a cover U=⋃U_i glue uniquely: define the value at a cell
using any covering patch; overlap agreement makes it well defined, and coverage
makes it total. Any alternative glue has the same value at each cell.

`section_presheaf` materializes this example; `glue_sections` implements the
union construction and rejects gaps or conflicts. The example concerns local
**data** on a discrete space. It does not assert that discrete diffusion
solutions form this sheaf, or that independently solved subdomains glue without
flux/interface constraints. It is not a general sheaf cohomology library.

## 7. The actual finite-volume operator as a diagram

For n cells and n+1 oriented face fluxes, define B_(i,i)=1,
B_(i,i+1)=-1. With M=diag(V_i), the implemented balance is

\[
M\dot c=BF+Ms,\qquad \mathbf1^TB=(1,0,\ldots,0,-1).
\]

This identity proves cancellation of internal exchanges and retains the two
external ports. It does **not** say total mass is constant under boundary
forcing or sources. Tests compare this factorization to the solver with
nonuniform cut volumes, Dirichlet data, prescribed flux, and source terms.

For closed, unforced diffusion restrict to wet cells and faces with positive
conductance. On this **dual graph**, nodes are wet cells and edges are their
conducting internal interfaces. With edge incidence B_int, diagonal positive
conductance K, and positive wet-cell mass M,

\[
F=-K B_{\rm int}^T c,\qquad L=-M^{-1}B_{\rm int}KB_{\rm int}^T.
\]

This is a two-term chain complex C1→C0 (zero differential below C0). Its dual
coboundary is B_int^T. There are no 2-cells here; claiming a nontrivial 2-D
boundary-of-boundary result would be unjustified. H0 has one generator per
conducting connected component. H1 is zero for this bounded path graph; zero
conductances split components. A component indicator z obeys z^T B_int=0,
so z^T M c is conserved by closed dynamics.

With inner product <x,y>_M=x^TMy, L is self-adjoint and negative semidefinite:

\[
\langle x,Ly\rangle_M=-(B_{\rm int}^Tx)^TK(B_{\rm int}^Ty),\qquad
\frac{d}{dt}\frac12c^TMc=-\|K^{1/2}B_{\rm int}^Tc\|^2\le0.
\]

These identities concern the semidiscrete system. The time discretization still
needs its documented stability bound. Weighted linear adjoints are not the
same notion as categorical adjoint functors. Dirichlet or nonzero flux boundary
data are intentionally rejected by `closed_diffusion_diagram` rather than
silently erased to make the closed-system identities hold.

## 8. Conservative coarse maps, and where naturality fails

For a contiguous partition, A sums fine masses into blocks and Q selects the
fine faces at block boundaries. Telescoping within each block gives the exact
integer identity

\[
AB_f=B_cQ.
\]

Thus (Q,A) is a chain map between the ported two-term complexes. Nested
partitions compose by matrix multiplication, as verified against direct
coarsening. Boundary ports mean this statement alone is not a closed-system
homology assertion.

Let V_c=AV_f. Define concentration restriction R=M_c^-1 A M_f and prolongation
P by assigning the block's value to each wet cell. Solid placeholders are set
to zero by P and are omitted from the actual weighted spaces. Every coarse
block must contain positive wet volume. Then

\[
RP=I,\quad M_cR=P^TM_f,\quad
\mathbf1^TM_cR=\mathbf1^TM_f,\quad (PR)^2=PR.
\]

The second equation makes R the weighted adjoint of P, and PR the orthogonal
projection onto block-constant wet-cell fields. This explains both conservation
and the information lost by averaging. It does not make P and R inverse maps
on arbitrary fine fields.

`dynamics_defect` takes full-grid generators. If a grid has solid cells,
embed the wet-only generator returned by `closed_diffusion_diagram` in a
zero matrix at its `wet_cells` row/column indices before passing it in.

Evolution commutes only under the **additional** condition RL_f=L_cR. The
showcase and tests supply ordinary uniform grids where this fails despite all
the preceding mass identities holding. Even a Galerkin choice L_c=RL_fP only
guarantees the projected formula; it need not close evolution for arbitrary
fine states. No adaptive multiscale solver is enabled by this diagnostic API.

## 9. Boundaries of the implemented theory

| Further topic | Coherent next construction | Required validity gate |
|---|---|---|
| General adjunctions and monads | Functor unit/counit and multiplication records | Naturality, both triangles, unit and associativity equations |
| Equivalences / localization | Explicit quasi-inverse and natural isomorphisms | Unit/counit isomorphism checks; do not equate a lossy restriction with equivalence |
| Enriched categories | Specify a monoidal base for weighted hom-objects | Enriched unit/composition and base-category laws |
| Ends, coends, profunctors | Finite coequalizer models and dinaturality | Universal factorization, variance, and composition coherence |
| Operads / wiring diagrams | Multi-input patch operations | Equivariance, units, substitution associativity; typed interfaces |
| Structured cospans / double categories | Geometry-bearing apices and interface maps | Structure functor, pushouts, interchange, associators and unitors |
| DPO graph rewriting | Rule span and pushout-complement construction | Gluing/dangling conditions and two actual pushout squares |
| Toposes / categorical logic | Subobject classifier and characteristic maps | Pullback classifier property; distinguish finite examples from Grothendieck toposes |
| Abelian / derived / homotopical theories | Higher-dimensional chain complexes and chain homotopies | d²=0, induced homology maps, quasi-isomorphisms; no higher-cell claims from a path |
| Higher categories | Coherent higher transformations | A specified model and its higher coherence laws |
| General sheaf cohomology | A nontrivial site, coefficient objects, resolutions | Descent and derived-functor prerequisites |
| Formal category proofs | A separately scoped Lean/mathlib development | Kernel-checked generic theorems and explicit implementation correspondence |

The legacy `TheoryRegistry` remains descriptive compatibility metadata. Petri
net invariants require w^T N=0 for the stoichiometric matrix N; token count is
not conserved in general. Graph rewrites do not generally preserve connectivity.
Neither registry text nor diagnostic similarity scores are category proofs.

## References

Definitions and further reading are aligned with these primary sources. The
finite-volume derivations above specialize the project's own operator and do
not claim these texts verify the code.

- Emily Riehl, [Category Theory in Context](https://emilyriehl.github.io/files/context.pdf):
  categories, Yoneda, universal constructions, adjunctions, monads, Kan extensions.
- Brendan Fong and David I. Spivak,
  [Seven Sketches in Compositionality](https://arxiv.org/abs/1803.05316):
  orders, compositional systems, and sheaf viewpoints.
- John C. Baez and Kenny Courser,
  [Structured Cospans](https://arxiv.org/abs/1911.04630): the additional structure
  needed beyond the plain finite-set cospans implemented here.
- [Numerical contract](NUMERICS.md) and
  [pinned Oceananigans source comparison](OCEANANIGANS_REFERENCE.md).
