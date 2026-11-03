# Categorical conservation on dynamic cut-cell meshes

This research track treats conservation as a verifiable interface contract. The
geometric pressure and solver experiments provide models to audit against that
contract. The algebra below is over integers for topology and over rationals
or reals for extensive quantities. Python uses exact integers and rationals
for small witnesses. Native floating-point runs have separate tolerances.

## Audit of the baseline

The starting tree is [`0715af7`](https://github.com/joedimos/cut_cell_cat/commit/0715af764e9632be89ade1131199a3afc0681327).

| Existing path | What it establishes | Gap addressed here |
| --- | --- | --- |
| [`categorical_numerics.py`](../cutcell/categorical_numerics.py) | Path incidence, contiguous weighted restriction, closed diffusion generator and its commutation defect | Arbitrary oriented dual graphs, ports, stagewise extensive budgets and component obstructions |
| [`category/order.py`](../cutcell/category/order.py) and [category notes](CATEGORY_THEORY.md) | Unique gluing for functions on a discrete cell set | Face restrictions compare outward traces in a common orientation on each interface atom |
| [`research/mitgcm.py`](../cutcell/research/mitgcm.py) | Static barotropic pressure, face cross sections and a concrete 2-D quotient | Wet/dry events and certificates on flux records instead of the pressure solver |
| [`subgrid_capacity.py`](../cutcell/research/subgrid_capacity.py) | Convex polygon capacity, frozen-conductance implicit solve and column budget | Distinguish endpoint capacity, swept geometry and overlap remap |
| [`lean_verification.py`](../lean_verification.py) | Certificates about specific budget scalars | Theorem statements with free variables and explicit hypotheses |

### Relevant established methods

- [Desbrun, Hirani, Leok & Marsden, DEC](https://arxiv.org/abs/math/0508341)
  separates incidence and coboundary from metric/dual data on simplicial
  complexes. [Arnold, Falk & Winther, FEEC](https://arxiv.org/abs/0906.4325)
  uses commuting projections with analytical hypotheses. [Di Pietro &
  Droniou, polyhedral DDR](https://doi.org/10.1007/s10208-021-09542-8)
  constructs arbitrary-order local spaces and discrete differential
  operators with topology-sensitive exactness. Our two-term flux balance
  uses only incidence. It supplies no DDR interpolation, discrete Hodge
  theory, approximation estimate or regularity statement.
- [Hansen & Ghrist, cellular sheaves](https://doi.org/10.1007/s41468-019-00038-7)
  identifies global sections with local data satisfying incidence
  compatibility. [Baez & Courser, structured cospans](https://arxiv.org/abs/1911.04630)
  and [Baez & Fong, open passive networks](https://www.tac.mta.ca/tac/volumes/33/38/33-38abs.html)
  formalize gluing through interfaces and forgetting the internal witness.
  Here the open-system semantics is a relation on signed fluxes with
  existentially quantified patch data. [Krishnan, flow-cut dualities
  for sheaves on graphs](https://people.math.osu.edu/krishnan.118/papers/mfmc.pdf)
  also studies sheaf-valued capacities and flows, relevant to any proposed
  extension from the unrestricted lift here to capacity-constrained repair.
- [Guillard & Farhat, discrete geometric conservation](https://doi.org/10.1016/S0045-7825(00)00173-0)
  addresses constant-state preservation under mesh motion. [Berger &
  Colella, AMR](https://doi.org/10.1016/0021-9991(89)90035-1) is a precedent for
  refluxing integrated face discrepancies. [Casulli](https://doi.org/10.1002/fld.1896),
  [Adcroft](https://doi.org/10.1016/j.ocemod.2013.03.002) and
  [Brus et al.](https://doi.org/10.1029/2026MS005842) already cover substantial
  parts of wet/dry volume, apertures and subgrid hypsometry. See the
  [prior-art gate](MITGCM_NOVELTY_GATE.md) for distinctions among the schemes.

## 1. Oriented finite-volume meshes and chain maps

A mesh object is an oriented finite *dual* multigraph. Its vertices are
control volumes, internal edges are faces with a tail and a head, and exterior
ports have exactly one incident cell. A face of area zero can remain a labeled
ambient face. Parallel faces retain their identities. For cell set \(C\) and
face set \(F\), take \(C_1=\mathbb Z^F\), \(C_0=\mathbb Z^C\) and the
differential \(B\) with column \(-e_{\rm tail}+e_{\rm head}\), omitting an
exterior endpoint. Set \(\epsilon_C=\mathbf1^T\).
This is a two-term chain complex with a distinguished total functional.
Exterior ports give \(\epsilon_C B\ne0\) in general, so this functional is
not an augmentation of the *chain complex* to an exact augmented sequence.

A morphism \(K\to L\) is an integral pair \((Q,A)\) with
\(A B_K=B_L Q\) and \(\epsilon_L A=\epsilon_K\).
Identity pairs and componentwise matrix composition satisfy both equations.
The latter follows by substitution,
\((A_2A_1)B_K=A_2B_LQ_1=B_M(Q_2Q_1)\). Thus there is a category and a
forgetful functor to two-term chain complexes. Dualization is
contravariant and gives \(B_K^T A^T=Q^T B_L^T\). General morphisms can have
negative coefficients. A partition is a special morphism that sums cell
masses and adds signed faces between each pair of coarse endpoints, sending
within-block faces to zero. The equality holds face by face, including
external ports. The total boundary ledger is natural because
\(\epsilon_LB_LQ=\epsilon_K B_K\). Parallel boundary segments may need
their labels preserved for interface verification even when the quotient
sums them for the balance.

The primal polytopal complex is a separate construction. For a finite
regular oriented polytopal mesh, the free group on its oriented \(k\)-cells
has the signed facet boundary \(\partial_k\). For a convex cell and each
codimension-two face, the two incident facets induce opposite signs there,
hence \(\partial_{k-1}\partial_k=0\). For example a counterclockwise
square has edge chain \(e_{01}+e_{12}+e_{23}+e_{30}\), whose vertex boundary
telescopes. Chains, cochains, Hodge weights, and dual-volume incidence are
different data. This branch implements the dual two-term category and
includes a finite primal-square witness in Lean.

## 2. Dynamic geometry and the geometric conservation law

On each time slab \(I=[t_n,t_{n+1}]\), keep ambient cell IDs and declare a set
\(F_I\) of permitted faces. Endpoint capacities \(v^0,v^1\ge0\) may
include zero. Positive endpoint capacity and slab face permission are
distinct. An event may activate a face incident to a cell with zero
initial volume. Active-face complexes map into the ambient complex by
the identity on cells and face inclusion. Two successive masks may each
contain a face missing from the other, so adjacent snapshots are related
through their ambient inclusions. They need no monotone-inclusion assumption.
A resolved mask change splits the time slab. A time-integrated flux over a
slab belongs to the allowed-face support for that slab.

For swept geometric face volume \(g\) and a prescribed geometric source
\(s_v\), the discrete GCL is \(v^1-v^0=B g+s_v\). For a uniform scalar
\(c\), the compatible integrated transport \(q=cg,\ s=cs_v\)
sends \(m^0=cv^0\) to \(m^1=cv^1\). Conversely, with those transport
terms, preservation of any nonzero uniform scalar implies the GCL.
This is an algebraic constant-state test. A physical scheme must specify
its swept volumes and source/transport conventions and how it measures
them. Geometry or mass crossing an exterior port appears explicitly.

When the physical domain has the same measure before and after a change
of partition, intersections supply \(O_{ji}\ge0\) with column sums
\(v_i^0\) and row sums \(v_j^1\). Set
\(T=O\operatorname{diag}(1/v^0)\) on wet columns and zero on dry columns.
The two margins prove total-mass conservation on supported inputs and
\(T v^0=v^1\), respectively. Nonnegativity proves positivity for
nonnegative mass. This construction requires zero mass on dry sources.
Domain creation/loss requires an additional boundary or reservoir ledger.
A sequence of remaps composes along a path by matrix multiplication.
Direct overlap between endpoints can disagree with composition because
projection to a piecewise constant field discards information. On two
equal cells, merge and split sends \((1,0)\) to \((1/2,1/2)\), while
direct identity leaves \((1,0)\) unchanged.

## 3. Resolution/evolution defects and feasible repair

Suppose a fine and a coarse evolution use the same time interval and
each satisfies \(m^1=m^0+Bq+s\), where \(q,s\) are integrated.
For a chain map and synchronized initial extensive data \(A m_f^0=m_c^0\),
subtract the two balances:

\[
D := A m_f^1-m_c^1 = B_c\underbrace{(Qq_f-q_c)}_{\delta_q}
                         + \underbrace{(As_f-s_c)}_{\delta_s}.
\]

If initial data differ, add \(A m_f^0-m_c^0\). If either budget has residual
\(r=m^1-m^0-Bq-s\), add \(A r_f-r_c\). Each term has units of the
conserved extensive quantity. A state naturality condition for a family of
evolutions is the equality on *all admissible inputs*, with forcing
synchronized and a choice of transfer at each endpoint. With time-varying
transfers \(A_0,A_1\), use the end transfer for the spatial chain identity.
Even with \(A_0m_f^0=m_c^0\), the right side then includes
\((A_1-A_0)m_f^0\). \(D=0\) can coexist with a nonzero cycle flux
or compensating sources. Likewise \(\epsilon_c D=0\) leaves room for
local state error. This formula extends the earlier generator check to
a finite update with face and source witnesses.

For a surjective restriction \(A\) and a fine update \(E_f\), a coarse
map with \(AE_f=E_c A\) exists iff \(AE_f\) is constant on every fiber
of \(A\). Applying \(E_c\) to equal restrictions gives necessity.
For sufficiency, define \(E_c(y)\) from any preimage of \(y\). For linear
\(E_f\) this is equivalently invariance of \(\ker A\). In the path
\(0\to1\to2\) grouped as \(\{0,1\}|\{2\}\), a diffusive interface
flux sees cell 1. Fine states \((2,0,0)\) and \((0,2,0)\) have the same
coarse mass but different flux at \(1\to2\), and give different coarse
next states with a positive stable timestep. This rules out any autonomous
coarse update on those masses for that fine dynamics.

**Open-component criterion.** Let \(B_I^{\rm int}\) use only allowed
internal faces. An unrestricted additive correction \(B_I^{\rm int}x=d\)
exists iff \(\sum_{i\in K}d_i=0\) for every open connected component
\(K\), including isolated cells. Necessity follows from summing the
incidence columns. For sufficiency root a spanning tree in each component.
Assign each tree-edge flux the sum of defects in its child subtree, with
sign positive when the face points into that subtree. At each nonroot
cell the signed incoming subtree sum minus the child sums equals its
defect. At the root the same identity uses the zero component total.
Set non-tree face values to zero. This proof works over \(\mathbb Z\)
as well as over a field. Equivalently the component totals represent the
class of \(d\) in \(\operatorname{coker}B_I^{\rm int}\).

A two-cell wet/dry example has fine mass transfer \(1\) on an open face
and coarse transfer \(0\) across a closed face. The shared mesh map is
conservative on the ambient graph, and \(D=(-1,1)\) sums to zero. Each
isolated coarse cell is its own component, so neither can repair the
defect by a legal internal exchange. With an open edge a correction of
one unit exists algebraically. If the donor has only half a unit above
its floor, it cannot pay this correction without another transfer.
Thus the graph criterion applies to unrestricted signed corrections.
Capacity, CFL and tracer positivity add separate inequalities.

## 4. Cellular flux sheaf and open-system semantics

Choose a common, finite space-time partition of each permitted face.
An atom is a labeled interval or spatial subface with an integrated
trace. For nonmatching grids or substeps, the caller must supply the
integrals on a common refinement. A total integral on a larger bin
cannot determine integrals on its sub-bins. Local histories must use the
same units, time support, interface geometry and orientation convention.

Use the incidence graph with cells as vertices and internal faces as
edges, adjoining one interface vertex outside the domain for each port
if an ordinary graph is desired. For a cell \(i\), its vertex stalk has
coordinates \((m_i^0,m_i^1,s_i,x_{i,f,a})\) in the linear subspace
\(m_i^1-m_i^0-s_i+\sum_{f,a}x_{i,f,a}=0\).
The \(x\)'s are outward integrals. The edge stalk for face \(f\) is
\(\mathbb Q^{\{a\}}\). The restriction from the tail projects to its
outward trace. The restriction from the head projects and changes sign.
An exterior vertex has a free trace stalk with the matching restriction.
Fixed mass and source data take an affine fiber of the section space.
Masks remove the closed-face coordinates, enforcing zero transfer.

A global section is a consistent assignment of vertex data whose two
restrictions agree on every edge. Given local records covering every
incident atom, this is equivalent to
\(x_{\rm tail,f,a}+x_{\rm head,f,a}=0\) on internal atoms.
A compatible pair determines a unique oriented atom flux, and summing
atoms gives the face ledger. Substituting into local balance yields
\(m^1-m^0=Bq+s\). If the local balances hold only over the whole slab,
this conclusion is also only a slab balance. Substep balance needs
substep mass and source data. Opposite atom residuals \((1,-1)\) are an
explicit counterexample to the converse based on face totals. A bad
interface can have zero total residual and all local full-slab balances.

For patches, project the local solution set onto boundary trace stalks
to obtain a relation. To compose along an interface, equate oriented
shared traces and existentially eliminate them. More formally the
ambient category is \(\mathbf{Rel}\) on rational trace spaces with
ordinary relational composition and Cartesian-product tensor. Input/output
polarity is encoded before composition by reversing outward signs on
inputs. A patch contributes an affine relation or the empty relation.
Sequential composition is associative because both bracketings assert
the same equalities and quantify the same internal traces. The identity
is equality in normalized orientation. Disjoint patches compose by
Cartesian product. This describes the boundary semantics of the finite
witnesses, without asserting a structured-cospan construction on all
geometric meshes or a well-posed open PDE solver.

## Evidence and scope

| Result | Evidence | Status |
| --- | --- | --- |
| Incidence, chain maps, flux gluing, GCL, overlap margins and relation composition | Proofs above and exact regressions | Established constructions and laws |
| State/face/source defect factorization and open-component lift | Proofs above, exhaustive small graph tests and finite Lean cases | Established algebra applied to the wet/dry interface problem |
| Closed-mask repair obstruction, trace aliasing and path-dependent overlap remap | Explicit counterexamples and exact regressions | Verified examples. Priority in the literature is undetermined |
| Common-atom transport witnesses across heterogeneous native solvers and restarts | Proposed instrumented experiment | Research hypothesis. Compare with flux registers, ALE/GCL checks and existing model diagnostics |
| Native #1048 comparison | [Reproduction and results](MITGCM_CATEGORICAL_BENCHMARK.md) | Scoped closed-shelf experiment |

The Lean file formalizes the factorization after projection to the coarse
ledger, one integral chain-map case, a primal square, a geometric scalar
identity, relational associativity, unique gluing on one atom, temporal aliasing, and the two-cell
repair criterion and obstruction. The general spanning-forest and sheaf
arguments are given above. There is no Lean refinement proof linking
Fortran, floating-point output or the Python implementation to those
theorems.

```sh
python -m unittest discover -s tests -v
lean verification/CategoricalConservation.lean
```

The upstream gate stays closed. A candidate contribution would need a
documented diagnostic gap under comparable native simulations, a measurable
benefit from retaining common-atom witnesses, and a closer prior-art
comparison. These results establish the verification question and supply
falsifiable cases.
