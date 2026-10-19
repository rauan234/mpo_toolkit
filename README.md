# Symbolic MPO toolkit

`MPOTensor` stores exact SymPy matrices, composes physical legs, extracts exact
semisimple virtual blocks, and reconstructs a finite fusion category when its
required identities can be certified. The package includes a copy of the earlier
algebra-decomposition code; it does not depend on files elsewhere on your machine.

The conventions follow [arXiv:1511.08090v2](https://arxiv.org/html/1511.08090v2),
especially Sections 3.1, 3.2, 3.4 and the fusion/associator equations (17), (29),
and (33). There are explicit generalizations here: inverse retractions replace
Hermitian adjoints, and injectivity can hold after blocking. No unitarity,
braiding, pivotal/spherical structure, or MPO projector weights are inferred.

## Use in Jupyter

Keep the entire `mpo_toolkit` directory together and add its **parent** to the
Python path. In the delivered location:

```python
import sys
sys.path.insert(0, "/Users/rauan/Documents/Codex/2026-10-10/t/outputs")
import sympy as sp
from mpo_toolkit import MPOTensor, Limits, generate_fusion_category

X = sp.Matrix([[0, 1], [1, 0]])
Z = sp.diag(1, -1)
T = MPOTensor(2, [[sp.eye(2), X], [Z, sp.zeros(2)]])
assert T.test_injective()
print(T.injectivity_length)  # 2
```

Python 3.10+ and SymPy are required; tested with Python 3.13 and SymPy 1.14.0.
There are no other third-party dependencies. Use exact constants such as
`sp.Rational(1, 3)`, `sp.sqrt(2)`, and `sp.I`; floating-point entries, unresolved
parameters, and infinities are rejected. Substitute parameters before constructing
the tensor. Exact transcendental constants are accepted, but symbolic questions
about them may be undecidable by the available routines.

`dim_phys`, `n`, and `A_list` are read-only properties. The matrices and the
nested array are immutable copies. `name` and `limits` are optional constructor
keywords. By default numerical hints must be confirmed symbolically before they
can affect a returned certificate.

## Numerical hints and symbolic confirmation

The three global settings live in `mpo_toolkit.zero_testing`:

```python
from mpo_toolkit import zero_testing

zero_testing.DIFFERENCE_THRESHOLD = sp.Rational(1, 10**8)
zero_testing.EQUALITY_THRESHOLD = sp.Rational(1, 10**50)
zero_testing.REQUIRE_SYMBOLIC_CERTIFICATION = True

zero_testing.equal(sp.sqrt(2)**2, 2)  # True
zero_testing.is_zero(sp.sin(sp.pi))  # True
```

These are the defaults. Set attributes **on the module**; assigning a name imported
with `from ... import EQUALITY_THRESHOLD` does not modify the module's setting.
For the suggested looser equality threshold, assign `sp.Rational(1,10**10)`.

An uncertain scalar test receives a numerical indication from two precision
levels (100 and 160 decimal digits with the defaults). Evaluation precision grows
automatically for smaller equality thresholds. A magnitude strictly below the
equality threshold indicates probable zero; a magnitude strictly above the
difference threshold indicates probable nonzero. The intervening region, failed
evaluation, and disagreement between precision levels are inconclusive. These
are absolute thresholds: rescaling an expression can change its indication.

Cheap exact facts and canonical number-field reductions take priority. Thus
`is_zero(Rational(1,10**60))` correctly returns `False`, despite its tiny magnitude.
Otherwise numerical hints guide a symbolic check. In the default strict mode,
an unresolved test raises `RuntimeError`; numerical closeness alone never proves
equality. A conclusive symbolic answer overrides a conflicting numerical hint.

Setting `REQUIRE_SYMBOLIC_CERTIFICATION = False` permits an otherwise uncertain
test to use a stable numerical indication without symbolic confirmation. Results
then have `symbolically_certified=False`; this is a heuristic computation, not an
exact certificate of a decomposition or category. Inconclusive numerical tests
still try symbolic verification and raise `RuntimeError` if neither method decides.
The exact number-field backend continues using exact arithmetic in either mode.

MPO injectivity/decomposition caches are invalidated when any of these settings
changes. `T.injectivity_symbolically_certified`, `D.symbolically_certified`, and
`cat.symbolically_certified` distinguish the modes conservatively. Do not change
the globals concurrently with computations; category generation rejects a policy
change detected during the search.

## Composition order

Exactly as requested,

```python
C = T.compose(U)
# C[i,j] = sum_k kronecker_product(U[i,k], T[k,j])
```

The physical operator of `C` is `O_U O_T`; its virtual space is `V_U ⊗ V_T`.
For ordinary left-to-right multiplication use

```python
C = MPOTensor.product(T, U)  # physical O_T O_U; virtual V_T ⊗ V_U
```

The fusion-category coefficients use this latter order: `N[a][b][c]` describes
`MPOTensor.product(simples[a], simples[b])`. In terms of the requested method,
this is `simples[b].compose(simples[a])`.

`T.periodic_operator(length)` constructs a small periodic MPO explicitly using
the ordinary virtual trace. It is intended for small checks, and has an entry cap.

## Physical transpose (`transpose_physical`)

```python
T_transposed = T.transpose_physical()
```

This returns a new tensor with `T_transposed.A_list[i][j] = T.A_list[j][i]`.
Only the physical indices are exchanged: entries are not conjugated and virtual
matrices are not transposed. Dimensions and resource limits are preserved.
For every periodic length L, `T_transposed.periodic_operator(L)` equals
`T.periodic_operator(L).T`. Applying the method twice recovers the original data.

## Physical MPO adjoint (`conjugate`)

```python
T_dagger = T.conjugate()
```

As requested, `conjugate()` means the **physical MPO adjoint**, with local matrices
`T_dagger.A_list[i][j] = T.A_list[j][i].conjugate()`. It exchanges the physical
input/output indices and conjugates each virtual matrix entrywise, without
transposing its virtual indices. Using `A[j][i].H` would be a different operation.
For every periodic length L, `T_dagger.periodic_operator(L)` equals
`T.periodic_operator(L).H`. The method returns a new symbolic tensor with the same
physical dimension, virtual dimension, and resource limits. Applying it twice
recovers the original tensor data. It does not compute a categorical dual.

## Injectivity after blocking

For the local matrices `A[i,j]`, let `W_L` be the span of **words of exactly length
L**, not of lengths up to L. The routine returns `True` when `W_L = M_n(C)`.
It returns `False` if the algebra spanned by all nonempty words is proper, or if
an exact repeated fixed-length span certifies a cycle that never becomes full.
It raises `ResourceLimitError`, a `RuntimeError`, if the blocking cap is reached.
An undecidable symbolic test also raises a `RuntimeError`.

For example, the Pauli matrices `X,Z` generate `M_2(C)`, but their fixed-length
spans alternate between two proper subspaces. The routine correctly returns
`False`. Adding `I` makes the example injective at length two.

This is a conservative exact algorithm, not a claim to decide all questions
about arbitrary symbolic complex constants. Increase the cap with
`T.test_injective(max_blocking=256)` if needed. Certified answers are cached.

## Direct-sum decomposition and zero sectors

```python
D = T.direct_sum_decomp()  # zero sectors allowed by default
blocks, multiplicities, fusion_maps = D
blocking_lengths = [block.injectivity_length for block in blocks]

for a, block in enumerate(blocks):
    for mu in range(multiplicities[a]):
        F = fusion_maps[a][mu].inclusion  # also .X
        G = fusion_maps[a][mu].retraction # also .X_plus; generally not F.H

S = D.S
S_inverse = D.S_inverse
r = D.zero_dimension
F_zero, G_zero = D.zero_inclusion, D.zero_projection
```

The summands need only be **injective after blocking**; one-site injectivity is
not required. Each returned `block.injectivity_length` is a length L for which
products of exactly L of its local matrices span its full virtual matrix algebra.
For the example `T` above, decomposition returns a block with length two even
though its one-site span has dimension three, smaller than `dim(M_2)=4`.
The returned tensors retain the original physical dimension: the method certifies
blocking lengths without explicitly grouping physical sites or constructing the
larger blocked physical array.

The decomposition has

`T[i,j] = Σ_(a,mu) F[a,mu] blocks[a][i,j] G[a,mu]`

with `G[a,mu] F[b,nu] = δ_ab δ_mu,nu I`. Including the zero sector, the sum
of all inclusion–projection products is the full identity. In particular,

`S_inverse T[i,j] S = diag(blocks[a][i,j] ⊗ I_(multiplicities[a]), ..., 0_r)`.

Inside each summand, the basis order is `e_i ⊗ e_mu`, with `mu` varying fastest.
The zero-sector basis comes last. The zero sector is reported separately and
does not count as an injective tensor or as a simple object. An entirely zero
tensor returns empty block lists and its entire virtual space as the zero sector.
`allow_zero=False` rejects a nonzero zero-sector dimension.

Every returned local matrix is reconstructed and both intertwining relations
are verified exactly. Nonsemisimple algebras (including nonzero nilpotent or
triangular extensions) are rejected; contributions invisible under a periodic
trace are not silently deleted. `D.algebra_decomposition` exposes the underlying
algebra result, and `D.reconstruct(i,j)` is a convenient check.

**Meaning of block decomposition:** this method checks for an exact direct sum
of *individually* injective-after-blocking tensors. It does not require their
joint fixed-length span to be the entire direct sum of matrix algebras. These
conditions differ: the two scalar blocks `1` and `-1` are individually injective,
but remain jointly dependent at every fixed blocking length. Scalar/phase-related
blocks are kept distinct unless an actual simultaneous similarity identifies
them. This convention implements the requested direct-sum test, and should not
be confused with stronger uses of “block-injective” requiring joint independence.

Hermitian adjoints are generally insufficient in the original virtual gauge.
The code uses algebraic inclusion/retraction pairs throughout. A gauge making
a semisimple algebra adjoint-closed exists, but is not needed here.

## Exact local pull-through of a physical operator

```python
from mpo_toolkit import local_pull_through

ell = 2
# Supply h as a (T.dim_phys**ell)-by-(T.dim_phys**ell) SymPy matrix.
result = local_pull_through(T, h, ell)
if result.exists:
    h_tilde = result.h_tilde
    print("Pull-through operator:", h_tilde)
    print("Unique:", result.unique)
else:
    print("No exact local pull-through operator exists.")
```

This tests the **uncontracted local identity**. For physical multi-indices I,J,
write `B[I,J] = A[i1,j1] ... A[iell,jell]`. The requirement is

`sum_K B[I,K] h[K,J] = sum_K h_tilde[I,K] B[K,J]`

as an equality of virtual matrices for every I,J. The physical basis is ordered
lexicographically, with the last site varying fastest, consistently with
`sp.kronecker_product` and `periodic_operator`. The input is `d**ell` square;
`ell` is explicit to avoid ambiguity when `d=1`.

A successful identity gives a single h_tilde that works for all chain lengths
L >= ell, all allowed positions, and arbitrary boundary contractions. This is
stronger than asking only about contracted MPOs of length L > ell: the latter
can hide a nonzero local residual that is annihilated by surrounding matrices.
For instance, all longer MPOs made from a square-zero virtual matrix vanish,
but their local pull-through equations need not hold. This function checks the
local equations themselves, as requested.

No injectivity, semisimplicity, invertibility, or Hermiticity assumption is used.
In particular, h_tilde is not required to be Hermitian even if h is Hermitian.
The problem is linear in its entries. The code solves all its rows using one
shared coefficient matrix, incrementally retains independent equations, and
verifies a returned witness against every blocked virtual matrix.

In the default symbolic-certification mode:

* `result.exists=True` supplies an exact `result.h_tilde`.
* `result.exists=False` means the linear system is inconsistent; `h_tilde` is None.
* An undecidable symbolic calculation or an exhausted size limit raises
  `RuntimeError`, instead of returning a negative mathematical answer.

For nonunique solutions, the returned witness sets free variables to zero.
Each row of h_tilde can independently be shifted by a linear combination of
the transposes of the column vectors in `result.row_kernel`. An all-zero blocked
tensor accepts every h and returns the zero witness, with all row directions free.

The function honors `zero_testing.REQUIRE_SYMBOLIC_CERTIFICATION`; disabling it
marks `result.symbolically_certified=False`, and answers may then be heuristic.
Finite exact input constants are supported; floating-point entries and unresolved
free parameters in h are rejected, just as for MPOTensor data.

Two separate size caps can be adjusted:

```python
result = local_pull_through(
    T, h, ell,
    max_block_entries=1_000_000,  # d**(2*ell) * virtual_dimension**2
    max_linear_rows=100_000,     # d**ell * virtual_dimension**2
)
```

These are the defaults. Blocking grows exponentially with ell; the caps do not
bound the time spent within one symbolic arithmetic operation.

## An identical invertible physical gauge on every site

```python
from mpo_toolkit import tensor_power_intertwiner_exists

answer = tensor_power_intertwiner_exists(
    h, h_prime,
    dim_phys=T.dim_phys,
    ell=ell,
    progress=print,
)
```

This returns whether there is an invertible complex d-by-d matrix M with

`kronecker_product(M, ..., M) * h_prime = h * kronecker_product(M, ..., M)`

using ell copies of the **same** M. The physical operators must have shape
`(d**ell, d**ell)`, in the usual Kronecker-product basis order. M is not required
to be unitary. This is a different problem from allowing an arbitrary invertible
`d**ell`-by-`d**ell` intertwiner or independent matrices on different sites.

The algorithm introduces d**2 variables for the entries of M, plus a variable z.
It forms the polynomial equations above and adds `z*det(M)-1 = 0` to impose
invertibility. A Groebner basis detects whether these polynomials generate the
unit ideal. By the Nullstellensatz, the ideal contains 1 exactly when there is
no common complex zero. Thus this method handles positive-dimensional solution
sets and solutions whose coordinates lie outside the input number field.

The return value is a Boolean existence decision, not an explicit witness M.
Exact rational and algebraic complex entries (radicals, I, algebraic roots) are
supported. Float entries and free parameters are rejected. Other exact constants
can be handled immediately for identical inputs (M=I); otherwise an unavailable
certified coefficient field raises RuntimeError. This routine always uses exact
number-field/polynomial arithmetic, independently of the numerical zero-testing
option. A failed arithmetic computation is not reported as mathematical False.

Optional size caps are `max_variables=17` (requiring `d**2+1 <= 17`) and
`max_matrix_entries=10000` (requiring `d**(2*ell) <= 10000`). They can be increased:

```python
answer = tensor_power_intertwiner_exists(
    h, h_prime, dim_phys=5, ell=2,
    max_variables=26,
)
```

The identity shortcut can succeed without introducing polynomial variables.
Otherwise exceeding a cap raises ResourceLimitError. The caps do not bound the
time or intermediate memory of the Groebner calculation; even small systems can
take a long time. `progress=print` identifies the field-construction, equation,
and Groebner stages.

## Generate a fusion category

A simple example generates the two-object pointed category for Z/2:

```python
Z = sp.diag(1, -1)
# Virtual dimension one: the entries of the physical Z operator are 1x1 matrices.
z = MPOTensor(2, [[sp.Matrix([Z[i,j]]) for j in range(2)] for i in range(2)])
cat = generate_fusion_category([z], max_tensors=1000, progress=print)

print(cat.num_simples)
print(cat.N)                  # N[a][b][c]
print(cat.unit)               # index of the simple unit
print(cat.duals)              # duals[a]
print(cat.fusion(0, 0))       # {output_label: multiplicity}
print(cat.pentagon_verified)
```

Labels depend on the input order and algebraic decomposition choices. Do not
assume the unit is label zero. Duplicate gauge-equivalent inputs are merged.
`T.gauge_to(U)` returns an invertible `H` with `U[i,j] H = H T[i,j]`, or `None`
if inequivalence is certified; both inputs must be injective.

Closure uses exact simultaneous similarity of the **unblocked local matrices**.
It does not identify an overall scalar or phase, or tensors that become equivalent
only after blocking. Each discovered product summand is aligned to the chosen
representative, and its fusion maps are transformed accordingly.

The search processes all ordered pairs, adding new summands as needed. It then
checks fusion associativity, exact invertible associator matrices, all pentagons,
a unique simple unit, triangle identities, and both duality snake identities.
Failure of these conditions raises `RuntimeError`; finite closure alone is not
enough. For example, an identity and a distinct idempotent physical projector
generate a finite monoid, but not a rigid fusion category, and are rejected.

The unit must be present in the product closure; it is not silently adjoined.
It is the categorical unit on that closure and is not required to equal the
identity operator on the whole physical Hilbert space. A category on an invariant
physical support can have a projector as its unit MPO.

### Fusion maps and F-symbols

`cat.fusion_tensors[a,b,c][mu]` is an inclusion/retraction pair for
`V_c ↔ V_a ⊗ V_b`. `cat.pair_zero_sectors[a,b]` reports the zero-sector maps
of each pair product. Set `allow_zero=False` in the search to reject those.

`block = cat.F_symbols[a,b,c,d]` contains:

* `left_paths`: tuples `(e, mu, nu)`, with channels `a b → e` and `e c → d`;
* `right_paths`: tuples `(f, lam, sig)`, with channels `b c → f` and `a f → d`;
* `matrix`: rows indexed by right paths, columns indexed by left paths.

Writing `X` for inclusions, the exact convention is

`(X_ab^(e,mu) ⊗ I_c) X_ec^(d,nu)
 = Σ_(f,lam,sig) (I_a ⊗ X_bc^(f,lam)) X_af^(d,sig)
                 F[row(f,lam,sig), column(e,mu,nu)]`.

These are the coefficients in Eq. (29), with row/column placement specified
explicitly. Coefficients are extracted using the right-tree retractions and
verified by reconstructing both tree inclusions and retractions. Empty path
spaces have no dictionary entry. Fusion-basis choices affect the numerical
values of F-symbols, as expected.

`left_unitors[a]` and `right_unitors[a]` multiply the retraction channels
`unit ⊗ a → a` and `a ⊗ unit → a`, respectively. For `b = cat.duals[a]`,

```python
evaluation = cat.evaluations[a] * cat.fusion_tensors[b,a,cat.unit][0].X_plus
coevaluation = cat.coevaluations[a] * cat.fusion_tensors[a,b,cat.unit][0].X
```

Thus evaluation has type `V_b ⊗ V_a → V_unit`, and coevaluation has type
`V_unit → V_a ⊗ V_b`. These are one consistent choice of left-dual data.
`cat.verify_pentagon()` can rerun the exact pentagon check. Returned category
containers are ordinary Python data; modifying them invalidates the original
certificate. The test suite deliberately corrupts an F-symbol and verifies
that the pentagon check detects it.

## Limits and failures

```python
limits = Limits(max_blocking=256, max_virtual_dim=128,
                max_pentagon_paths=200000)
cat = generate_fusion_category([z], max_tensors=100, limits=limits)
```

Default limits are 128 blocking lengths, virtual dimension 256, 4096 gauge
unknowns, 10000 nonempty F blocks, 100000 initial pentagon paths, and 1000000
label quadruples. The algebra splitter uses 40 trials and at most eight field
extensions. Products and triple spaces are checked against the virtual-dimension
limit. All caps are adjustable through `Limits`, except `max_tensors`, which
is a search argument. `progress=print` reports closure and verification stages.

With symbolic confirmation enabled, `ResourceLimitError` means the computation could not decide within its budget;
it does **not** prove that the closure is infinite or a decomposition impossible.
`NotBlockDecomposableError` reports a certified obstruction to the requested
direct sum, and `NotFusionCategoryError` reports a category-axiom failure.
Other undecidable symbolic verification failures also inherit `RuntimeError`.
Invalid inputs raise `ValueError` or `TypeError`.

These limits bound counts and dimensions, not wall-clock time or memory within
a single SymPy operation. Large exact field extensions and decompositions can
still be expensive. The package is intended for modest symbolic examples, not
as a polynomial-time solution or a complete decision procedure.

## Tests and examples

From the parent of `mpo_toolkit`:

```sh
python3 -m unittest mpo_toolkit.test_mpo -v
python3 -m unittest mpo_toolkit.test_pull_through -v
python3 -m unittest mpo_toolkit.test_tensor_power_gauge -v
python3 -c 'from mpo_toolkit.semisimple_decomposition_GPT import run_self_tests; run_self_tests()'
python3 -m mpo_toolkit.examples
```

Tests cover composition order, fixed-length periodicity, symbolic constants,
nonunitary gauges, repeated blocks, oblique zero sectors, nilpotent rejection,
finite nonrigid closure, an infinite candidate stopped by its budget,
noncommutative pointed fusion, and nontrivial F-symbols for `Rep(S_3)` and
`Rep(A_4)`. The latter tests `3 ⊗ 3 = 1 + 1' + 1'' + 2·3` and a 7x7 F block,
including all pentagons. `test_results.txt` records the delivered test run.
