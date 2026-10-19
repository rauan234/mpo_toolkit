# WARNING: code by ChatGPT

"""Exact Wedderburn decomposition of a complex matrix algebra using SymPy.

Public entry point: decompose_algebra(generators, *, unital=True,
                                    substitutions=None, max_trials=40,
                                    seed=0, verify_products=True, backend='auto',
                                    max_extensions=8, progress=None).

Input entries must be finite, exact constants: rationals, algebraic numbers,
or other closed symbolic expressions. Substitute all free parameters first.
Floats are rejected, rather than interpreted as exact numbers. No numerical
rank or eigenvalue tests are used. Scalar numerical hints are configured through
mpo_toolkit.zero_testing and require symbolic confirmation by default. An
unresolved symbolic decision raises RuntimeError.

Algebraic constants use canonical number-field arithmetic, polynomial-factor
projectors, and direct restriction when physical multiplicity is one. For
repeated blocks a smaller commutant is used when feasible. Transcendental
constants retain the original expression backend. It returns an explicit algebra
isomorphism to a tuple of full matrix algebras and its inverse. It does NOT
return an ambient similarity transformation on the original physical space.

Requires Python >= 3.10 and SymPy >= 1.14. Run this file to execute its self-tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import isqrt
from random import Random
from typing import Iterable, Mapping, Sequence

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

_DOMAIN_ERRORS = (ValueError, TypeError, BasePolynomialError)


class SymbolicVerificationError(RuntimeError):
    """SymPy could not certify a necessary symbolic equality/inequality."""


class NonSemisimpleError(ValueError):
    """Dickson's test failed; attributes contain an exact radical certificate."""

    def __init__(self, basis, trace_matrix, radical_basis):
        self.algebra_basis = tuple(basis)
        self.trace_matrix = sp.ImmutableMatrix(trace_matrix)
        self.radical_basis = tuple(radical_basis)
        super().__init__(
            f"The generated algebra is not semisimple: dimension {len(basis)}, "
            f"radical dimension {len(radical_basis)}. "
            "See .algebra_basis, .trace_matrix, and .radical_basis."
        )


@lru_cache(maxsize=128)
def _root_field(roots):
    K = sp.QQ.algebraic_field(*roots)
    return K, dict((root, K.from_sympy(root)) for root in roots)


def _reduce_root_rational_function(x):
    """Reduce rational functions of exact roots without asking SymPy to infer
    a minimal polynomial of the whole quotient (which can be very expensive).
    Unsupported expression forms fall back to the ordinary symbolic backend.
    """
    roots = tuple(sorted(x.atoms(sp.CRootOf), key=sp.default_sort_key))
    if not roots:
        return x
    K, values = _root_field(roots)
    def evaluate(y):
        if y in values: return values[y]
        if y.is_Rational: return K.from_sympy(y)
        if y.is_Add:
            return sum((evaluate(z) for z in y.args), K.zero)
        if y.is_Mul:
            result = K.one
            for z in y.args: result *= evaluate(z)
            return result
        if y.is_Pow and y.exp.is_Integer:
            return evaluate(y.base) ** int(y.exp)
        raise ValueError('Not a rational function of the selected roots')
    return K.to_sympy(evaluate(x))


@lru_cache(maxsize=8192)
def _simp(x):
    x = sp.sympify(x)
    if x.is_Rational:
        return x
    if x.has(sp.CRootOf):
        try:
            return _reduce_root_rational_function(x)
        except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
            pass
    return sp.cancel(sp.simplify(x))


def _zero(x):
    """Configured scalar decision; default requires exact confirmation."""
    from .zero_testing import is_zero
    return is_zero(x, simplifier=_simp)


def _require_zero(x, label):
    if _zero(x) is not True:
        raise SymbolicVerificationError(f"Cannot certify {label}: {_simp(x)}")


def _require_nonzero(x, label):
    if _zero(x) is not False:
        raise SymbolicVerificationError(f"Cannot certify {label}: {_simp(x)}")


def _clean(M):
    return sp.ImmutableMatrix(M).applyfunc(_simp)


def _equal(A, B, label):
    A, B = sp.Matrix(A), sp.Matrix(B)
    if A.shape != B.shape:
        raise SymbolicVerificationError(f"Shape mismatch while checking {label}")
    for x in A - B:
        _require_zero(x, label)


def _vec(A):
    A = sp.Matrix(A)
    return A.reshape(A.rows * A.cols, 1)


def _unvec(v, n):
    return _clean(sp.Matrix(v).reshape(n, n))


def _linear_sum(coefficients, matrices, n):
    result = sp.zeros(n)
    for c, A in zip(coefficients, matrices):
        result += c * A
    return _clean(result)


def _rref(A):
    """Gaussian elimination with a strict three-state symbolic pivot test."""
    R = sp.MutableDenseMatrix(A).applyfunc(_simp)
    row, pivots = 0, []
    for col in range(R.cols):
        if row == R.rows:
            break
        pivot, uncertain = None, False
        for j in range(row, R.rows):
            flag = _zero(R[j, col])
            if flag is False:
                pivot = j
                break
            uncertain |= flag is None
        if pivot is None:
            if uncertain:
                raise SymbolicVerificationError(
                    f"Undecidable Gaussian-elimination pivot in column {col}"
                )
            continue
        R.row_swap(row, pivot)
        divisor = R[row, col]
        for j in range(R.cols):
            R[row, j] = _simp(R[row, j] / divisor)
        for i in range(R.rows):
            if i == row:
                continue
            c = R[i, col]
            if _zero(c) is True:
                continue
            for j in range(R.cols):
                R[i, j] = _simp(R[i, j] - c * R[row, j])
        pivots.append(col)
        row += 1
    return sp.ImmutableMatrix(R), tuple(pivots)


def _inverse(A, label):
    A = sp.Matrix(A)
    if A.rows != A.cols:
        raise SymbolicVerificationError(f"Nonsquare matrix: {label}")
    n = A.rows
    if n == 0:
        return sp.ImmutableMatrix(0, 0, [])
    R, pivots = _rref(A.row_join(sp.eye(n)))
    if pivots != tuple(range(n)):
        raise SymbolicVerificationError(f"Singular matrix: {label}")
    inv = _clean(R[:, n:])
    _equal(A * inv, sp.eye(n), f"right inverse of {label}")
    _equal(inv * A, sp.eye(n), f"left inverse of {label}")
    return inv


def _nullspace(A):
    A = sp.Matrix(A)
    R, pivots = _rref(A)
    columns = []
    for f in range(A.cols):
        if f in pivots:
            continue
        v = sp.zeros(A.cols, 1)
        v[f] = 1
        for row, p in enumerate(pivots):
            v[p] = -R[row, f]
        v = _clean(v)
        _equal(A * v, sp.zeros(A.rows, 1), "nullspace vector")
        columns.append(v)
    return tuple(columns)


class _Span:
    """Incremental exact independence test, preserving original matrices."""

    def __init__(self, n):
        self.n = n
        self.basis = []
        self.echelon = []
        self.pivots = []

    def add(self, A):
        A = _clean(A)
        v = _vec(A)
        for p, w in zip(self.pivots, self.echelon):
            v = (v - v[p] * w).applyfunc(_simp)
        pivot, uncertain = None, False
        for j, value in enumerate(v):
            flag = _zero(value)
            if flag is False:
                pivot = j
                break
            uncertain |= flag is None
        if pivot is None:
            if uncertain:
                raise SymbolicVerificationError("Undecidable linear dependence")
            return False
        self.pivots.append(pivot)
        self.echelon.append(_clean(v / v[pivot]))
        self.basis.append(A)
        return True


def _independent(matrices, n):
    span = _Span(n)
    for A in matrices:
        span.add(A)
    return tuple(span.basis)


class _Coordinates:
    def __init__(self, basis, n):
        self.basis, self.n = tuple(basis), n
        self.F = (sp.Matrix.hstack(*map(_vec, basis)) if basis
                  else sp.zeros(n * n, 0))
        if not basis:
            self.rows, self.inv = (), sp.ImmutableMatrix(0, 0, [])
            return
        _, self.rows = _rref(self.F.T)
        if len(self.rows) != len(basis):
            raise SymbolicVerificationError("Coordinate basis is dependent")
        self.inv = _inverse(self.F.extract(self.rows, range(len(basis))),
                            "coordinate submatrix")

    def __call__(self, A):
        v = _vec(A)
        if v.rows != self.n * self.n:
            raise ValueError("Wrong ambient matrix shape")
        selected = v.extract(self.rows, [0])
        c = _clean(self.inv * selected)
        _equal(self.F * c, v, "coordinate reconstruction / algebra membership")
        return c


def _poly_value(poly, A, identity):
    result = sp.zeros(A.rows)
    for c in poly.all_coeffs():
        result = _clean(result * A + c * identity)
    return result


def _exact_roots(poly):
    """Find a complete set of exact, simple roots, or fail explicitly."""
    try:
        roots = list(sp.roots(poly.as_expr(), poly.gen).keys())
    except (NotImplementedError, ValueError, sp.PolynomialError):
        roots = []
    if len(roots) != poly.degree():
        try:
            roots = poly.all_roots(radicals=False)
        except (NotImplementedError, ValueError, sp.PolynomialError) as exc:
            raise SymbolicVerificationError(
                "SymPy could not express all necessary exact polynomial roots"
            ) from exc
    if len(roots) != poly.degree():
        raise SymbolicVerificationError("Incomplete exact root list")
    roots = tuple(sorted(roots, key=sp.default_sort_key))
    for root in roots:
        _require_zero(poly.eval(root), "polynomial root")
    for i, root in enumerate(roots):
        for other in roots[:i]:
            _require_nonzero(root - other, "distinct polynomial roots")
    return roots


def _candidates(basis, rng, max_trials, n):
    # Basis elements first often have simpler spectra than random combinations.
    basis_trials = min(len(basis), max_trials // 2)
    for trial in range(max_trials):
        if trial < basis_trials:
            yield basis[trial]
        else:
            yield _linear_sum([sp.Integer(rng.randint(-5, 5))
                               for _ in basis], basis, n)


def _split_element(basis, coords, identity, degree, rng, max_trials):
    """Return b and a squarefree degree-'degree' polynomial annihilating b."""
    for b in _candidates(basis, rng, max_trials, identity.rows):
        L = sp.Matrix.hstack(*(coords(b * c) for c in basis))
        try:
            characteristic = L.charpoly()
            # SymPy may rename a requested charpoly generator. Use the actual
            # returned generator instead of relying on its requested name.
            q = sp.Poly(characteristic.as_expr(), characteristic.gen).sqf_part().monic()
        except (NotImplementedError, ValueError, sp.PolynomialError) as exc:
            raise SymbolicVerificationError("Exact characteristic polynomial failed") from exc
        if q.degree() != degree:
            continue
        # Characteristic-polynomial computation alone is not the certificate.
        _equal(_poly_value(q, b, identity), sp.zeros(identity.rows),
               "squarefree annihilating polynomial")
        return b, q
    raise SymbolicVerificationError(
        f"No suitable splitting element after {max_trials} exact trials; "
        "increase max_trials or change seed."
    )


@dataclass(frozen=True)
class SimpleBlock:
    size: int
    multiplicity: int
    central_idempotent: sp.ImmutableMatrix
    primitive_idempotent: sp.ImmutableMatrix
    ideal_basis: tuple
    left_ideal_basis: tuple
    algebra_basis_images: tuple
    generator_images: tuple
    method: str = "minimal_left_ideal"
    representation_basis: sp.ImmutableMatrix | None = None


@dataclass(frozen=True)
class AlgebraDecomposition:
    algebra_basis: tuple
    identity: sp.ImmutableMatrix
    center_basis: tuple
    dickson_matrix: sp.ImmutableMatrix
    blocks: tuple[SimpleBlock, ...]
    generators: tuple
    unital: bool
    _coordinates: _Coordinates = field(repr=False, compare=False)
    _inverse_map: sp.ImmutableMatrix = field(repr=False, compare=False)
    coefficient_field: object = None
    backend: str = "expressions"
    extensions: tuple = ()
    _field_maps: object = field(default=None, repr=False, compare=False)

    @property
    def dimension(self):
        return len(self.algebra_basis)

    @property
    def block_sizes(self):
        return tuple(block.size for block in self.blocks)

    @property
    def generator_images(self):
        """generator_images[i][a] is the image of generator i in block a."""
        return tuple(tuple(block.generator_images[i] for block in self.blocks)
                     for i in range(len(self.generators)))

    def to_blocks(self, X):
        """Apply the isomorphism; certify that X lies in the original algebra."""
        if self._field_maps is not None:
            value = self._field_maps.to_blocks(X)
            if value is not None:
                return value
        c = self._coordinates(sp.Matrix(X))
        return tuple(_linear_sum(c, block.algebra_basis_images, block.size)
                     for block in self.blocks)

    def from_blocks(self, images: Sequence[sp.MatrixBase]):
        """Inverse isomorphism: reconstruct an original n-by-n matrix."""
        if self._field_maps is not None:
            value = self._field_maps.from_blocks(images)
            if value is not None:
                return value
        if len(images) != len(self.blocks):
            raise ValueError("Expected one matrix for each simple block")
        pieces = []
        for image, block in zip(images, self.blocks):
            image = sp.Matrix(image)
            if image.shape != (block.size, block.size):
                raise ValueError(f"Expected a {block.size}-by-{block.size} block")
            pieces.append(_vec(image))
        v = sp.Matrix.vstack(*pieces) if pieces else sp.zeros(0, 1)
        c = self._inverse_map * v
        return _linear_sum(c, self.algebra_basis, self.identity.rows)


def _prepare_generators(generators, substitutions):
    if not generators:
        raise ValueError("Supply at least one square matrix (possibly zero)")
    matrices = tuple(_clean(sp.Matrix(A).subs(substitutions or {}))
                     for A in generators)
    n = matrices[0].rows
    if n < 1 or any(A.shape != (n, n) for A in matrices):
        raise ValueError("All generators must have the same positive square shape")
    for A in matrices:
        for entry in A:
            if entry.atoms(sp.Float):
                raise ValueError("Floating-point entries are not exact; use Rational")
            if entry.free_symbols:
                raise ValueError(
                    "Unresolved parameters can change the algebra. Supply exact "
                    f"substitutions for {sorted(entry.free_symbols, key=str)}."
                )
            if entry.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                raise ValueError("Entries must be finite complex constants")
    return matrices, n


def decompose_algebra(
    generators: Iterable[sp.MatrixBase], *, unital: bool = True,
    substitutions: Mapping | None = None, max_trials: int = 40,
    seed: int = 0, verify_products: bool = True,
    backend: str = "auto", max_extensions: int = 8,
    commutant_max_variables: int = 1024, progress=None,
) -> AlgebraDecomposition:
    """Compute and symbolically verify an exact complex Wedderburn decomposition.

    unital=True includes I_n in the generated algebra. With unital=False only
    nonempty words in the generators are included. The zero algebra is supported.

    Raises NonSemisimpleError with .radical_basis if Dickson's test fails.
    Raises SymbolicVerificationError on undecidable exact tests, unavailable
    exact roots, or exhaustion of the bounded search for generic elements.

    verify_products=False skips ONLY the redundant final multiplication
    certificate. The algebraic backend checks every basis element times every
    independent generator; linearity and induction on words certify all products.
    Exact left-action, idempotent, membership, and bijectivity certificates
    are always checked, and already imply that the returned map is an isomorphism.
    For a certified full ambient matrix algebra, the algebraic backend returns
    the identity isomorphism in the standard matrix-unit basis. Multiplication
    preservation then holds identically, so no pairwise product checks are needed.

    backend='auto' uses canonical number-field arithmetic when all entries are
    algebraic, otherwise the expression backend. backend='algebraic' requires
    algebraic entries; backend='expressions' forces the original implementation.
    progress, if supplied, is a callback receiving descriptive stage strings.
    max_extensions bounds field enlargements. commutant_max_variables bounds
    the number of unknowns in a dense commutant linear system; 0 disables it.
    """
    if not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer")
    generators, n = _prepare_generators(tuple(generators), substitutions)
    if backend not in ("auto", "algebraic", "expressions"):
        raise ValueError("backend must be 'auto', 'algebraic', or 'expressions'")
    if not isinstance(max_extensions, int) or max_extensions < 0:
        raise ValueError("max_extensions must be a nonnegative integer")
    if not isinstance(commutant_max_variables, int) or commutant_max_variables < 0:
        raise ValueError("commutant_max_variables must be a nonnegative integer")
    algebraic = all(x.is_algebraic is True for A in generators for x in A)
    if backend == "algebraic" and not algebraic:
        raise ValueError("The algebraic backend requires algebraic constants")
    if backend != "expressions" and algebraic:
        return _decompose_number_field(
            generators, n, unital, max_trials, seed, verify_products,
            max_extensions, commutant_max_variables, progress,
        )
    if progress:
        progress("Using expression arithmetic for nonalgebraic entries")
    rng = Random(seed)

    span = _Span(n)
    if unital:
        span.add(sp.eye(n))
    for A in generators:
        span.add(A)
    cursor = 0
    while cursor < len(span.basis):
        B = span.basis[cursor]
        for A in generators:
            span.add(B * A)
        cursor += 1
    basis = tuple(span.basis)
    m = len(basis)
    coords = _Coordinates(basis, n)

    # Certify generator membership and closure under every basis product.
    for A in generators:
        coords(A)
    if verify_products:
        for B in basis:
            for C in basis:
                coords(B * C)

    G = _clean(sp.Matrix(m, m, lambda i, j: sp.trace(basis[i] * basis[j])))
    radical_coordinates = _nullspace(G)
    if radical_coordinates:
        radical = tuple(_linear_sum(v, basis, n) for v in radical_coordinates)
        raise NonSemisimpleError(basis, G, radical)

    if m == 0:
        return AlgebraDecomposition(
            (), sp.ImmutableMatrix(sp.zeros(n)), (), G, (), generators, unital,
            coords, sp.ImmutableMatrix(0, 0, []),
        )

    # Compute the center by commuting with the original generators.
    commutators = sp.Matrix.vstack(*(
        sp.Matrix.hstack(*(_vec(B * A - A * B) for B in basis))
        for A in generators
    ))
    center = tuple(_linear_sum(v, basis, n) for v in _nullspace(commutators))
    center_coords = _Coordinates(center, n)
    s = len(center)
    if s == 0:
        raise SymbolicVerificationError("A nonzero semisimple algebra needs a center")

    if unital:
        E = sp.ImmutableMatrix(sp.eye(n))
    else:
        # Its internal identity is the unique central element acting as 1.
        equations = sp.Matrix.vstack(*(
            sp.Matrix.hstack(*(_vec(Z * B) for Z in center)) for B in basis
        ))
        rhs = sp.Matrix.vstack(*(_vec(B) for B in basis))
        R, pivots = _rref(equations.row_join(rhs))
        if pivots != tuple(range(s)):
            raise SymbolicVerificationError("Cannot solve for the internal identity")
        identity_coeffs = R[:s, s]
        E = _linear_sum(identity_coeffs, center, n)
        _equal(equations * identity_coeffs, rhs, "internal identity equations")
    coords(E)
    for B in basis:
        _equal(E * B, B, "left identity")
        _equal(B * E, B, "right identity")

    z, center_poly = _split_element(center, center_coords, E, s, rng, max_trials)
    lambdas = _exact_roots(center_poly)
    central_idempotents = []
    for a, lam in enumerate(lambdas):
        p = E
        for b, other in enumerate(lambdas):
            if a != b:
                p = _clean(p * (z - other * E) / (lam - other))
        coords(p)
        _equal(p * p, p, "central idempotent")
        for B in basis:
            _equal(p * B, B * p, "centrality")
        for previous in central_idempotents:
            _equal(p * previous, sp.zeros(n), "orthogonal central idempotents")
        central_idempotents.append(p)
    _equal(_linear_sum([1] * s, central_idempotents, n), E,
           "sum of central idempotents")

    blocks = []
    for p in central_idempotents:
        ideal = _independent((p * B for B in basis), n)
        d = isqrt(len(ideal))
        if d * d != len(ideal) or d == 0:
            raise SymbolicVerificationError("Simple ideal dimension is not a positive square")
        ideal_coords = _Coordinates(ideal, n)
        b, q = _split_element(ideal, ideal_coords, p, d, rng, max_trials)
        # Only one eigenvalue is needed for a minimal left ideal. Prefer a
        # CRootOf for rational irreducible polynomials of degree > 2; this avoids
        # expensive cubic/quartic radical formulas.
        try:
            if q.degree() > 2 and q.domain.is_QQ:
                mu = q.all_roots(radicals=False)[0]
            else:
                mu = _exact_roots(q)[0]
        except (NotImplementedError, ValueError, sp.PolynomialError) as exc:
            raise SymbolicVerificationError("Cannot obtain an exact block eigenvalue") from exc
        _require_zero(q.eval(mu), "block eigenvalue")
        denominator = _simp(q.diff().eval(mu))
        _require_nonzero(denominator, "spectral projector denominator")
        # Synthetic division: h(t)=q(t)/(t-mu). Then e=h(b)/q'(mu).
        coeffs = q.all_coeffs()
        hcoeffs = [coeffs[0]]
        for c in coeffs[1:-1]:
            hcoeffs.append(_simp(c + mu * hcoeffs[-1]))
        _require_zero(coeffs[-1] + mu * hcoeffs[-1], "synthetic division remainder")
        h = sp.Poly.from_list(hcoeffs, q.gen)
        e = _clean(_poly_value(h, b, p) / denominator)
        ideal_coords(e)
        _equal(e * e, e, "primitive spectral idempotent")
        left_ideal = _independent((C * e for C in ideal), n)
        if len(left_ideal) != d:
            raise SymbolicVerificationError("Spectral idempotent is not primitive")
        left_coords = _Coordinates(left_ideal, n)

        images = tuple(_clean(sp.Matrix.hstack(*(
            left_coords(B * v) for v in left_ideal
        ))) for B in basis)
        # Every reconstructed action above is checked entry by entry.
        identity_image = _linear_sum(coords(p), images, d)
        _equal(identity_image, sp.eye(d), "block identity representation")
        generator_images = tuple(_linear_sum(coords(A), images, d)
                                 for A in generators)
        physical_rank = _simp(sp.trace(p))  # exact rank of a verified idempotent
        if physical_rank.is_Integer is not True:
            raise SymbolicVerificationError("Cannot certify physical block rank")
        physical_rank = int(physical_rank)
        if physical_rank <= 0 or physical_rank % d:
            raise SymbolicVerificationError("Invalid physical block multiplicity")
        blocks.append(SimpleBlock(
            d, physical_rank // d, p, e, ideal, left_ideal,
            images, generator_images,
        ))

    if sum(block.size ** 2 for block in blocks) != m:
        raise SymbolicVerificationError("Block dimensions do not sum to algebra dimension")
    # Matrix of the complete linear isomorphism in row-major coordinates.
    forward = sp.Matrix.vstack(*(
        sp.Matrix.hstack(*(_vec(image) for image in block.algebra_basis_images))
        for block in blocks
    ))
    inverse = _inverse(forward, "complete Wedderburn map")
    result = AlgebraDecomposition(
        basis, E, center, G, tuple(blocks), generators, unital, coords, inverse,
    )

    # Final exact certificates; no numerical conclusions are accepted.
    for i, B in enumerate(basis):
        _equal(result.from_blocks(result.to_blocks(B)), B, "isomorphism round trip")
        if verify_products:
            for j, C in enumerate(basis):
                product_images = result.to_blocks(B * C)
                for block, image in zip(blocks, product_images):
                    _equal(image, block.algebra_basis_images[i] *
                           block.algebra_basis_images[j], "multiplication preservation")
    for a, block in enumerate(blocks):
        expected = tuple(sp.eye(other.size) if a == b else sp.zeros(other.size)
                         for b, other in enumerate(blocks))
        for actual, wanted in zip(result.to_blocks(block.central_idempotent), expected):
            _equal(actual, wanted, "central idempotent block image")
    return result


# ---------------------------------------------------------------------------
# Canonical number-field backend. All matrices below are tuples of rows of
# native QQ/AlgebraicField elements. No expression simplification occurs in
# Gaussian elimination, coordinate solves, or certificate checks.
# ---------------------------------------------------------------------------


class _NotMember(ValueError):
    pass


class _ExtendField(Exception):
    def __init__(self, polynomial, reason):
        self.polynomial, self.reason = polynomial, reason


class _FK:
    def __init__(self, K):
        self.K, self.zero, self.one = K, K.zero, K.one
        self._conversions = {}

    def element(self, expression):
        expression = sp.sympify(expression)
        if expression not in self._conversions:
            self._conversions[expression] = self.K.from_sympy(expression)
        return self._conversions[expression]

    def matrix(self, A):
        A = sp.Matrix(A)
        return tuple(tuple(self.element(A[i, j]) for j in range(A.cols))
                     for i in range(A.rows))

    def output(self, A):
        return sp.ImmutableMatrix([[self.K.to_sympy(x) for x in row] for row in A])


def _fz(rows, cols, F):
    return tuple(tuple(F.zero for _ in range(cols)) for _ in range(rows))


def _fi(n, F):
    return tuple(tuple(F.one if i == j else F.zero for j in range(n))
                 for i in range(n))


def _ft(A):
    return tuple(zip(*A))


def _fflat(A):
    return tuple(x for row in A for x in row)


def _freshape(v, rows, cols):
    return tuple(tuple(v[i * cols:(i + 1) * cols]) for i in range(rows))


def _fadd(A, B):
    return tuple(tuple(a + b for a, b in zip(ar, br)) for ar, br in zip(A, B))


def _fsub(A, B):
    return tuple(tuple(a - b for a, b in zip(ar, br)) for ar, br in zip(A, B))


def _fscale(A, c):
    return tuple(tuple(c * x for x in row) for row in A)


def _fmul(A, B, F):
    nr, inner, nc = len(A), len(B), len(B[0])
    if len(A[0]) != inner:
        raise SymbolicVerificationError("Field matrix multiplication shape mismatch")
    out = [[F.zero for _ in range(nc)] for _ in range(nr)]
    sparse_B = tuple(tuple((j, b) for j, b in enumerate(row) if b != F.zero)
                     for row in B)
    for i, row in enumerate(A):
        for k, a in enumerate(row):
            if a == F.zero:
                continue
            for j, b in sparse_B[k]:
                out[i][j] += a * b
    return tuple(tuple(row) for row in out)


def _flin(coefficients, matrices, F):
    if not matrices:
        raise SymbolicVerificationError("Empty internal linear combination")
    result = [[F.zero] * len(matrices[0][0]) for _ in matrices[0]]
    for c, A in zip(coefficients, matrices):
        if c != F.zero:
            for i, row in enumerate(A):
                for j, x in enumerate(row):
                    if x != F.zero:
                        result[i][j] += c * x
    return tuple(tuple(row) for row in result)


class _FLinearCombination:
    """Precompute nonzero entries for repeated combinations of a fixed basis."""

    def __init__(self, matrices, F):
        self.F = F
        self.rows, self.cols = len(matrices[0]), len(matrices[0][0])
        self.terms = tuple(tuple((i, j, x) for i, row in enumerate(A)
                                 for j, x in enumerate(row) if x != F.zero)
                           for A in matrices)

    def __call__(self, coefficients):
        result = [[self.F.zero] * self.cols for _ in range(self.rows)]
        for c, terms in zip(coefficients, self.terms):
            if c != self.F.zero:
                for i, j, x in terms:
                    result[i][j] += c * x
        return tuple(tuple(row) for row in result)


def _ftrace(A, F):
    return sum((A[i][i] for i in range(len(A))), F.zero)


def _ftrace_product(A, B, F):
    # Computing the full matrix product just to take its trace costs an extra
    # factor of n. Use Tr(AB)=sum_{ij} A_ij B_ji directly.
    return sum((a * B[j][i] for i, row in enumerate(A) for j, a in enumerate(row)
                if a != F.zero and B[j][i] != F.zero), F.zero)


def _fequal(A, B, label):
    if A != B:
        raise SymbolicVerificationError(f"Exact field certificate failed: {label}")


def _frref(A, F):
    R = [list(row) for row in A]
    nr, nc = len(R), len(R[0]) if R else 0
    r, pivots = 0, []
    for c in range(nc):
        if r == nr:
            break
        p = next((i for i in range(r, nr) if R[i][c] != F.zero), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        inverse = F.one / R[r][c]
        R[r] = [x * inverse if x != F.zero else x for x in R[r]]
        pivot_entries = tuple((j, x) for j, x in enumerate(R[r]) if x != F.zero)
        for i in range(nr):
            if i != r and R[i][c] != F.zero:
                multiplier = R[i][c]
                for j, y in pivot_entries:
                    R[i][j] -= multiplier * y
        pivots.append(c)
        r += 1
    return tuple(tuple(row) for row in R), tuple(pivots)


def _fsolve(A, B, F, label):
    """Unique exact solution to a possibly overdetermined full-column-rank system."""
    columns = len(A[0])
    augmented = tuple(a + b for a, b in zip(A, B))
    R, pivots = _frref(augmented, F)
    if pivots != tuple(range(columns)):
        raise SymbolicVerificationError(f"Nonunique or inconsistent field solve: {label}")
    X = tuple(row[columns:] for row in R[:columns])
    _fequal(_fmul(A, X, F), B, label)
    return X


def _finverse(A, F, label):
    if len(A) != len(A[0]):
        raise SymbolicVerificationError(f"Nonsquare field inverse: {label}")
    identity = _fi(len(A), F)
    inverse = _fsolve(A, identity, F, label)
    _fequal(_fmul(inverse, A, F), identity, label)
    return inverse


def _fnullspace(A, F):
    R, pivots = _frref(A, F)
    columns = len(A[0])
    vectors = []
    for c in range(columns):
        if c in pivots:
            continue
        v = [F.zero for _ in range(columns)]
        v[c] = F.one
        for r, p in enumerate(pivots):
            v[p] = -R[r][c]
        column = tuple((x,) for x in v)
        _fequal(_fmul(A, column, F), _fz(len(A), 1, F), "field nullspace")
        vectors.append(tuple(v))
    return tuple(vectors)


class _FSpan:
    def __init__(self, F):
        self.F, self.basis, self.echelon, self.pivots = F, [], [], []
        self._sparse_echelon = []

    def add(self, A):
        v = list(_fflat(A))
        for p, row in zip(self.pivots, self._sparse_echelon):
            if v[p] != self.F.zero:
                multiplier = v[p]
                for j, y in row:
                    v[j] -= multiplier * y
        pivot = next((i for i, x in enumerate(v) if x != self.F.zero), None)
        if pivot is None:
            return False
        inverse = self.F.one / v[pivot]
        self.pivots.append(pivot)
        row = tuple(x * inverse if x != self.F.zero else x for x in v)
        self.echelon.append(row)
        self._sparse_echelon.append(tuple((j, x) for j, x in enumerate(row)
                                         if x != self.F.zero))
        self.basis.append(A)
        return True


def _findependent(matrices, F):
    span = _FSpan(F)
    for A in matrices:
        span.add(A)
    return tuple(span.basis)


class _FC:
    """Exact coordinates in any basis of matrices of the same shape."""

    def __init__(self, basis, F):
        self.basis, self.F = tuple(basis), F
        if not basis:
            raise SymbolicVerificationError("Empty field coordinate basis")
        self.columns = tuple(zip(*(_fflat(B) for B in basis)))
        self._sparse_columns = tuple(tuple((j, x) for j, x in enumerate(row)
                                          if x != F.zero) for row in self.columns)
        _, self.rows = _frref(_ft(self.columns), F)
        if len(self.rows) != len(basis):
            raise SymbolicVerificationError("Dependent field coordinate basis")
        self.inverse = _finverse(tuple(self.columns[i] for i in self.rows), F,
                                 "field coordinate inverse")
        self._sparse_inverse = tuple(tuple((j, x) for j, x in enumerate(row)
                                          if x != F.zero) for row in self.inverse)

    def coefficients(self, A):
        v = _fflat(A)
        if len(v) != len(self.columns):
            raise _NotMember("Wrong matrix size")
        selected = tuple(v[i] for i in self.rows)
        c = tuple(sum((x * selected[j] for j, x in row
                       if selected[j] != self.F.zero), self.F.zero)
                  for row in self._sparse_inverse)
        recovered = tuple(sum((x * c[j] for j, x in row if c[j] != self.F.zero),
                              self.F.zero) for row in self._sparse_columns)
        if recovered != v:
            raise _NotMember("Matrix is outside the requested subspace")
        return c


def _fcolumn_basis(A, F):
    columns = _findependent((tuple((row[j],) for row in A)
                             for j in range(len(A[0]))), F)
    if not columns:
        raise SymbolicVerificationError("Zero invariant subspace")
    return tuple(tuple(C[i][0] for C in columns) for i in range(len(A)))


def _frepresentation(S, algebra_basis, F):
    """Restrict operators to range(S), with all invariance equations checked."""
    d = len(S[0])
    columns = tuple(tuple((row[j],) for row in S) for j in range(d))
    coords = _FC(columns, F)
    images = []
    for B in algebra_basis:
        BS = _fmul(B, S, F)
        try:
            image_columns = [coords.coefficients(tuple((row[j],) for row in BS))
                             for j in range(d)]
        except _NotMember as exc:
            raise SymbolicVerificationError("Proposed subspace is not invariant") from exc
        R = tuple(zip(*image_columns))
        _fequal(_fmul(S, R, F), BS, "restriction intertwining equation")
        images.append(R)
    return tuple(images)


def _fminimal_polynomial(b, identity, F):
    """Find a minimal polynomial from a linear dependence among powers."""
    span = _FSpan(F)
    span.add(identity)
    powers, current = [identity], identity
    variable = sp.Dummy("t")
    # The corner acts faithfully on a subspace of the ambient matrix space.
    for degree in range(1, len(b) + 1):
        current = _fmul(current, b, F)
        if not span.add(current):
            coefficients = _FC(powers, F).coefficients(current)
            native = [F.one] + [-c for c in reversed(coefficients)]
            q = sp.Poly.from_list([F.K.to_sympy(c) for c in native], variable,
                                 domain=F.K)
            _fequal(_fpoly_value(q, b, identity, F), _fz(len(b), len(b), F),
                    "minimal polynomial")
            return q
        powers.append(current)
    raise SymbolicVerificationError("No matrix minimal polynomial found")


def _fpoly_value(q, b, identity, F):
    result = _fz(len(b), len(b), F)
    # Native coefficients stay in the field: no to_number_field per operation.
    for c in q.rep.to_list():
        result = _fadd(_fmul(result, b, F), _fscale(identity, c))
    return result


def _fcrt_projector(q, factor, b, identity, F):
    complement = q.exquo(factor)
    u, v, gcd = sp.gcdex(factor, complement)
    if gcd.degree() != 0:
        raise SymbolicVerificationError("CRT polynomial factors are not coprime")
    # Normalize even if the gcd routine does not return the monic constant 1.
    h = (v * complement).rem(q).exquo_ground(gcd.LC())
    e = _fpoly_value(h, b, identity, F)
    _fequal(_fmul(e, e, F), e, "CRT idempotent")
    _fequal(_fmul(identity, e, F), e, "corner idempotent")
    return e


def _fcandidates(basis, rng, max_trials, F):
    count = min(len(basis), max_trials // 2)
    for trial in range(max_trials):
        if trial < count:
            yield basis[trial]
        else:
            weights = [F.K.convert(rng.randint(-5, 5)) for _ in basis]
            yield _flin(weights, basis, F)


def _fextension_root(q):
    """Exact root objects, avoiding cubic/quartic radical formulas."""
    try:
        # SymPy 1.14 supports algebraic coefficients through extension=True.
        roots = sp.polys.polytools.all_roots(q, radicals=False, extension=True)
    except (*_DOMAIN_ERRORS, NotImplementedError) as exc:
        raise SymbolicVerificationError("Cannot construct the required exact extension root") from exc
    if not roots:
        raise SymbolicVerificationError("Empty extension root list")
    return roots[0]


def _fsplitting_projectors(basis, identity, F, rng, max_trials, reason):
    """Try base-field CRT splits; enlarge only after the bounded cheap search."""
    extension_factor = None
    for b in _fcandidates(basis, rng, max_trials, F):
        q = _fminimal_polynomial(b, identity, F)
        _, factors = q.factor_list()
        if len(factors) > 1:
            projectors = tuple(_fcrt_projector(q, f ** exponent, b, identity, F)
                               for f, exponent in factors)
            _fequal(_flin([F.one] * len(projectors), projectors, F), identity,
                    "CRT projector completeness")
            for i, e in enumerate(projectors):
                for previous in projectors[:i]:
                    _fequal(_fmul(e, previous, F), _fz(len(e), len(e), F),
                            "CRT projector orthogonality")
            return projectors
        if factors and factors[0][0].degree() > 1:
            factor = factors[0][0]
            if extension_factor is None or factor.degree() < extension_factor.degree():
                extension_factor = factor
    if extension_factor is not None:
        raise _ExtendField(extension_factor, reason)
    raise SymbolicVerificationError(
        f"No useful polynomial-factor split in {max_trials} trials ({reason}); "
        "increase max_trials or change seed."
    )


def _fcentral_projectors(center, identity, F, rng, max_trials):
    pending, result = [identity], []
    while pending:
        p = pending.pop()
        corner_center = _findependent((_fmul(p, z, F) for z in center), F)
        if len(corner_center) == 1:
            result.append(p)
        else:
            pending.extend(_fsplitting_projectors(
                corner_center, p, F, rng, max_trials, "splitting the center"
            ))
    return tuple(result)


def _frank_idempotent(p, F):
    trace = _ftrace(p, F)
    expression = F.K.to_sympy(trace)
    if expression.is_Integer is not True:
        raise SymbolicVerificationError("Idempotent rank is not a certified integer")
    return int(expression)


def _fprimitive(basis, identity, expected_rank, F, rng, max_trials):
    p = identity
    while True:
        corner = _findependent((_fmul(_fmul(p, B, F), p, F) for B in basis), F)
        if len(corner) == 1:
            if _frank_idempotent(p, F) != expected_rank:
                raise SymbolicVerificationError("Unexpected primitive projector rank")
            return p
        projectors = _fsplitting_projectors(
            corner, p, F, rng, max_trials, "finding a primitive idempotent"
        )
        p = min(projectors, key=lambda e: _frank_idempotent(e, F))


def _fcommutant(generators, expected_dimension, F):
    n = len(generators[0])
    equations = []
    for A in generators:
        for i in range(n):
            for j in range(n):
                row = [F.zero for _ in range(n * n)]
                for c in range(n):
                    row[i * n + c] += A[c][j]
                for r in range(n):
                    row[r * n + j] -= A[i][r]
                if any(x != F.zero for x in row):
                    equations.append(tuple(row))
    if equations:
        vectors = _fnullspace(tuple(equations), F)
    else:
        vectors = tuple(tuple(F.one if i == j else F.zero for i in range(n * n))
                        for j in range(n * n))
    basis = tuple(_freshape(v, n, n) for v in vectors)
    if len(basis) != expected_dimension:
        raise SymbolicVerificationError("Unexpected commutant dimension")
    for C in basis:
        for A in generators:
            _fequal(_fmul(A, C, F), _fmul(C, A, F), "commutant membership")
    return basis


class _FPublicCoordinates:
    """Public coordinates, with an expression fallback for new scalar symbols."""

    def __init__(self, coordinates, n, F):
        self.coordinates, self.n, self.F = coordinates, n, F
        self.expression_inverse = F.output(coordinates.inverse)
        self.expression_basis = tuple(F.output(B) for B in coordinates.basis)

    def __call__(self, X):
        X = sp.Matrix(X)
        if X.shape != (self.n, self.n):
            raise ValueError("Wrong ambient matrix shape")
        try:
            M = self.F.matrix(X)
        except _DOMAIN_ERRORS:
            M = None
        if M is not None:
            try:
                coefficients = self.coordinates.coefficients(M)
            except _NotMember as exc:
                raise SymbolicVerificationError("Matrix is outside the generated algebra") from exc
            return sp.ImmutableMatrix([self.F.K.to_sympy(c) for c in coefficients])
        selected = _vec(X).extract(self.coordinates.rows, [0])
        c = _clean(self.expression_inverse * selected)
        _equal(_linear_sum(c, self.expression_basis, self.n), X,
               "symbolic coordinate membership")
        return c


class _FPublicMaps:
    def __init__(self, F, basis, coords, image_bases, inverse, n):
        self.F, self.basis, self.coords = F, basis, coords
        self.image_bases, self.inverse, self.n = image_bases, inverse, n

    def to_blocks(self, X):
        X = sp.Matrix(X)
        if X.shape != (self.n, self.n):
            raise ValueError("Wrong ambient matrix shape")
        try:
            M = self.F.matrix(X)
        except _DOMAIN_ERRORS:
            return None  # use the general expression-linear map
        try:
            coefficients = self.coords.coefficients(M)
        except _NotMember as exc:
            raise SymbolicVerificationError("Matrix is outside the generated algebra") from exc
        return tuple(self.F.output(_flin(coefficients, images, self.F))
                     for images in self.image_bases)

    def from_blocks(self, images):
        if len(images) != len(self.image_bases):
            raise ValueError("Expected one matrix per simple block")
        field_images = []
        for A, basis in zip(images, self.image_bases):
            A = sp.Matrix(A)
            size = len(basis[0])
            if A.shape != (size, size):
                raise ValueError(f"Expected a {size}-by-{size} block")
            try:
                field_images.append(self.F.matrix(A))
            except _DOMAIN_ERRORS:
                return None
        target = tuple((x,) for A in field_images for x in _fflat(A))
        coefficients = tuple(row[0] for row in _fmul(self.inverse, target, self.F))
        return self.F.output(_flin(coefficients, self.basis, self.F))


class _FullMatrixCoordinates:
    """Coordinates in the row-major matrix-unit basis of M_n(C)."""

    def __init__(self, n):
        self.n = n

    def __call__(self, X):
        X = sp.Matrix(X)
        if X.shape != (self.n, self.n):
            raise ValueError("Wrong ambient matrix shape")
        return sp.ImmutableMatrix(_vec(_clean(X)))


class _FullMatrixMaps:
    """The identity isomorphism accepts arbitrary exact symbolic scalars."""

    def __init__(self, n):
        self.n = n

    def to_blocks(self, X):
        X = sp.Matrix(X)
        if X.shape != (self.n, self.n):
            raise ValueError("Wrong ambient matrix shape")
        return (_clean(X),)

    def from_blocks(self, images):
        if len(images) != 1:
            raise ValueError("Expected one matrix per simple block")
        X = sp.Matrix(images[0])
        if X.shape != (self.n, self.n):
            raise ValueError(f"Expected a {self.n}-by-{self.n} block")
        return _clean(X)


def _full_matrix_decomposition(generators, n, F, unital, span, emit):
    """Use an exact full-rank certificate to return M_n(C) in standard units.

    Each accepted span vector is an exact linear combination of generator
    words (and, when requested, the ambient identity). The n^2 distinct pivots
    certify that their span is all of M_n, hence closure and semisimplicity.
    Changing to E_ij makes both maps the identity. The trace form is the
    transpose permutation, an involution, so its kernel is zero. The left
    ideal basis E_i0 has action X E_j0 = sum_i X_ij E_i0. These identities
    certify every product and all structural outputs without dense re-solves.
    """
    m = n * n
    if len(span.basis) != m or sorted(span.pivots) != list(range(m)):
        raise SymbolicVerificationError("Missing full matrix algebra rank certificate")
    emit(f"Algebra dimension {m}: full M_{n}(C) certified by exact rank")
    emit("Using standard matrix units and the identity isomorphism")
    units = []
    for k in range(m):
        entries = [sp.S.Zero] * m
        entries[k] = sp.S.One
        units.append(sp.ImmutableMatrix(n, n, entries))
    basis = tuple(units)
    identity = sp.ImmutableMatrix(sp.eye(n))
    left = tuple(basis[i * n] for i in range(n))
    # Tr(E_ij E_kl) = delta_jk delta_il: a permutation matrix of full rank.
    transpose = tuple((k % n) * n + k // n for k in range(m))
    if any(transpose[transpose[k]] != k for k in range(m)):
        raise SymbolicVerificationError("Invalid matrix-unit trace permutation")
    gram = sp.ImmutableMatrix(sp.ImmutableSparseMatrix(
        m, m, {(k, transpose[k]): sp.S.One for k in range(m)}))
    block = SimpleBlock(n, 1, identity, basis[0], basis, left, basis,
                        tuple(generators), "direct_restriction", identity)
    return AlgebraDecomposition(
        basis, identity, (identity,), gram, (block,), generators, unital,
        _FullMatrixCoordinates(n), sp.ImmutableMatrix(sp.eye(m)),
        coefficient_field=F.K, backend="algebraic", _field_maps=_FullMatrixMaps(n),
    )


def _number_field_once(generators, n, K, unital, max_trials, seed,
                       verify_products, commutant_max_variables, progress):
    F, rng = _FK(K), Random(seed)
    emit = progress if progress is not None else lambda message: None
    emit("Computing an algebra basis using exact field arithmetic")
    original_generators = tuple(F.matrix(A) for A in generators)
    active_generators = _findependent(original_generators, F)
    span = _FSpan(F)
    if unital:
        span.add(_fi(n, F))
    for A in active_generators:
        span.add(A)
    cursor = 0
    while cursor < len(span.basis) and len(span.basis) < n * n:
        B = span.basis[cursor]
        for A in active_generators:
            span.add(_fmul(B, A, F))
            if len(span.basis) == n * n:
                break
        cursor += 1
    if len(span.basis) == n * n:
        return _full_matrix_decomposition(generators, n, F, unital, span, emit)
    basis = tuple(span.basis)
    m = len(basis)
    emit(f"Algebra dimension {m}; checking Dickson's criterion")
    if m == 0:
        # Reuse the already verified zero-algebra conventions.
        return AlgebraDecomposition(
            (), sp.ImmutableMatrix(sp.zeros(n)), (), sp.ImmutableMatrix(0, 0, []),
            (), generators, unital, _Coordinates((), n), sp.ImmutableMatrix(0, 0, []),
            coefficient_field=K, backend="algebraic",
        )
    coords = _FC(basis, F)
    for A in original_generators:
        coords.coefficients(A)
    G = tuple(tuple(_ftrace_product(B, C, F) for C in basis) for B in basis)
    radical = _fnullspace(G, F)
    if radical:
        raise NonSemisimpleError(tuple(F.output(B) for B in basis), F.output(G),
                                tuple(F.output(_flin(v, basis, F)) for v in radical))

    emit("Computing the center")
    commutator_columns = [tuple(_fflat(_fsub(_fmul(B, A, F), _fmul(A, B, F)))
                                for B in basis) for A in active_generators]
    equations = tuple(row for columns in commutator_columns for row in zip(*columns)
                      if any(x != F.zero for x in row))
    center = (tuple(_flin(v, basis, F) for v in _fnullspace(equations, F))
              if equations else basis)
    if not center:
        raise SymbolicVerificationError("Nonzero semisimple algebra has no center")
    E = _fi(n, F)
    try:
        coords.coefficients(E)
    except _NotMember:
        # Solve for the internal identity within the exact center.
        equations = tuple(row for B in basis for row in zip(*(
            _fflat(_fmul(Z, B, F)) for Z in center
        )))
        rhs = tuple((x,) for B in basis for x in _fflat(B))
        coefficients = tuple(row[0] for row in _fsolve(
            equations, rhs, F, "internal algebra identity"
        ))
        E = _flin(coefficients, center, F)
    for B in basis:
        _fequal(_fmul(E, B, F), B, "left identity")
        _fequal(_fmul(B, E, F), B, "right identity")

    emit(f"Splitting the center of dimension {len(center)}")
    projectors = _fcentral_projectors(center, E, F, rng, max_trials)
    _fequal(_flin([F.one] * len(projectors), projectors, F), E,
            "central projector completeness")
    pending_blocks = []
    for index, p in enumerate(projectors):
        coords.coefficients(p)
        for B in basis:
            _fequal(_fmul(p, B, F), _fmul(B, p, F), "centrality")
        ideal = _findependent((_fmul(p, B, F) for B in basis), F)
        d = isqrt(len(ideal))
        N = _frank_idempotent(p, F)
        if not d or d * d != len(ideal) or N <= 0 or N % d:
            raise SymbolicVerificationError("Invalid block dimension or multiplicity")
        multiplicity = N // d
        emit(f"Block {index}: matrix size {d}, physical multiplicity {multiplicity}")
        representation_basis = None
        if multiplicity == 1:
            method = "direct_restriction"
            representation_basis = _fcolumn_basis(p, F)
            images = _frepresentation(representation_basis, basis, F)
        elif multiplicity < d and N * N <= commutant_max_variables:
            method = "commutant_restriction"
            ambient_basis = _fcolumn_basis(p, F)
            ambient_generators = _findependent(
                _frepresentation(ambient_basis, active_generators, F), F
            )
            emit(f"Block {index}: splitting its commutant of dimension {multiplicity ** 2}")
            commutant = _fcommutant(ambient_generators, multiplicity ** 2, F)
            f = _fprimitive(commutant, _fi(N, F), d, F, rng, max_trials)
            representation_basis = _fmul(ambient_basis, _fcolumn_basis(f, F), F)
            images = _frepresentation(representation_basis, basis, F)
        else:
            method = "factor_minimal_left_ideal"
            emit(f"Block {index}: finding a minimal left ideal by polynomial factors")
            e = _fprimitive(ideal, p, multiplicity, F, rng, max_trials)
            left = _findependent((_fmul(B, e, F) for B in ideal), F)
            if len(left) != d:
                raise SymbolicVerificationError("Incorrect minimal left ideal dimension")
            left_coords = _FC(left, F)
            images = tuple(tuple(zip(*(
                left_coords.coefficients(_fmul(B, v, F)) for v in left
            ))) for B in basis)
        generator_images = tuple(_flin(coords.coefficients(A), images, F)
                                 for A in original_generators)
        _fequal(_flin(coords.coefficients(p), images, F), _fi(d, F),
                "block identity image")
        pending_blocks.append((d, multiplicity, p, ideal, images, generator_images,
                               method, representation_basis))

    if sum(block[0] ** 2 for block in pending_blocks) != m:
        raise SymbolicVerificationError("Block dimensions do not sum correctly")
    emit("Verifying and inverting the complete decomposition map")
    forward = tuple(row for block in pending_blocks for row in zip(*(
        _fflat(image) for image in block[4]
    )))
    inverse = _finverse(forward, F, "complete algebra isomorphism")
    blocks, offset = [], 0
    for d, multiplicity, p, ideal, images, generator_images, method, S in pending_blocks:
        # Lift E_{i1} through the verified inverse map. This preserves the old
        # convention: images are left-action matrices in left_ideal_basis.
        left = []
        for i in range(d):
            target = tuple((F.one if j == offset + i * d else F.zero,)
                           for j in range(m))
            coefficients = tuple(row[0] for row in _fmul(inverse, target, F))
            left.append(_flin(coefficients, basis, F))
        e = left[0]
        _fequal(_fmul(e, e, F), e, "lifted primitive idempotent")
        if _frank_idempotent(e, F) != multiplicity:
            raise SymbolicVerificationError("Lifted primitive rank mismatch")
        for B, image in zip(basis, images):
            for j, v in enumerate(left):
                _fequal(_fmul(B, v, F), _flin([image[i][j] for i in range(d)], left, F),
                        "minimal left ideal action")
        blocks.append(SimpleBlock(
            d, multiplicity, F.output(p), F.output(e), tuple(F.output(B) for B in ideal),
            tuple(F.output(v) for v in left), tuple(F.output(R) for R in images),
            tuple(F.output(R) for R in generator_images), method,
            F.output(S) if S is not None else None,
        ))
        offset += d * d

    image_bases = tuple(block[4] for block in pending_blocks)
    if verify_products:
        # The active generators span the original generators. For each A in
        # this generating set, check phi(B A) = phi(B) phi(A) on every basis B.
        # Linearity extends this to every B in the algebra. Induction extends
        # it to every generator word; these words (plus I when unital) span A.
        # The already certified internal identity maps to the block identities,
        # including when unital=False. Thus this certifies ALL products exactly.
        emit("Certifying all products using basis-times-generator checks")
        image_combinations = tuple(_FLinearCombination(images, F) for images in image_bases)
        active_coefficients = tuple(coords.coefficients(A) for A in active_generators)
        active_images = tuple(tuple(combine(c) for c in active_coefficients)
                              for combine in image_combinations)
        for i, B in enumerate(basis):
            for j, C in enumerate(active_generators):
                coefficients = coords.coefficients(_fmul(B, C, F))
                for images, generator_images, combine in zip(
                        image_bases, active_images, image_combinations):
                    _fequal(combine(coefficients),
                            _fmul(images[i], generator_images[j], F),
                            "multiplication preservation")
    for i, block in enumerate(pending_blocks):
        coefficients = coords.coefficients(block[2])
        for j, images in enumerate(image_bases):
            d = pending_blocks[j][0]
            _fequal(_flin(coefficients, images, F), _fi(d, F) if i == j else _fz(d, d, F),
                    "central projector block image")
    public_coords = _FPublicCoordinates(coords, n, F)
    public_maps = _FPublicMaps(F, basis, coords, image_bases, inverse, n)
    return AlgebraDecomposition(
        tuple(F.output(B) for B in basis), F.output(E), tuple(F.output(Z) for Z in center),
        F.output(G), tuple(blocks), generators, unital, public_coords, F.output(inverse),
        coefficient_field=K, backend="algebraic", _field_maps=public_maps,
    )


def _decompose_number_field(generators, n, unital, max_trials, seed, verify_products,
                            max_extensions, commutant_max_variables, progress):
    emit = progress if progress is not None else lambda message: None
    emit("Constructing the exact coefficient field")
    K = sp.QQ
    # Only enlarge when necessary, avoiding redundant copies of the same radicals.
    for expression in dict.fromkeys(x for A in generators for x in A):
        try:
            K.from_sympy(expression)
        except _DOMAIN_ERRORS:
            old = () if K == sp.QQ else (K.ext.as_expr(),)
            try:
                K = sp.QQ.algebraic_field(*old, expression)
                K.from_sympy(expression)
            except (*_DOMAIN_ERRORS, NotImplementedError) as exc:
                raise SymbolicVerificationError("Cannot construct the input number field") from exc
    extensions = []
    while True:
        emit(f"Using coefficient field {K}")
        try:
            result = _number_field_once(
                generators, n, K, unital, max_trials, seed, verify_products,
                commutant_max_variables, progress,
            )
            # Frozen result: use dataclasses.replace to attach field history.
            from dataclasses import replace
            return replace(result, extensions=tuple(extensions))
        except _ExtendField as request:
            if len(extensions) >= max_extensions:
                raise SymbolicVerificationError(
                    f"Reached max_extensions={max_extensions}: {request.reason}"
                ) from request
            emit(f"Adjoining an exact algebraic root: {request.reason}")
            root = _fextension_root(request.polynomial)
            old = () if K == sp.QQ else (K.ext.as_expr(),)
            try:
                larger = sp.QQ.algebraic_field(*old, root)
                native_root = larger.from_sympy(root)
                value = larger.zero
                for c in request.polynomial.rep.to_list():
                    value = value * native_root + larger.from_sympy(K.to_sympy(c))
            except (*_DOMAIN_ERRORS, NotImplementedError) as exc:
                raise SymbolicVerificationError("Cannot construct a required splitting field") from exc
            if value != larger.zero:
                raise SymbolicVerificationError("Exact extension-root verification failed")
            if larger == K:
                raise SymbolicVerificationError("Requested field extension made no progress")
            K = larger
            extensions.append(root)
            # Restart so all native field elements belong to the new domain.
            # This also validates the complete final answer in a single field.
        except NonSemisimpleError:
            raise
        except (*_DOMAIN_ERRORS, NotImplementedError) as exc:
            raise SymbolicVerificationError(
                "SymPy could not complete an exact field or polynomial operation"
            ) from exc


def run_self_tests():
    """Run exact examples and invariance checks; raises on any failure.

    These are also usage examples. The complete computation and all returned
    certificates are checked by decompose_algebra on every successful case.
    """
    X = sp.Matrix([[0, 1], [1, 0]])
    Z = sp.diag(1, -1)

    def check(generators, expected, **options):
        D = decompose_algebra(generators, **options)
        assert sorted(D.block_sizes) == sorted(expected), D.block_sizes
        for A, images in zip(D.generators, D.generator_images):
            _equal(D.from_blocks(images), A, "self-test generator round trip")
        return D

    full = check([X, Z], [2])
    assert full.backend == "algebraic"
    assert full.blocks[0].method == "direct_restriction"
    assert not full.extensions
    # Obtain a noncentral idempotent using the inverse map.
    projector = full.from_blocks([sp.diag(1, 0)])
    _equal(projector * projector, projector, "self-test lifted idempotent")
    print("PASS: M_2(C), including forward/inverse maps")

    # The full-algebra path changes to standard matrix units. Check the
    # complete returned structure against independent symbolic operations.
    for adjoin_identity in (False, True):
        for verify in (False, True):
            D = check([X, Z], [2], unital=adjoin_identity, verify_products=verify)
            assert D.generator_images == ((sp.ImmutableMatrix(X),),
                                          (sp.ImmutableMatrix(Z),))
            gram = sp.ImmutableMatrix([
                [sp.trace(B * C) for C in D.algebra_basis] for B in D.algebra_basis
            ])
            assert D.dickson_matrix == gram and gram.det() != 0
            block = D.blocks[0]
            for B in D.algebra_basis:
                for C in D.algebra_basis:
                    _equal(D.to_blocks(B * C)[0],
                           D.to_blocks(B)[0] * D.to_blocks(C)[0],
                           "full-algebra multiplication")
                for j, v in enumerate(block.left_ideal_basis):
                    image = D.to_blocks(B)[0]
                    _equal(B * v, _linear_sum(image[:, j], block.left_ideal_basis, 2),
                           "full-algebra left action")
            extra = sp.Matrix([[sp.Symbol("s"), sp.sqrt(3)], [sp.pi, sp.I]])
            _equal(D.from_blocks(D.to_blocks(extra)), extra,
                   "full-algebra symbolic and outside-field scalars")
            for bad_call in (lambda: D.to_blocks(sp.eye(3)),
                             lambda: D.from_blocks([]),
                             lambda: D.from_blocks([sp.eye(3)])):
                try:
                    bad_call()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Full-algebra maps accepted invalid shapes")
    check([sp.Matrix([[2]])], [1], unital=False)
    print("PASS: full-algebra structural certificates, symbolic maps, and shapes")

    doubled = check([sp.diag(X, X), sp.diag(Z, Z)], [2])
    assert doubled.blocks[0].multiplicity == 2
    assert doubled.blocks[0].method == "factor_minimal_left_ideal"
    assert not doubled.extensions
    print("PASS: one simple block with physical multiplicity two")

    P = sp.diag(1, 1, 0, 0)
    generators = [sp.diag(X, X), sp.diag(Z, Z), P]
    direct_sum = check(generators, [2, 2])
    assert all(b.multiplicity == 1 for b in direct_sum.blocks)
    print("PASS: M_2(C) direct sum M_2(C)")

    # A nonunitary similarity preserves the abstract algebra. This also checks
    # that neither orthonormality nor closure under conjugate transpose is used.
    S = sp.Matrix([[1, 1, 0, 1], [0, 1, 1, 0],
                   [0, 0, 1, 1], [0, 0, 0, 1]])
    Sinv = S.inv()
    conjugated = check([S * A * Sinv for A in generators], [2, 2])
    print("PASS: conjugated direct sum with nonunitary change of basis")

    check([sp.diag(1, 2, 3)], [1, 1, 1])
    check([sp.diag(X, sp.zeros(1)), sp.diag(Z, sp.zeros(1))], [2, 1])
    check([sp.eye(3)], [1])
    print("PASS: commutative, mixed-size, and scalar algebras")

    nonunital = check([sp.diag(X, sp.zeros(1)), sp.diag(Z, sp.zeros(1))],
                     [2], unital=False)
    _equal(nonunital.identity, sp.diag(1, 1, 0), "self-test internal identity")
    check([sp.zeros(2)], [], unital=False)
    print("PASS: nonambient identity and zero algebra")

    check([sp.diag(1, 2, 3), sp.Matrix([[0, 1, 0],
                                      [0, 0, 1], [1, 0, 0]])], [3])
    print("PASS: M_3(C)")

    # The motivating input: many repeated identities and a full 4-by-4 algebra.
    Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    identity4 = sp.eye(4)
    motivating = [sp.kronecker_product(sp.eye(2), X) / sp.sqrt(2),
                  sp.kronecker_product(X, X) / sp.sqrt(2), identity4, identity4,
                  sp.kronecker_product(sp.eye(2), Y) / sp.sqrt(2),
                  -sp.kronecker_product(X, Y) / sp.sqrt(2)]
    motivating += [identity4] * 4
    motivating += [sp.kronecker_product(Z, sp.eye(2))]
    motivating += [identity4] * 5
    fast = check(motivating, [4], unital=False)
    assert fast.blocks[0].method == "direct_restriction"
    assert not fast.extensions
    print("PASS: motivating input takes the direct path without new algebraic roots")

    D3 = sp.diag(1, 2, 3)
    C3 = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    repeated3 = check([sp.kronecker_product(D3, sp.eye(2)),
                       sp.kronecker_product(C3, sp.eye(2))], [3])
    assert repeated3.blocks[0].multiplicity == 2
    assert repeated3.blocks[0].method == "commutant_restriction"
    assert not repeated3.extensions
    print("PASS: smaller commutant selects a copy of M_3(C)")

    # A genuinely necessary central extension.
    central_extension = check([sp.Matrix([[0, 2], [1, 0]])], [1, 1])
    assert len(central_extension.extensions) == 1
    print("PASS: necessary central quadratic field extension")

    # Rational Hamilton quaternion left-regular representation. Over C it is
    # M2 with multiplicity 2, but its primitive idempotents require an extension.
    qi = sp.Matrix([[0, -1, 0, 0], [1, 0, 0, 0],
                    [0, 0, 0, -1], [0, 0, 1, 0]])
    qj = sp.Matrix([[0, 0, -1, 0], [0, 0, 0, 1],
                    [1, 0, 0, 0], [0, -1, 0, 0]])
    quaternion = check([qi, qj], [2])
    assert quaternion.blocks[0].multiplicity == 2
    assert len(quaternion.extensions) == 1
    print("PASS: necessary noncentral quadratic field extension")

    # Independently enumerate every product on small proper subalgebras to
    # regress the shorter generator-based certificate, including multiplicity,
    # a nonambient identity, a nonunitary basis change, and a field extension.
    for D in (doubled, direct_sum, conjugated, nonunital, quaternion):
        for B in D.algebra_basis:
            for C in D.algebra_basis:
                actual = D.to_blocks(B * C)
                expected = tuple(U * V for U, V in zip(D.to_blocks(B), D.to_blocks(C)))
                for U, V in zip(actual, expected):
                    _equal(U, V, "exhaustive regression of multiplication certificate")
                _equal(D.from_blocks(actual), B * C, "exhaustive product round trip")
    outside = sp.zeros(4)
    outside[0, 0] = 1  # The two repeated 2-by-2 blocks must agree.
    try:
        doubled.to_blocks(outside)
    except SymbolicVerificationError:
        pass
    else:
        raise AssertionError("Optimized coordinates accepted an outside matrix")
    print("PASS: exhaustive products on proper subalgebras")

    check([sp.zeros(2)], [1])

    check([sp.sqrt(2) * X, sp.I * Z], [2])
    check([sp.pi * X, Z], [2])
    print("PASS: algebraic and transcendental exact constants")

    try:
        decompose_algebra([sp.Matrix([[0, 1], [0, 0]])])
    except NonSemisimpleError as exc:
        assert len(exc.radical_basis) == 1
        _equal(exc.radical_basis[0] ** 2, sp.zeros(2), "self-test radical")
    else:
        raise AssertionError("Failed to reject a nonsemisimple algebra")
    print("PASS: nonsemisimple algebra and exact radical certificate")

    parameter = sp.Symbol("a")
    check([parameter * X, Z], [2], substitutions={parameter: sp.sqrt(2)})
    for invalid in ([parameter * X, Z], [sp.Float(0.5) * X, Z]):
        try:
            decompose_algebra(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed to reject unresolved/inexact input")
    print("PASS: substitutions and invalid-input rejection")
    print("All exact self-tests passed.")


if __name__ == "__main__":
    run_self_tests()
