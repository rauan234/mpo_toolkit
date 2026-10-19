"""Exact symbolic MPO tensors and certified finite fusion-category reconstruction.

Conventions: product(a,b) has virtual space V_a tensor V_b and physical
operator O_a O_b; a.compose(b) = product(b,a), as requested by the caller.
No adjoints are used as substitutes for algebraic inverse maps.
"""
from dataclasses import dataclass, field
from itertools import product
from typing import Optional
import sympy as sp
from . import semisimple_decomposition_GPT as alg
from .decomposition_embeddings import extract_embeddings
from . import zero_testing


class MPOError(RuntimeError):
    """A requested exact MPO construction could not be certified."""


class ResourceLimitError(MPOError):
    """Search exhausted; this is NOT a proof of infinitude or nonexistence."""


class NotBlockDecomposableError(MPOError):
    """No exact direct sum of injective blocks of the requested type exists."""


class NotFusionCategoryError(MPOError):
    """The computed closure fails a required fusion-category axiom."""


@dataclass(frozen=True)
class Limits:
    max_blocking: int = 128
    max_virtual_dim: int = 256
    max_gauge_variables: int = 4096
    max_f_blocks: int = 10000
    max_pentagon_paths: int = 100000
    max_label_quadruples: int = 1000000
    max_trials: int = 40
    max_extensions: int = 8

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < (0 if name == 'max_extensions' else 1):
                raise ValueError(f'Invalid resource limit {name}={value!r}')


def _clean(A):
    return alg._clean(A)


def _scalar(x):
    return alg._simp(x)


def _eq(A, B, label):
    alg._equal(A, B, label)


def _nonzero(x, label):
    flag = alg._zero(x)
    if flag is not False:
        raise MPOError(f'Cannot certify a nonzero scalar for {label}')
    return _scalar(x)


class _Arithmetic:
    """Reuse the exact number-field backend, with conservative symbolic fallback."""
    def __init__(self, matrices):
        values = tuple(dict.fromkeys(x for A in matrices for x in A))
        self.F = None
        if all(x.is_algebraic is True for x in values):
            try:
                K = sp.QQ
                for x in values:
                    try:
                        K.from_sympy(x)
                    except alg._DOMAIN_ERRORS:
                        previous = () if K == sp.QQ else (K.ext.as_expr(),)
                        K = sp.QQ.algebraic_field(*previous, x)
                self.F = alg._FK(K)
            except (*alg._DOMAIN_ERRORS, NotImplementedError):
                # An unavailable number-field conversion is not a negative
                # mathematical answer. Fall back to certified expression tests.
                self.F = None

    def native(self, A):
        return self.F.matrix(A) if self.F else _clean(A)

    def output(self, A):
        return self.F.output(A) if self.F else _clean(A)

    def span(self, n):
        return alg._FSpan(self.F) if self.F else alg._Span(n)

    def multiply(self, A, B):
        return alg._fmul(A, B, self.F) if self.F else _clean(A * B)

    def signature(self, basis):
        if not basis:
            return ()
        if self.F:
            rows = tuple(alg._fflat(A) for A in basis)
            R, _ = alg._frref(rows, self.F)
            return R
        rows = sp.Matrix.vstack(*(sp.Matrix(1, len(A), list(A)) for A in basis))
        R, _ = alg._rref(rows)
        return tuple(tuple(R[i, j] for j in range(R.cols)) for i in range(R.rows))


def _nullspace(M):
    if M.rows == 0:
        return tuple(sp.eye(M.cols)[:, i] for i in range(M.cols))
    arithmetic = _Arithmetic([M])
    if arithmetic.F:
        rows = arithmetic.native(M)
        vectors = alg._fnullspace(rows, arithmetic.F)
        return tuple(sp.ImmutableMatrix([arithmetic.F.K.to_sympy(x) for x in v])
                     for v in vectors)
    return alg._nullspace(M)


@dataclass(frozen=True)
class FusionMap:
    """inclusion: V_simple -> V_product; retraction: V_product -> V_simple."""
    inclusion: sp.ImmutableMatrix
    retraction: sp.ImmutableMatrix

    @property
    def X(self):
        return self.inclusion

    @property
    def X_plus(self):
        """Algebraic retraction; generally NOT X.H."""
        return self.retraction


@dataclass(frozen=True)
class DirectSumDecomposition:
    tensors: tuple
    multiplicities: tuple
    fusion_tensors: tuple
    S: sp.ImmutableMatrix
    S_inverse: sp.ImmutableMatrix
    zero_inclusion: sp.ImmutableMatrix
    zero_projection: sp.ImmutableMatrix
    algebra_decomposition: object = field(repr=False, compare=False)
    symbolically_certified: bool = True

    def __iter__(self):
        # The requested three-output unpacking, while retaining zero-sector data.
        yield list(self.tensors)
        yield list(self.multiplicities)
        yield [list(maps) for maps in self.fusion_tensors]

    @property
    def zero_dimension(self):
        return self.zero_inclusion.cols

    def reconstruct(self, i, j):
        n = self.S.rows
        result = sp.zeros(n)
        for tensor, maps in zip(self.tensors, self.fusion_tensors):
            for f in maps:
                result += f.inclusion * tensor.A_list[i][j] * f.retraction
        return _clean(result)


class MPOTensor:
    """An immutable array of exact symbolic virtual matrices.

    Numerical Float entries, free parameters and infinite entries are rejected.
    The optional Limits object bounds searches without conflating failure to
    decide with a mathematical negative answer. Numerical hints require symbolic
    confirmation unless the global zero-testing policy explicitly disables it.
    """
    def __init__(self, dim_phys, A_list, *, limits=None, name=None):
        if type(dim_phys) is not int or dim_phys < 1:
            raise ValueError('dim_phys must be a positive integer')
        rows = tuple(tuple(row) for row in A_list)
        if len(rows) != dim_phys or any(len(row) != dim_phys for row in rows):
            raise ValueError('A_list must have shape (dim_phys, dim_phys)')
        flat, n = alg._prepare_generators(tuple(A for row in rows for A in row), None)
        self._dim_phys, self._n = dim_phys, n
        self._A_list = tuple(tuple(flat[i * dim_phys + j] for j in range(dim_phys))
                             for i in range(dim_phys))
        self.limits = limits or Limits()
        self.name = name
        self._injective = None
        self._injectivity_length = None
        self._decomposition = None
        self._cache_policy = zero_testing.policy_key()

    def _check_policy(self):
        policy = zero_testing.policy_key()
        if policy != self._cache_policy:
            self._injective = self._injectivity_length = self._decomposition = None
            self._cache_policy = policy
        return policy

    @property
    def dim_phys(self): return self._dim_phys

    @property
    def n(self): return self._n

    @property
    def A_list(self): return self._A_list

    @property
    def matrices(self): return tuple(A for row in self.A_list for A in row)

    @property
    def injectivity_length(self):
        """First certified blocking length, None if not yet certified injective."""
        self._check_policy()
        return self._injectivity_length

    @property
    def injectivity_symbolically_certified(self):
        """Whether the cached injectivity decision used strict verification."""
        self._check_policy()
        return self._injective is not None and self._cache_policy[2]

    def __repr__(self):
        return f'MPOTensor(dim_phys={self.dim_phys}, n={self.n}, name={self.name!r})'

    @staticmethod
    def product(left, right, *, limits=None):
        """Physical O_left O_right, with virtual ordering V_left tensor V_right."""
        if not isinstance(left, MPOTensor) or not isinstance(right, MPOTensor):
            raise TypeError('Both operands must be MPOTensor objects')
        if left.dim_phys != right.dim_phys:
            raise ValueError('Physical dimensions must agree')
        lim = limits or left.limits
        if left.n * right.n > lim.max_virtual_dim:
            raise ResourceLimitError('Product virtual dimension exceeds max_virtual_dim')
        p = left.dim_phys
        matrices = [[sum((sp.kronecker_product(left.A_list[i][k], right.A_list[k][j])
                          for k in range(p)), sp.zeros(left.n * right.n))
                     for j in range(p)] for i in range(p)]
        return MPOTensor(p, matrices, limits=lim)

    def compose(self, other):
        """Return other after self: sum_k other[i,k] tensor self[k,j]."""
        return self.product(other, self, limits=self.limits)

    def transpose_physical(self):
        """Return B[i,j] = A[j,i], exchanging only the physical indices.

        Entries are not conjugated and virtual matrices are not transposed.
        For every periodic length L, B.periodic_operator(L) equals
        self.periodic_operator(L).T. Returns a new tensor with the same
        physical/virtual dimensions and resource limits.
        """
        return MPOTensor(self.dim_phys,
            [[self.A_list[j][i] for j in range(self.dim_phys)]
             for i in range(self.dim_phys)], limits=self.limits)

    def conjugate(self):
        """Return the physical MPO adjoint, B[i,j] = conjugate(A[j,i]).

        Physical input/output indices are exchanged; virtual matrices are
        conjugated entrywise, NOT transposed. Thus for every periodic length L,
        B.periodic_operator(L) equals self.periodic_operator(L).H.
        The result is a new symbolic tensor with the same dimensions and limits.
        """
        return MPOTensor(self.dim_phys,
            [[self.A_list[j][i].conjugate() for j in range(self.dim_phys)]
             for i in range(self.dim_phys)], limits=self.limits)

    def test_injective(self, *, max_blocking=None):
        """Decide fixed-length injectivity, not merely full generated algebra.

        True: words of one common length span M_n. False: the total word
        algebra is proper, or exact fixed-length spans repeat before reaching
        M_n (a deterministic cycle). Resource/undecidable tests raise RuntimeError.
        """
        self._check_policy()
        if self._injective is not None:
            return self._injective
        bound = self.limits.max_blocking if max_blocking is None else max_blocking
        if type(bound) is not int or bound < 1:
            raise ValueError('max_blocking must be a positive integer')
        if self.n > self.limits.max_virtual_dim:
            raise ResourceLimitError('Virtual dimension exceeds max_virtual_dim')
        arithmetic = _Arithmetic(self.matrices)
        initial = arithmetic.span(self.n)
        for A in self.matrices:
            initial.add(arithmetic.native(A))
        generators = tuple(initial.basis)
        total = arithmetic.span(self.n)
        for A in generators: total.add(A)
        cursor = 0
        while cursor < len(total.basis) and len(total.basis) < self.n ** 2:
            B = total.basis[cursor]
            for A in generators:
                total.add(arithmetic.multiply(B, A))
                if len(total.basis) == self.n ** 2: break
            cursor += 1
        if len(total.basis) != self.n ** 2:
            self._injective = False
            return False
        current, history = generators, []
        for length in range(1, bound + 1):
            if len(current) == self.n ** 2:
                self._injective, self._injectivity_length = True, length
                return True
            signature = arithmetic.signature(current)
            if any(signature == previous for previous in history):
                self._injective = False
                return False
            history.append(signature)
            if length < bound:
                following = arithmetic.span(self.n)
                for B in current:
                    for A in generators:
                        following.add(arithmetic.multiply(B, A))
                current = tuple(following.basis)
        raise ResourceLimitError(f'Injectivity undecided after {bound} blocking lengths')

    def gauge_to(self, other):
        """Return H: V_self -> V_other with other[i,j] H = H self[i,j].

        Both tensors must be injective. Returns None for certified inequivalence.
        No overall scalar/phase is quotiented out. Every returned H is invertible.
        """
        if not isinstance(other, MPOTensor): raise TypeError('Expected MPOTensor')
        if (self.n, self.dim_phys) != (other.n, other.dim_phys): return None
        if not self.test_injective() or not other.test_injective():
            raise MPOError('gauge_to requires two injective tensors')
        if self.matrices == other.matrices:
            return sp.ImmutableMatrix(sp.eye(self.n))
        n = self.n
        if n * n > self.limits.max_gauge_variables:
            raise ResourceLimitError('Gauge system exceeds max_gauge_variables')
        eye = sp.eye(n)
        equations = sp.Matrix.vstack(*(
            sp.kronecker_product(B, eye) - sp.kronecker_product(eye, A.T)
            for A, B in zip(self.matrices, other.matrices)))
        vectors = _nullspace(equations)
        if not vectors: return None
        if len(vectors) != 1:
            raise MPOError('Unexpected intertwiner dimension for injective tensors')
        H = _clean(sp.Matrix(n, n, list(vectors[0])))
        alg._inverse(H, 'gauge intertwiner')
        for A, B in zip(self.matrices, other.matrices):
            _eq(B * H, H * A, 'gauge intertwining')
        return H

    def direct_sum_decomp(self, *, allow_zero=True):
        """Exact direct sum of tensors injective after blocking, with multiplicities.

        Returns DirectSumDecomposition, unpackable as (tensors, multiplicities,
        fusion_tensors). fusion_tensors[a][mu] contains inclusion and retraction.
        One-site injectivity is not required: each returned tensor has a certified
        injectivity_length L, at which its length-L matrix products span M_n.
        The returned matrices remain one-site tensors with the original physical
        dimension; this method does not explicitly group physical sites.
        Zero-action sectors are reported separately, never counted as simples.
        Joint fixed-length independence of distinct blocks is NOT assumed.
        """
        policy = self._check_policy()
        if self._decomposition is not None:
            if not allow_zero and self._decomposition.zero_dimension:
                raise NotBlockDecomposableError('A zero-action sector is present')
            return self._decomposition
        if self.n > self.limits.max_virtual_dim:
            raise ResourceLimitError('Virtual dimension exceeds max_virtual_dim')
        try:
            D = alg.decompose_algebra(self.matrices, unital=False,
                                      max_trials=self.limits.max_trials,
                                      max_extensions=self.limits.max_extensions)
        except alg.NonSemisimpleError as exc:
            raise NotBlockDecomposableError('Virtual algebra is not semisimple') from exc
        tensors = []
        for a in range(len(D.blocks)):
            images = [row[a] for row in D.generator_images]
            tensor = MPOTensor(self.dim_phys,
                [images[i*self.dim_phys:(i+1)*self.dim_phys] for i in range(self.dim_phys)],
                limits=self.limits)
            if not tensor.test_injective():
                raise NotBlockDecomposableError('A simple algebra block is not injective after blocking')
            tensors.append(tensor)
        embedding = extract_embeddings(D)
        maps = tuple(tuple(FusionMap(F, G) for F, G in zip(Fs, Gs))
                     for Fs, Gs in zip(embedding['F'], embedding['G']))
        result = DirectSumDecomposition(tuple(tensors), tuple(embedding['m']), maps,
            embedding['S'], embedding['phi'], embedding['F_zero'], embedding['G_zero'], D,
            symbolically_certified=policy[2])
        # This also certifies no triangular or nilpotent leftovers were discarded.
        for i in range(self.dim_phys):
            for j in range(self.dim_phys):
                _eq(result.reconstruct(i, j), self.A_list[i][j], 'tensor reconstruction')
                for tensor, channels in zip(tensors, maps):
                    for f in channels:
                        _eq(self.A_list[i][j] * f.X, f.X * tensor.A_list[i][j], 'inclusion zipper')
                        _eq(f.X_plus * self.A_list[i][j], tensor.A_list[i][j] * f.X_plus,
                            'retraction zipper')
        self._decomposition = result
        if not allow_zero and result.zero_dimension:
            raise NotBlockDecomposableError('A zero-action sector is present')
        return result

    def periodic_operator(self, length, *, max_entries=1000000):
        """Small-system utility, ordinary periodic trace; no boundary insertion."""
        if type(length) is not int or length < 1: raise ValueError('length must be positive')
        p = self.dim_phys
        if p ** (2 * length) > max_entries:
            raise ResourceLimitError('Periodic operator exceeds max_entries')
        words = tuple(product(range(p), repeat=length))
        entries = []
        for ii in words:
            for jj in words:
                matrix = sp.eye(self.n)
                for i, j in zip(ii, jj): matrix = matrix * self.A_list[i][j]
                entries.append(_scalar(sp.trace(matrix)))
        return sp.ImmutableMatrix(len(words), len(words), entries)


@dataclass(frozen=True)
class FSymbol:
    """Rows (f,lambda,sigma), columns (e,mu,nu), with L_col=sum R_row F[row,col]."""
    left_paths: tuple
    right_paths: tuple
    matrix: sp.ImmutableMatrix


@dataclass
class FusionCat:
    """A certified finite semisimple rigid monoidal category in fusion bases.

    N[a][b][c] uses physical O_a O_b and virtual V_a tensor V_b, NOT the
    reversed argument convention of MPOTensor.compose. F_symbols[(a,b,c,d)]
    follows arXiv:1511.08090v2 Eq. (29). Fusion channels are zero-based.
    This is not a claim of a unitary, braided, spherical, or modular structure.
    """
    simples: tuple
    N: tuple
    fusion_tensors: dict
    pair_zero_sectors: dict
    F_symbols: dict = field(default_factory=dict)
    unit: Optional[int] = None
    left_unitors: tuple = ()
    right_unitors: tuple = ()
    duals: tuple = ()
    evaluations: tuple = ()
    coevaluations: tuple = ()
    pentagon_verified: bool = False
    symbolically_certified: bool = False
    limits: Limits = field(default_factory=Limits)

    @property
    def num_simples(self): return len(self.simples)

    def fusion(self, a, b):
        return {c: value for c, value in enumerate(self.N[a][b]) if value}

    def _left_paths(self, a, b, c, d):
        return tuple((e, mu, nu) for e in range(self.num_simples)
                     for mu in range(self.N[a][b][e]) for nu in range(self.N[e][c][d]))

    def _right_paths(self, a, b, c, d):
        return tuple((f, lam, sig) for f in range(self.num_simples)
                     for lam in range(self.N[b][c][f]) for sig in range(self.N[a][f][d]))

    def _compute_F_symbols(self):
        if self.num_simples ** 4 > self.limits.max_label_quadruples:
            raise ResourceLimitError('Associator label enumeration exceeds max_label_quadruples')
        count = 0
        for a, b, c, d in product(range(self.num_simples), repeat=4):
            left_paths, right_paths = self._left_paths(a,b,c,d), self._right_paths(a,b,c,d)
            if len(left_paths) != len(right_paths):
                raise NotFusionCategoryError('Fusion coefficients fail associativity')
            if not left_paths: continue
            count += 1
            if count > self.limits.max_f_blocks:
                raise ResourceLimitError('F-symbol computation exceeds max_f_blocks')
            na, nc, nd = self.simples[a].n, self.simples[c].n, self.simples[d].n
            if na * self.simples[b].n * nc > self.limits.max_virtual_dim:
                raise ResourceLimitError('Triple virtual dimension exceeds max_virtual_dim')
            L, LP, R, RP = [], [], [], []
            for e, mu, nu in left_paths:
                first, second = self.fusion_tensors[a,b,e][mu], self.fusion_tensors[e,c,d][nu]
                L.append(_clean(sp.kronecker_product(first.X, sp.eye(nc)) * second.X))
                LP.append(_clean(second.X_plus * sp.kronecker_product(first.X_plus, sp.eye(nc))))
            for f, lam, sig in right_paths:
                first, second = self.fusion_tensors[b,c,f][lam], self.fusion_tensors[a,f,d][sig]
                R.append(_clean(sp.kronecker_product(sp.eye(na), first.X) * second.X))
                RP.append(_clean(second.X_plus * sp.kronecker_product(sp.eye(na), first.X_plus)))
            values = []
            for projection in RP:
                for inclusion in L:
                    overlap = _clean(projection * inclusion)
                    value = _scalar(sp.trace(overlap) / nd)
                    _eq(overlap, value * sp.eye(nd), 'Schur scalar for associator')
                    values.append(value)
            matrix = sp.ImmutableMatrix(len(R), len(L), values)
            inverse = alg._inverse(matrix, 'F-symbol block')
            for j, inclusion in enumerate(L):
                _eq(inclusion, sum((R[i] * matrix[i,j] for i in range(len(R))),
                                   sp.zeros(*inclusion.shape)), 'associator inclusion identity')
            for i, projection in enumerate(RP):
                _eq(projection, sum((matrix[i,j] * LP[j] for j in range(len(L))),
                                    sp.zeros(*projection.shape)), 'associator projection identity')
            # In particular, the two fusion trees have exactly the same support.
            _eq(matrix * inverse, sp.eye(len(R)), 'associator inverse')
            self.F_symbols[a,b,c,d] = FSymbol(left_paths, right_paths, matrix)

    def _tree_paths(self, tree, labels):
        """Decorated tree: leaf (a,), internal (out,left,right,channel)."""
        if isinstance(tree, int):
            yield (labels[tree],)
            return
        for left in self._tree_paths(tree[0], labels):
            for right in self._tree_paths(tree[1], labels):
                for out in range(self.num_simples):
                    for channel in range(self.N[left[0]][right[0]][out]):
                        yield (out, left, right, channel)

    def _rotate(self, path, location):
        if location:
            side = location[0]
            for changed, coefficient in self._rotate(path[1+side], location[1:]):
                parts = list(path)
                parts[1+side] = changed
                yield tuple(parts), coefficient
            return
        d, left, C, nu = path
        e, A, B, mu = left
        block = self.F_symbols[A[0], B[0], C[0], d]
        column = block.left_paths.index((e, mu, nu))
        for row, (f, lam, sig) in enumerate(block.right_paths):
            coefficient = block.matrix[row, column]
            if alg._zero(coefficient) is not True:
                yield (d, A, (f, B, C, lam), sig), coefficient

    def _route(self, start, locations):
        states = {start: sp.S.One}
        for location in locations:
            following = {}
            for state, coefficient in states.items():
                for target, entry in self._rotate(state, location):
                    following[target] = _scalar(following.get(target, 0) + coefficient * entry)
            states = following
        return states

    def verify_pentagon(self):
        """Check all five-tree pentagons using the stored F matrices, not numerical tolerances."""
        self.pentagon_verified = False
        if self.num_simples ** 4 > self.limits.max_label_quadruples:
            raise ResourceLimitError('Pentagon label enumeration exceeds max_label_quadruples')
        count = 0
        for labels in product(range(self.num_simples), repeat=4):
            for path in self._tree_paths((((0,1),2),3), labels):
                count += 1
                if count > self.limits.max_pentagon_paths:
                    raise ResourceLimitError('Pentagon check exceeds max_pentagon_paths')
                long = self._route(path, ((0,), (), (1,)))
                short = self._route(path, ((), ()))
                for target in long.keys() | short.keys():
                    alg._require_zero(long.get(target, 0) - short.get(target, 0), 'pentagon')
        self.pentagon_verified = True
        return True

    def _unit_and_duals(self):
        s = self.num_simples
        units = [e for e in range(s) if all(
            self.N[e][a][b] == self.N[a][e][b] == int(a == b)
            for a in range(s) for b in range(s))]
        if len(units) != 1:
            raise NotFusionCategoryError('Finite closure does not have a unique simple unit')
        e = units[0]
        self.unit = e
        # Triangle: l_b F^(a,e,b)_d = r_a I in the unit-channel bases.
        def unit_scalar(a,b):
            value = None
            for d in range(s):
                multiplicity = self.N[a][b][d]
                if not multiplicity: continue
                block = self.F_symbols[a,e,b,d]
                current = _scalar(block.matrix[0,0])
                _nonzero(current, 'unit associator')
                _eq(block.matrix, current * sp.eye(multiplicity), 'unit-channel scalar')
                if value is None: value = current
                else: alg._require_zero(value-current, 'consistent unit scalar')
            if value is None:
                raise NotFusionCategoryError('A tensor product of simples is zero')
            return value
        alg._require_zero(unit_scalar(e,e)-1, 'unit associativity')
        left = tuple(_scalar(1 / unit_scalar(e,b)) for b in range(s))
        right = tuple(unit_scalar(a,e) for a in range(s))
        for a,b in product(range(s),repeat=2):
            alg._require_zero(left[b]*unit_scalar(a,b)-right[a], 'triangle identity')
        self.left_unitors, self.right_unitors = left, right
        duals, evaluations, coevaluations = [], [], []
        for a in range(s):
            accepted = []
            for b in range(s):
                if self.N[a][b][e] != 1 or self.N[b][a][e] != 1: continue
                block = self.F_symbols[a,b,a,a]
                v = _scalar(block.matrix[block.right_paths.index((e,0,0)),
                                         block.left_paths.index((e,0,0))])
                flag = alg._zero(v)
                if flag is True: continue
                if flag is None: raise MPOError('Duality scalar is undecidable')
                evaluation = _scalar(left[a] / (right[a] * v))
                other = self.F_symbols[b,a,b,b]
                inverse = alg._inverse(other.matrix, 'dual associator')
                w = inverse[other.left_paths.index((e,0,0)), other.right_paths.index((e,0,0))]
                second_snake = _scalar(evaluation * w * left[b] / right[b] - 1)
                flag = alg._zero(second_snake)
                if flag is None: raise MPOError('Snake identity is undecidable')
                if flag: accepted.append((b,evaluation))
            if len(accepted) != 1:
                raise NotFusionCategoryError(f'Simple object {a} has no unique certified dual')
            b, evaluation = accepted[0]
            duals.append(b)
            evaluations.append(evaluation)
            coevaluations.append(sp.S.One)
        self.duals = tuple(duals)
        # ev_a: dual(a) tensor a -> unit, coev_a: unit -> a tensor dual(a).
        # Scalars multiply the corresponding channel retraction/inclusion.
        self.evaluations, self.coevaluations = tuple(evaluations), tuple(coevaluations)


def generate_fusion_category(tensors, max_tensors=1000, *, limits=None,
                             allow_zero=True, progress=None):
    """Close injective tensors under ordered physical multiplication, then certify
    associators, pentagons, a simple unit, triangle identities, and duality snakes.

    Gauge equivalence is exact simultaneous similarity; phases/scalars are NOT
    identified. No new physical-identity object is silently adjoined. Reaching
    any cap raises ResourceLimitError, never an assertion of infinite closure.
    """
    policy = zero_testing.policy_key()
    inputs = tuple(tensors)
    if not inputs or any(not isinstance(T, MPOTensor) for T in inputs):
        raise ValueError('Supply a nonempty list of MPOTensor objects')
    if type(max_tensors) is not int or max_tensors < 1:
        raise ValueError('max_tensors must be positive')
    if any(T.dim_phys != inputs[0].dim_phys for T in inputs):
        raise ValueError('All physical dimensions must agree')
    lim = limits or inputs[0].limits
    emit = progress or (lambda message: None)
    # Work on independent immutable data, so search settings do not mutate inputs.
    simples = []
    def register(tensor):
        for c, known in enumerate(simples):
            gauge = tensor.gauge_to(known)  # maps V_new -> V_known
            if gauge is not None: return c, gauge
        if len(simples) >= max_tensors:
            raise ResourceLimitError(f'Closure exceeds max_tensors={max_tensors}; finiteness undecided')
        label = len(simples)
        simples.append(tensor)
        emit(f'Added simple {label}, virtual dimension {tensor.n}')
        return label, sp.ImmutableMatrix(sp.eye(tensor.n))
    for T in inputs:
        tensor = MPOTensor(T.dim_phys,T.A_list,limits=lim,name=T.name)
        if not tensor.test_injective():
            raise NotBlockDecomposableError('Every input must be injective after blocking')
        register(tensor)
    pairs, maps, zeros = {}, {}, {}
    while True:
        todo = next(((a,b) for a in range(len(simples)) for b in range(len(simples))
                     if (a,b) not in pairs), None)
        if todo is None: break
        a,b = todo
        emit(f'Decomposing product ({a}, {b}); {len(simples)} simples known')
        tensor = MPOTensor.product(simples[a],simples[b],limits=lim)
        decomposition = tensor.direct_sum_decomp(allow_zero=allow_zero)
        counts = {}
        for summand, multiplicity, channels in zip(decomposition.tensors,
                decomposition.multiplicities,decomposition.fusion_tensors):
            c,H = register(summand)
            Hinv = alg._inverse(H, 'gauge alignment')
            aligned = tuple(FusionMap(_clean(f.X * Hinv), _clean(H * f.X_plus)) for f in channels)
            for f in aligned:
                _eq(f.X_plus * f.X, sp.eye(simples[c].n), 'aligned fusion splitting')
                for A,C in zip(tensor.matrices,simples[c].matrices):
                    _eq(A*f.X,f.X*C,'aligned inclusion zipper')
                    _eq(f.X_plus*A,C*f.X_plus,'aligned retraction zipper')
            maps[a,b,c] = maps.get((a,b,c),()) + aligned
            counts[c] = counts.get(c,0) + multiplicity
        pairs[a,b] = counts
        zeros[a,b] = FusionMap(decomposition.zero_inclusion,decomposition.zero_projection)
    n = len(simples)
    N = tuple(tuple(tuple(pairs[a,b].get(c,0) for c in range(n))
                    for b in range(n)) for a in range(n))
    category = FusionCat(tuple(simples),N,maps,zeros,limits=lim)
    emit(f'Closure complete: {n} simples; computing exact associators')
    category._compute_F_symbols()
    emit('Checking all pentagons')
    category.verify_pentagon()
    emit('Checking the unit, triangles, and duality snakes')
    category._unit_and_duals()
    if zero_testing.policy_key() != policy:
        raise MPOError('Zero-testing policy changed during the search; rerun under one policy')
    category.symbolically_certified = policy[2]
    emit('Fusion category certified' if policy[2] else
         'Fusion-category checks passed under heuristic policy; not a symbolic certificate')
    return category
