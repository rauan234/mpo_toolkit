"""Exact local pull-through, with a single witness independent of boundaries."""
from dataclasses import dataclass
from functools import wraps
from typing import Optional
import sympy as sp
from . import semisimple_decomposition_GPT as alg
from . import zero_testing
from .mpo import MPOTensor, MPOError, ResourceLimitError, _Arithmetic


@dataclass(frozen=True)
class LocalPullThroughResult:
    exists: bool
    h_tilde: Optional[sp.ImmutableMatrix]
    # If a solution exists, any row may independently be shifted by a linear
    # combination of these column vectors transposed into row vectors.
    row_kernel: tuple
    symbolically_certified: bool

    @property
    def unique(self):
        return self.exists and not self.row_kernel


def _arithmetic_errors_as_runtime(function):
    @wraps(function)
    def checked(*args,**kwargs):
        try:
            return function(*args,**kwargs)
        except (alg.BasePolynomialError,ZeroDivisionError) as exc:
            raise MPOError('Local pull-through arithmetic failed; existence is undecided') from exc
    return checked


@_arithmetic_errors_as_runtime
def local_pull_through(T, h, ell, *, max_block_entries=1_000_000,
                       max_linear_rows=100_000):
    """Find h_tilde with B h = h_tilde B on exactly ell physical sites.

    B[I,J] = A[i1,j1] ... A[iell,jell], with lexicographic physical basis
    order (the last site varies fastest). Equalities hold entrywise on the
    virtual indices, without boundary contraction. No injectivity or
    semisimplicity assumption is needed; h_tilde need not be Hermitian.

    Return LocalPullThroughResult: exists=False is a certified inconsistency
    under the default symbolic policy, while an undecidable computation raises
    RuntimeError. h_tilde is one solution, with free variables set to zero.
    In heuristic zero-testing mode, symbolically_certified is False.

    max_block_entries caps (d**ell)**2 * T.n**2. max_linear_rows caps
    d**ell * T.n**2 scalar equations per row of h_tilde. Equations are reduced
    incrementally, retaining only an independent subset rather than a large
    dense linear system. These are size limits, not wall-clock limits.
    """
    if not isinstance(T, MPOTensor): raise TypeError('T must be an MPOTensor')
    if type(ell) is not int or ell < 1: raise ValueError('ell must be a positive integer')
    for name,value in (('max_block_entries',max_block_entries),
                       ('max_linear_rows',max_linear_rows)):
        if type(value) is not int or value < 1: raise ValueError(f'{name} must be positive')
    policy = zero_testing.policy_key()
    d,n = T.dim_phys,T.n
    p = d**ell
    if p*p*n*n > max_block_entries:
        raise ResourceLimitError('Blocked tensor exceeds max_block_entries')
    if p*n*n > max_linear_rows:
        raise ResourceLimitError('Local pull-through system exceeds max_linear_rows')
    h = sp.ImmutableMatrix(h)
    if h.shape != (p,p): raise ValueError(f'h must have shape {(p,p)} for ell={ell}')
    (h,),_ = alg._prepare_generators((h,),None)
    arithmetic = _Arithmetic((*T.matrices,h))
    F = arithmetic.F
    zero,one = (F.zero,F.one) if F else (sp.S.Zero,sp.S.One)
    native_h = arithmetic.native(h)
    local = [[arithmetic.native(T.A_list[i][j]) for j in range(d)] for i in range(d)]
    blocks = [[arithmetic.native(sp.eye(n))]]
    for _ in range(ell):
        size = len(blocks)
        following = [[None]*(size*d) for _ in range(size*d)]
        for I in range(size):
            for J in range(size):
                for i in range(d):
                    for j in range(d):
                        following[I*d+i][J*d+j] = arithmetic.multiply(blocks[I][J],local[i][j])
        blocks = following

    def entry(matrix,i,j): return matrix[i][j] if F else matrix[i,j]
    def finish(exists,witness=None,kernel=()):
        if zero_testing.policy_key() != policy:
            raise MPOError('Zero-testing policy changed during local pull-through')
        return LocalPullThroughResult(exists,witness,kernel,policy[2])

    # Each output row i of h_tilde obeys the SAME coefficient matrix M.
    # M[(j,u,v),k] = B[k,j][u,v]; RHS[(j,u,v),i] = (B h)[i,j][u,v].
    # Solve all p right-hand sides simultaneously, for h_tilde.T.
    rows = arithmetic.span(2*p)
    for j in range(p):
        for u in range(n):
            for v in range(n):
                coefficients = tuple(entry(blocks[k][j],u,v) for k in range(p))
                rhs = tuple(sum((entry(blocks[i][k],u,v)*entry(native_h,k,j)
                                 for k in range(p)),zero) for i in range(p))
                row = coefficients+rhs
                added = rows.add((row,) if F else sp.ImmutableMatrix([row]))
                if added and rows.pivots[-1] >= p:
                    return finish(False)

    if not rows.basis:
        reduced,pivots = (),()
    elif F:
        reduced,pivots = alg._frref(tuple(A[0] for A in rows.basis),F)
    else:
        R,pivots = alg._rref(sp.Matrix.vstack(*rows.basis))
        reduced = tuple(tuple(R[i,j] for j in range(R.cols)) for i in range(R.rows))
    if any(k >= p for k in pivots): return finish(False)
    solution = [[zero]*p for _ in range(p)]
    for r,k in enumerate(pivots):
        for i in range(p): solution[i][k] = reduced[r][p+i]
    witness = arithmetic.output(tuple(tuple(row) for row in solution) if F else sp.Matrix(solution))
    kernel = []
    for free in range(p):
        if free in pivots: continue
        vector = [zero]*p
        vector[free] = one
        for r,k in enumerate(pivots): vector[k] = -reduced[r][free]
        column = tuple((x,) for x in vector) if F else sp.Matrix(vector)
        kernel.append(arithmetic.output(column))

    # Verify the actual blocked virtual matrices, independently of the layout
    # of the flattened linear equations used above.
    for i in range(p):
        for j in range(p):
            if F:
                left = alg._flin(tuple(entry(native_h,k,j) for k in range(p)),blocks[i],F)
                right = alg._flin(solution[i],tuple(blocks[k][j] for k in range(p)),F)
                alg._fequal(left,right,'exact local pull-through')
            else:
                left = sum((blocks[i][k]*native_h[k,j] for k in range(p)),sp.zeros(n))
                right = sum((solution[i][k]*blocks[k][j] for k in range(p)),sp.zeros(n))
                alg._equal(left,right,'exact local pull-through')
    return finish(True,witness,tuple(kernel))
