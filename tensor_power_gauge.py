"""Existence of an invertible identical on-site change of physical basis."""
import sympy as sp
from . import semisimple_decomposition_GPT as alg
from .mpo import _Arithmetic, MPOError, ResourceLimitError


def tensor_power_intertwiner_exists(h, h_prime, dim_phys, ell, *,
                                    max_variables=17, max_matrix_entries=10_000,
                                    progress=None):
    """Decide whether M**(tensor ell) h_prime = h M**(tensor ell) for M in GL_d(C).

    Return a bool. Exact rational/algebraic complex coefficients are supported.
    Unresolved parameters and floats are rejected. Other exact constants may
    return immediately for identical inputs; otherwise they raise RuntimeError
    if a certified number field is unavailable. No numerical tolerance or
    heuristic zero-testing policy is used to decide the polynomial ideal.

    A Groebner basis tests whether the equations together with z*det(M)-1
    generate the unit ideal. This also handles positive-dimensional solution
    sets, and permits M to have entries outside the input coefficient field.
    A True answer establishes existence over C; it does not extract a witness.

    max_variables bounds d**2+1; max_matrix_entries bounds d**(2*ell).
    These do NOT impose a time limit on the Groebner computation, which can
    be expensive even for small matrices. progress=print reports its stages.
    """
    for name,value in (('dim_phys',dim_phys),('ell',ell),
                       ('max_variables',max_variables),('max_matrix_entries',max_matrix_entries)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    d,p = dim_phys,dim_phys**ell
    if p*p > max_matrix_entries:
        raise ResourceLimitError('Physical operator exceeds max_matrix_entries')
    h,h_prime = sp.ImmutableMatrix(h),sp.ImmutableMatrix(h_prime)
    if h.shape != (p,p) or h_prime.shape != (p,p):
        raise ValueError(f'h and h_prime must both have shape {(p,p)}')
    (h,h_prime),_ = alg._prepare_generators((h,h_prime),None)
    if h == h_prime:
        return True  # M = I, including identical transcendental input matrices.
    if d*d+1 > max_variables:
        raise ResourceLimitError('Polynomial system exceeds max_variables (d**2 + 1)')
    emit = progress or (lambda message: None)
    emit('Constructing the exact coefficient field')
    try:
        arithmetic = _Arithmetic((h,h_prime))
        if arithmetic.F is None:
            raise MPOError('Cannot construct a certified algebraic coefficient field; existence undecided')
        K = arithmetic.F.K
        # Canonicalize in one common field before constructing polynomials.
        h = arithmetic.output(arithmetic.native(h))
        h_prime = arithmetic.output(arithmetic.native(h_prime))
        variables = tuple(sp.Dummy(f'm_{i}_{j}') for i in range(d) for j in range(d))
        z = sp.Dummy('inverse_determinant')
        unknowns = variables+(z,)
        M = sp.Matrix(d,d,variables)
        lifted = M
        for _ in range(1,ell): lifted = sp.kronecker_product(lifted,M)
        emit(f'Building polynomial equations in {len(unknowns)} variables')
        residual = lifted*h_prime-h*lifted
        # Deduplicate scalar multiples, and discard exact zero equations.
        equations = {}
        for entry in residual:
            polynomial = sp.Poly(entry,*unknowns,domain=K)
            if not polynomial.is_zero:
                polynomial = polynomial.monic()
                equations.setdefault(polynomial.as_expr(),polynomial)
        invertibility = sp.Poly(z*M.det()-1,*unknowns,domain=K)
        polynomials = tuple(equations.values())+(invertibility,)
        emit(f'Computing a Groebner basis for {len(polynomials)} equations')
        basis = sp.groebner(polynomials,*unknowns,domain=K,order='grevlex')
        # Hilbert's Nullstellensatz: no common complex zero iff 1 is in the
        # ideal. A nonzero constant in the reduced Groebner basis certifies this.
        exists = not any(P.is_ground and not P.is_zero for P in basis.polys)
    except (alg.BasePolynomialError,NotImplementedError,ZeroDivisionError) as exc:
        raise MPOError('Polynomial/number-field calculation failed; existence undecided') from exc
    emit('An invertible tensor-power intertwiner exists' if exists else
         'No invertible tensor-power intertwiner exists')
    return exists
