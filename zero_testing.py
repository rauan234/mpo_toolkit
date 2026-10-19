"""Configurable high-precision hints for scalar zero/equality decisions.

Change these globals on this module, not on a locally imported copy. Numerical
agreement is never a proof, even at 160 decimal digits. The default requires
symbolic confirmation; disabling it is explicitly heuristic.
"""
import sympy as sp
from sympy.core.evalf import PrecisionExhausted

DIFFERENCE_THRESHOLD = sp.Rational(1, 10**8)
EQUALITY_THRESHOLD = sp.Rational(1, 10**50)
REQUIRE_SYMBOLIC_CERTIFICATION = True


class UndecidableZeroError(RuntimeError):
    """Neither the active exact nor numerical policy can decide a scalar test."""


def policy_key():
    difference, equality = sp.sympify(DIFFERENCE_THRESHOLD), sp.sympify(EQUALITY_THRESHOLD)
    if (difference.is_positive is not True or equality.is_positive is not True
            or difference.is_finite is not True or equality.is_finite is not True
            or (difference > equality) is not sp.S.true):
        raise ValueError('Require finite thresholds 0 < EQUALITY_THRESHOLD < DIFFERENCE_THRESHOLD')
    if type(REQUIRE_SYMBOLIC_CERTIFICATION) is not bool:
        raise ValueError('REQUIRE_SYMBOLIC_CERTIFICATION must be a bool')
    return difference, equality, REQUIRE_SYMBOLIC_CERTIFICATION


def numerical_hint(expression):
    """True: likely zero, False: likely nonzero, None: no stable indication.

    Two precision levels must agree. Thresholds are absolute, so scaling matters.
    The interval between thresholds is deliberately inconclusive.
    """
    difference, equality, _ = policy_key()
    x = sp.sympify(expression)
    if x.free_symbols: return None
    # Leave at least 50 decimal guard digits beyond the smaller threshold.
    digits = max(100, int(sp.ceiling(-sp.log(equality, 10)).evalf()) + 50)
    hints = []
    for precision in (digits, digits+60):
        try:
            value = x.evalf(precision, maxn=4*precision, strict=True)
            magnitude = sp.Abs(value).evalf(precision)
            if magnitude.is_number is not True or magnitude.is_finite is not True:
                return None
            if (magnitude < equality) is sp.S.true: hints.append(True)
            elif (magnitude > difference) is sp.S.true: hints.append(False)
            else: return None
        except (PrecisionExhausted, ValueError, TypeError, NotImplementedError,
                OverflowError, ZeroDivisionError):
            return None
    return hints[0] if hints[0] == hints[1] else None


def is_zero(expression, *, simplifier=None):
    """Return a bool, or raise RuntimeError if the policy cannot decide.

    Cheap exact facts take priority. Otherwise numerical evaluation guides the
    test. In strict mode it must be resolved symbolically, with an independently
    established exact answer taking priority over a contradictory numerical hint.
    """
    _, _, strict = policy_key()
    x = sp.sympify(expression)
    if x.free_symbols or x.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise UndecidableZeroError('Zero tests require finite complex constants without parameters')
    if x == 0: return True
    if x.has(sp.CRootOf):
        # Canonical field reduction is much cheaper than high-precision
        # cancellation testing for rational functions of exact polynomial roots.
        from .semisimple_decomposition_GPT import _reduce_root_rational_function
        try:
            reduced = _reduce_root_rational_function(x)
            # The field representation is canonical: a nonzero coordinate
            # vector is already an exact nonzero certificate.
            return reduced == 0
        except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
            pass
    answer = x.is_zero
    if answer is not None: return bool(answer)
    hint = numerical_hint(x)
    if not strict and hint is not None:
        return hint
    if simplifier is None:
        # Delayed import avoids a cycle; use the same exact root reductions as
        # the algebra backend, also for public calls to this function.
        from .semisimple_decomposition_GPT import _simp
        simplifier = _simp
    reduced = simplifier(x)
    if reduced == 0: return True
    answer = reduced.is_zero
    if answer is not None: return bool(answer)
    if reduced.is_number is True and reduced.is_algebraic is True:
        try:
            coordinates = sp.polys.numberfields.to_number_field(reduced).coeffs()
            return all(c == 0 for c in coordinates)
        except (NotImplementedError, ValueError, TypeError, sp.PolynomialError):
            pass
    indication = 'zero' if hint is True else 'nonzero' if hint is False else 'inconclusive'
    raise UndecidableZeroError(
        f'Cannot decide symbolic zero test; numerical indication is {indication}: {reduced}')


def equal(left, right):
    """Scalar equality under the active policy; undecidable tests raise RuntimeError."""
    return is_zero(sp.sympify(left)-sp.sympify(right))
