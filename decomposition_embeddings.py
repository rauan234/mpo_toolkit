"""Recover physical inclusion/projection maps from decompose_algebra's result."""
from functools import lru_cache
import sympy as sp
from .semisimple_decomposition_GPT import _simp, _zero


def extract_embeddings(D):
    """Return s, d, m, r, S, phi, F, G, F_zero, and G_zero as a dictionary.

    phi = S**(-1). F[a][k] is phi**(-1) iota_a varphi_(a,k);
    G[a][k] is epsilon_(a,k) pi_a phi. Python indices start at zero.
    Columns of S use tensor order e_i tensor e_k (k varies fastest), followed
    by a basis of the zero-action space. The maps match D.to_blocks exactly.
    Uses only public fields/methods of D; does not assume adjoint closure.
    """
    @lru_cache(maxsize=16384)
    def clean(x):
        return _simp(x)

    def zero(x):
        answer = _zero(x)
        if answer is not None:
            return answer
        raise RuntimeError(f"Cannot certify a required zero test: {x}")

    def immutable(A):
        return sp.ImmutableMatrix(A).applyfunc(clean)

    def require_equal(A, B, label):
        if A.shape != B.shape or not all(zero(x) for x in A - B):
            raise RuntimeError(f"Exact verification failed: {label}")

    n = D.identity.rows
    sizes = tuple(D.block_sizes)
    multiplicities = tuple(b.multiplicity for b in D.blocks)
    sectors = []
    for a, (d, multiplicity) in enumerate(zip(sizes, multiplicities)):
        def lift(i):
            images = [sp.zeros(size) for size in sizes]
            images[a][i, 0] = 1
            return immutable(D.from_blocks(images))

        E00 = lift(0)
        _, pivots = E00.rref(iszerofunc=zero, simplify=clean)
        if len(pivots) != multiplicity:
            raise RuntimeError("The lifted E_00 has the wrong rank.")
        V = E00[:, list(pivots)]  # Basis v_k of the multiplicity space.
        # The i-th group is (E_i0 v_0, ..., E_i0 v_(m-1)).
        sectors.append(immutable(sp.Matrix.hstack(
            V, *(lift(i) * V for i in range(1, d)))))

    # The algebra acts as zero exactly on ker(D.identity).
    R, pivots = sp.Matrix(D.identity).rref(iszerofunc=zero, simplify=clean)
    kernel = []
    for j in range(n):
        if j not in pivots:
            v = sp.zeros(n, 1)
            v[j] = 1
            for i, pivot in enumerate(pivots):
                v[pivot] = -R[i, j]
            kernel.append(v)
    r = len(kernel)
    if sum(d * m for d, m in zip(sizes, multiplicities)) + r != n:
        raise RuntimeError("Inconsistent representation dimensions.")
    S = immutable(sp.Matrix.hstack(*(sectors + kernel)))
    phi = immutable(S.inv(method="GE", iszerofunc=zero))
    require_equal(phi * S, sp.eye(n), "phi S = I")
    require_equal(S * phi, sp.eye(n), "S phi = I")

    F, G, offset = [], [], 0
    for d, multiplicity in zip(sizes, multiplicities):
        indices = [tuple(offset + i * multiplicity + k for i in range(d))
                   for k in range(multiplicity)]
        F.append(tuple(immutable(S[:, list(idx)]) for idx in indices))
        G.append(tuple(immutable(phi[list(idx), :]) for idx in indices))
        offset += d * multiplicity

    # Certify the simultaneous representation on the generators. Linearity
    # and products extend these intertwining equations to their whole algebra.
    for A, images in zip(D.generators, D.generator_images):
        pieces = [sp.kronecker_product(M, sp.eye(m))
                  for M, m in zip(images, multiplicities)]
        if r:
            pieces.append(sp.zeros(r))
        target = sp.diag(*pieces)
        require_equal(A * S, S * target, "generator intertwining")
    expected_identity = sp.diag(sp.eye(n - r), sp.zeros(r))
    require_equal(D.identity * S, S * expected_identity, "algebra identity")

    return dict(s=len(sizes), d=sizes, m=multiplicities, r=r, S=S, phi=phi,
                F=tuple(F), G=tuple(G),
                F_zero=immutable(S[:, offset:]), G_zero=immutable(phi[offset:, :]))
