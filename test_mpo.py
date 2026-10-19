"""Run: python3 -m unittest mpo_toolkit.test_mpo -v (from the parent directory)."""
import unittest
from unittest.mock import patch
from itertools import permutations
import sympy as sp
from .mpo import (MPOTensor, FusionMap, FSymbol, Limits, ResourceLimitError,
                  NotBlockDecomposableError, NotFusionCategoryError,
                  generate_fusion_category)
from . import zero_testing


def physical_matrix(M):
    return MPOTensor(M.rows, [[sp.Matrix([M[i,j]]) for j in range(M.cols)]
                              for i in range(M.rows)])


def standard_representation(degree, alternating=False):
    """Diagonal physical MPO encoding the standard permutation representation."""
    V = sp.zeros(degree, degree-1)
    for j in range(degree-1):
        V[j,j], V[j+1,j] = 1, -1
    W = (V.T * V).inv() * V.T
    matrices = []
    for perm in permutations(range(degree)):
        parity = sum(perm[i] > perm[j] for i in range(degree)
                     for j in range(i+1, degree)) % 2
        if alternating and parity: continue
        P = sp.zeros(degree)
        for j,i in enumerate(perm): P[i,j] = 1
        matrices.append(W * P * V)
    p = len(matrices)
    return MPOTensor(p, [[matrices[i] if i == j else sp.zeros(degree-1)
                         for j in range(p)] for i in range(p)])


class MPOTests(unittest.TestCase):
    def setUp(self):
        self.X = sp.Matrix([[0,1],[1,0]])
        self.Z = sp.diag(1,-1)
        self.I, self.O = sp.eye(2), sp.zeros(2)
        self.T = MPOTensor(2, [[self.I,self.X],[self.Z,self.O]])

    def test_input_immutability_and_validation(self):
        A = sp.eye(1)
        T = MPOTensor(1, [[A]])
        A[0,0] = 7
        self.assertEqual(T.A_list[0][0], sp.eye(1))
        with self.assertRaises(TypeError): T.A_list[0][0][0,0] = 8
        for entry in (0.2, sp.Symbol('x'), sp.oo):
            with self.assertRaises(ValueError): MPOTensor(1, [[sp.Matrix([entry])]])
        with self.assertRaises(ValueError): MPOTensor(2, [[sp.eye(1)]])

    def test_fixed_length_injectivity(self):
        self.assertTrue(self.T.test_injective())
        self.assertEqual(self.T.injectivity_length, 2)
        periodic = MPOTensor(2, [[self.X,self.Z],[self.O,self.O]])
        self.assertFalse(periodic.test_injective())
        with self.assertRaises(NotBlockDecomposableError): periodic.direct_sum_decomp()
        self.assertFalse(MPOTensor(1, [[self.I]]).test_injective())
        with self.assertRaises(ResourceLimitError):
            MPOTensor(2, self.T.A_list).test_injective(max_blocking=1)

    def test_decomposition_accepts_injective_after_blocking(self):
        # I, X, Z span only three dimensions at one site. Their length-two
        # products span M_2, so decomposition must accept this summand.
        D = self.T.direct_sum_decomp()
        self.assertEqual(len(D.tensors),1)
        block = D.tensors[0]
        local_span = sp.Matrix.hstack(*(A.reshape(4,1) for A in block.matrices))
        self.assertEqual(local_span.rank(),3)
        self.assertEqual(block.injectivity_length,2)
        self.assertTrue(block.test_injective())
        self.assertEqual(block.dim_phys,self.T.dim_phys)
        for i in range(2):
            for j in range(2):
                self.assertEqual(D.reconstruct(i,j),self.T.A_list[i][j])

    def test_compose_order(self):
        other = MPOTensor(2, [[self.X,self.O],[self.I,self.Z]])
        C = self.T.compose(other)
        for i in range(2):
            for j in range(2):
                self.assertEqual(C.A_list[i][j], sum((sp.kronecker_product(
                    other.A_list[i][k],self.T.A_list[k][j]) for k in range(2)),sp.zeros(4)))
        for length in (1,2):
            self.assertEqual(C.periodic_operator(length),
                             other.periodic_operator(length)*self.T.periodic_operator(length))
        a,b = physical_matrix(self.X), physical_matrix(self.Z)
        self.assertEqual(a.compose(b).periodic_operator(1), self.Z*self.X)
        self.assertNotEqual(a.compose(b).periodic_operator(1), b.compose(a).periodic_operator(1))

    def test_transpose_physical(self):
        T = MPOTensor(2, [[sp.Matrix([[1,sp.I],[2,3]]),
                           sp.Matrix([[0,2+sp.I],[1,1]])],
                          [sp.Matrix([[sp.I,3],[0,2]]),
                           sp.Matrix([[2,0],[1-sp.I,sp.I]])]])
        before = T.A_list
        transposed = T.transpose_physical()
        self.assertIsNot(transposed,T)
        self.assertEqual((transposed.dim_phys,transposed.n),(T.dim_phys,T.n))
        self.assertIs(transposed.limits,T.limits)
        self.assertEqual(T.A_list,before)
        for i in range(T.dim_phys):
            for j in range(T.dim_phys):
                self.assertEqual(transposed.A_list[i][j],T.A_list[j][i])
        self.assertNotEqual(transposed.A_list[0][1],T.A_list[1][0].T)
        self.assertNotEqual(transposed.A_list, T.conjugate().A_list)
        self.assertEqual(transposed.transpose_physical().A_list,T.A_list)
        for length in (1,2,3):
            self.assertEqual(transposed.periodic_operator(length),T.periodic_operator(length).T)

    def test_conjugate_is_physical_adjoint(self):
        # Complex, nonsymmetric virtual matrices distinguish entrywise
        # conjugation from a virtual adjoint; unequal off-diagonal physical
        # entries distinguish a physical adjoint from plain conjugation.
        T = MPOTensor(2, [[sp.Matrix([[1,sp.I],[2,3-sp.I]]),
                           sp.Matrix([[0,2+sp.I],[1,1]])],
                          [sp.Matrix([[sp.I,3],[0,2]]),
                           sp.Matrix([[2,0],[1-sp.I,sp.I]])]])
        before = T.A_list
        C = T.conjugate()
        self.assertIsNot(C,T)
        self.assertEqual((C.dim_phys,C.n),(T.dim_phys,T.n))
        self.assertIs(C.limits,T.limits)
        self.assertEqual(T.A_list,before)
        self.assertEqual(C.conjugate().A_list,T.A_list)
        self.assertNotEqual(C.A_list[0][1],T.A_list[1][0].H)
        for length in (1,2,3):
            self.assertEqual(C.periodic_operator(length),T.periodic_operator(length).H)
        # Adjoint reverses physical composition order.
        lhs = T.compose(self.T).conjugate()
        rhs = self.T.conjugate().compose(T.conjugate())
        self.assertEqual(lhs.periodic_operator(2),rhs.periodic_operator(2))

    def test_nonunitary_gauge(self):
        S = sp.Matrix([[1,2],[0,1]])
        other = MPOTensor(2, [[S*A*S.inv() for A in row] for row in self.T.A_list])
        H = self.T.gauge_to(other)
        self.assertIsNotNone(H)
        for A,B in zip(self.T.matrices, other.matrices): self.assertEqual(B*H,H*A)
        self.assertIsNone(physical_matrix(self.X).gauge_to(physical_matrix(-self.X)))

    def test_multiplicities_and_oblique_zero_sector(self):
        S = sp.eye(6)
        S[0,5], S[3,4], S[1,2] = 2, 3, -1
        matrices = [sp.diag(A,A,sp.Matrix([int(i==0)]),sp.zeros(1))
                    for i,A in enumerate(self.T.matrices)]
        matrices = [S*A*S.inv() for A in matrices]
        T = MPOTensor(2,[matrices[:2],matrices[2:]])
        D = T.direct_sum_decomp()
        self.assertEqual(sorted(zip((t.n for t in D.tensors),D.multiplicities)),[(1,1),(2,2)])
        self.assertEqual(D.zero_dimension,1)
        self.assertEqual(D.S*D.S_inverse,sp.eye(6))
        tensors,multiplicities,maps = D
        self.assertEqual(multiplicities,list(D.multiplicities))
        complete = D.zero_inclusion*D.zero_projection
        for a,channels in enumerate(maps):
            for k,F in enumerate(channels):
                complete += F.X*F.X_plus
                for b,others in enumerate(maps):
                    for l,G in enumerate(others):
                        expected = sp.eye(tensors[a].n) if (a,k)==(b,l) else sp.zeros(tensors[a].n,tensors[b].n)
                        self.assertEqual(F.X_plus*G.X,expected)
        self.assertEqual(complete,sp.eye(6))
        for i in range(2):
            for j in range(2): self.assertEqual(D.reconstruct(i,j),T.A_list[i][j])
        self.assertTrue(any(F.X_plus != F.X.H for channels in maps for F in channels))
        with self.assertRaises(NotBlockDecomposableError): T.direct_sum_decomp(allow_zero=False)

    def test_zero_and_radical(self):
        D = MPOTensor(1,[[sp.zeros(3)]]).direct_sum_decomp()
        self.assertEqual(D.tensors,())
        self.assertEqual(D.zero_dimension,3)
        self.assertEqual(D.reconstruct(0,0),sp.zeros(3))
        with self.assertRaises(NotBlockDecomposableError):
            MPOTensor(1,[[sp.Matrix([[0,1],[0,0]])]]).direct_sum_decomp()

    def test_algebraic_and_transcendental_constants(self):
        for x in (sp.sqrt(2)+sp.I,sp.pi):
            T = MPOTensor(1,[[sp.Matrix([x])]])
            self.assertTrue(T.test_injective())
            self.assertEqual(T.direct_sum_decomp().reconstruct(0,0),sp.Matrix([x]))

    def test_rootof_rational_function_zero(self):
        x = sp.Symbol('x')
        root = sp.CRootOf(x*x+x+1,0)
        zero = (1+root+root**2)/(4*root+4*root**2)
        T = MPOTensor(1,[[sp.Matrix([zero])]])
        self.assertEqual(T.A_list[0][0],sp.zeros(1))
        self.assertFalse(T.test_injective())


class CategoryTests(unittest.TestCase):
    def test_a4_fusion_multiplicity(self):
        C = generate_fusion_category([standard_representation(4,alternating=True)])
        self.assertEqual(C.num_simples,4)
        self.assertEqual(C.N[0][0],(2,1,1,1))
        self.assertEqual(C.F_symbols[0,0,0,0].matrix.shape,(7,7))
        self.assertEqual(len(C.fusion_tensors[0,0,0]),2)
        self.assertTrue(C.pentagon_verified)
        self.assertEqual(C.duals[C.duals[0]],0)
        self.assertTrue(C.symbolically_certified)

    def test_z2_and_corrupt_pentagon(self):
        C = generate_fusion_category([physical_matrix(sp.diag(1,-1))])
        self.assertEqual(C.num_simples,2)
        self.assertTrue(C.pentagon_verified)

        a = 1-C.unit
        self.assertEqual(C.fusion(a,a),{C.unit:1})
        self.assertEqual(C.duals[a],a)
        block = C.F_symbols[a,a,a,a]
        C.F_symbols[a,a,a,a] = FSymbol(block.left_paths,block.right_paths,2*block.matrix)
        with self.assertRaises(RuntimeError): C.verify_pentagon()
        self.assertFalse(C.pentagon_verified)

    def test_s3_nontrivial_associator(self):
        C = generate_fusion_category([standard_representation(3)])
        self.assertEqual(sorted(t.n for t in C.simples),[1,1,2])
        self.assertEqual(C.N[0][0],(1,1,1))
        self.assertTrue(C.pentagon_verified)
        self.assertEqual(C.F_symbols[0,0,0,0].matrix.shape,(3,3))
        self.assertNotEqual(C.F_symbols[0,0,0,0].matrix,sp.eye(3))
        # Change each fusion basis by a different scalar. Unitors and evaluation
        # maps must compensate for this freely chosen normalization.
        for (a,b,c),channels in list(C.fusion_tensors.items()):
            C.fusion_tensors[a,b,c] = tuple(FusionMap((2+a+2*b+3*c+k)*F.X,
                F.X_plus/(2+a+2*b+3*c+k)) for k,F in enumerate(channels))
        C._compute_F_symbols()
        self.assertTrue(C.verify_pentagon())
        C._unit_and_duals()
        self.assertNotEqual(C.left_unitors,(1,1,1))

    def test_finite_nonrigid_monoid_rejected(self):
        with self.assertRaises(NotFusionCategoryError):
            generate_fusion_category([physical_matrix(sp.eye(2)),physical_matrix(sp.diag(1,0))])

    def test_infinite_candidate_hits_budget(self):
        with self.assertRaises(ResourceLimitError):
            generate_fusion_category([physical_matrix(sp.Matrix([2]))],max_tensors=4)

    def test_noncommutative_fusion_order(self):
        A = sp.Matrix([[0,1,0],[1,0,0],[0,0,1]])
        B = sp.Matrix([[1,0,0],[0,0,1],[0,1,0]])
        C = generate_fusion_category([physical_matrix(A),physical_matrix(B)])
        self.assertEqual(C.num_simples,6)
        self.assertNotEqual(C.fusion(0,1),C.fusion(1,0))
        self.assertTrue(C.pentagon_verified)


class ZeroTestingTests(unittest.TestCase):
    def test_numerical_thresholds_and_gray_zone(self):
        self.assertIs(zero_testing.numerical_hint(sp.Rational(1,10**60)),True)
        self.assertIs(zero_testing.numerical_hint(sp.Rational(1,10**7)),False)
        self.assertIsNone(zero_testing.numerical_hint(sp.Rational(1,10**20)))
        # Exact knowledge takes precedence over a numerical hint.
        self.assertFalse(zero_testing.is_zero(sp.Rational(1,10**60)))
        self.assertTrue(zero_testing.equal(sp.sqrt(2)**2,2))

    def test_unconfirmed_hint_requires_runtime_error(self):
        unknown = sp.Function('UnknownConstant')(1)
        for hint in (True,False):
            with patch.object(zero_testing,'numerical_hint',return_value=hint):
                with self.assertRaises(RuntimeError): zero_testing.is_zero(unknown)
                with patch.object(zero_testing,'REQUIRE_SYMBOLIC_CERTIFICATION',False):
                    self.assertIs(zero_testing.is_zero(unknown),hint)
        with patch.object(zero_testing,'numerical_hint',return_value=None):
            with patch.object(zero_testing,'REQUIRE_SYMBOLIC_CERTIFICATION',False):
                with self.assertRaises(RuntimeError): zero_testing.is_zero(unknown)

    def test_policy_invalidates_caches_and_marks_results(self):
        T = physical_matrix(sp.Matrix([1]))
        D = T.direct_sum_decomp()
        self.assertTrue(D.symbolically_certified)
        self.assertTrue(T.test_injective())
        self.assertTrue(T.injectivity_symbolically_certified)
        with patch.object(zero_testing,'REQUIRE_SYMBOLIC_CERTIFICATION',False):
            D2 = T.direct_sum_decomp()
            self.assertIsNot(D2,D)
            self.assertFalse(D2.symbolically_certified)
            self.assertFalse(T.injectivity_symbolically_certified)
            C = generate_fusion_category([T])
            self.assertFalse(C.symbolically_certified)
        self.assertTrue(T.direct_sum_decomp().symbolically_certified)
        with patch.object(zero_testing,'EQUALITY_THRESHOLD',1):
            with self.assertRaises(ValueError): T.test_injective()


if __name__ == '__main__': unittest.main()
