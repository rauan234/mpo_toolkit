"""Tests of the local identity, rather than boundary-specific commutation."""
import unittest
from itertools import product
from unittest.mock import patch
import sympy as sp
from . import MPOTensor, local_pull_through, ResourceLimitError, zero_testing


def from_physical(U):
    return MPOTensor(U.rows,[[sp.Matrix([x]) for x in U.row(i)] for i in range(U.rows)])


class PullThroughTests(unittest.TestCase):
    def test_complex_invertible_one_site(self):
        U = sp.Matrix([[1,sp.I],[0,2]])
        h = sp.Matrix([[1,2],[3,4]])
        result = local_pull_through(from_physical(U),h,1)
        self.assertTrue(result.exists)
        self.assertTrue(result.unique)
        self.assertTrue(result.symbolically_certified)
        self.assertEqual(result.h_tilde,(U*h*U.inv()).applyfunc(sp.simplify))

    def test_two_site_order(self):
        U = sp.Matrix([[1,1],[0,2]])
        h = sp.Matrix([[0,1,2,3],[4,0,1,0],[0,2,1,3],[1,1,0,2]])
        result = local_pull_through(from_physical(U),h,2)
        UU = sp.kronecker_product(U,U)
        self.assertEqual(result.h_tilde,UU*h*UU.inv())

    def test_nontrivial_virtual_tensor_and_boundaries(self):
        S = sp.Matrix([[1,2],[0,1]])
        P = S*sp.diag(1,0)*S.inv()
        Q = sp.eye(2)-P
        T = MPOTensor(2,[[P,Q],[Q,P]])
        Z,Y = sp.diag(1,-1),sp.Matrix([[0,-sp.I],[sp.I,0]])
        self.assertFalse(local_pull_through(T,Z,1).exists)
        h = sp.kronecker_product(Z,Y)
        result = local_pull_through(T,h,2)
        self.assertTrue(result.exists)
        self.assertEqual(result.h_tilde,h)
        boundary = sp.Matrix([[2,sp.I],[3,4]])
        for L in (2,3,4):
            words = tuple(product(range(2),repeat=L))
            def coefficient(ii,jj):
                M = boundary
                for i,j in zip(ii,jj): M = M*T.A_list[i][j]
                return sp.trace(M)
            O = sp.Matrix([[coefficient(ii,jj) for jj in words] for ii in words])
            for x in range(L-1):
                local_h = sp.kronecker_product(sp.eye(2**x),h,sp.eye(2**(L-x-2)))
                self.assertEqual(O*local_h,local_h*O)

    def test_inconsistent_singular_physical_operator(self):
        T = from_physical(sp.diag(1,0))
        result = local_pull_through(T,sp.Matrix([[0,1],[1,0]]),1)
        self.assertFalse(result.exists)
        self.assertIsNone(result.h_tilde)
        self.assertFalse(result.unique)

    def test_nonunique_solution_and_kernel(self):
        P = sp.diag(1,0)
        h = sp.Matrix([[2,0],[3,4]])
        result = local_pull_through(from_physical(P),h,1)
        self.assertTrue(result.exists)
        self.assertFalse(result.unique)
        self.assertEqual(len(result.row_kernel),1)
        shifted = result.h_tilde+sp.Matrix([5,7])*result.row_kernel[0].T
        self.assertEqual(P*h,shifted*P)

    def test_zero_tensor(self):
        T = MPOTensor(2,[[sp.zeros(2) for _ in range(2)] for _ in range(2)])
        result = local_pull_through(T,sp.ones(4),2)
        self.assertTrue(result.exists)
        self.assertEqual(result.h_tilde,sp.zeros(4))
        self.assertEqual(len(result.row_kernel),4)

    def test_exact_local_is_stronger_than_external_annihilation(self):
        J = sp.Matrix([[0,1],[0,0]])
        T = MPOTensor(2,[[J,sp.zeros(2)],[sp.zeros(2),sp.zeros(2)]])
        # Every MPO of length >= 2 is zero for ALL boundaries, since J**2=0.
        # Nevertheless the one-site local identity cannot pull X through.
        self.assertEqual(J*J,sp.zeros(2))
        self.assertFalse(local_pull_through(T,sp.Matrix([[0,1],[1,0]]),1).exists)
        self.assertTrue(local_pull_through(T,sp.ones(4),2).exists)

    def test_transcendental_expression_backend(self):
        U = sp.diag(sp.pi,1)
        h = sp.Matrix([[0,1],[1,0]])
        self.assertEqual(local_pull_through(from_physical(U),h,1).h_tilde,U*h*U.inv())

    def test_validation_limits_and_heuristic_label(self):
        T = from_physical(sp.eye(2))
        for ell in (0,-1,True):
            with self.assertRaises(ValueError): local_pull_through(T,sp.eye(2),ell)
        with self.assertRaises(ValueError): local_pull_through(T,sp.eye(3),1)
        with self.assertRaises(ValueError): local_pull_through(T,sp.eye(2)*0.5,1)
        with self.assertRaises(ValueError): local_pull_through(T,sp.eye(2)*sp.Symbol('x'),1)
        with self.assertRaises(ResourceLimitError):
            local_pull_through(T,sp.eye(2),1,max_block_entries=1)
        with self.assertRaises(ResourceLimitError):
            local_pull_through(T,sp.eye(2),1,max_linear_rows=1)
        with patch.object(zero_testing,'REQUIRE_SYMBOLIC_CERTIFICATION',False):
            self.assertFalse(local_pull_through(T,sp.eye(2),1).symbolically_certified)

    def test_undecidable_pivot_raises(self):
        unknown = sp.Function('UnspecifiedConstant')(1)
        T = MPOTensor(1,[[sp.Matrix([unknown])]])
        with self.assertRaises(RuntimeError): local_pull_through(T,sp.eye(1),1)


if __name__=='__main__': unittest.main()
