import unittest
from unittest.mock import patch
import sympy as sp
from . import tensor_power_intertwiner_exists as exists
from . import ResourceLimitError, zero_testing


class TensorPowerGaugeTests(unittest.TestCase):
    def test_known_invertible_two_site_gauge(self):
        M = sp.Matrix([[1,1],[0,2]])
        K = sp.kronecker_product(M,M)
        hp = sp.Matrix([[0,1,2,3],[4,0,1,0],[0,2,1,3],[1,1,0,2]])
        self.assertTrue(exists(K*hp*K.inv(),hp,2,2))

    def test_similarity_is_not_sufficient(self):
        # These are conjugate by a permutation swapping the two sites, but
        # cannot be conjugate by an identical matrix on both sites.
        h,hp = sp.diag(0,2,1,3),sp.diag(0,1,2,3)
        self.assertEqual(h.charpoly().as_expr(),hp.charpoly().as_expr())
        self.assertFalse(exists(h,hp,2,2))

    def test_complex_solutions_outside_coefficient_field(self):
        J = sp.Matrix([[0,1],[0,0]])
        hp = sp.kronecker_product(J,J)
        # M=diag(i,1) is a witness although the inputs are rational.
        M = sp.diag(sp.I,1)
        K = sp.kronecker_product(M,M)
        self.assertEqual(K*hp,-hp*K)
        self.assertTrue(exists(-hp,hp,2,2))

    def test_singular_intertwiners_do_not_count(self):
        h,hp = sp.diag(0,1),sp.zeros(2)
        singular = sp.diag(1,0)
        self.assertEqual(singular*hp,h*singular)
        self.assertFalse(exists(h,hp,2,1))

    def test_algebraic_coefficients(self):
        hp = sp.diag(sp.sqrt(2),-sp.sqrt(2))
        M = sp.Matrix([[1,sp.I],[0,1]])
        self.assertTrue(exists(M*hp*M.inv(),hp,2,1))

    def test_one_dimensional_and_identical(self):
        self.assertFalse(exists(sp.Matrix([1]),sp.Matrix([2]),1,3))
        self.assertTrue(exists(sp.zeros(4),sp.zeros(4),2,2))
        self.assertTrue(exists(sp.eye(2)*sp.pi,sp.eye(2)*sp.pi,2,1))

    def test_validation_and_undecidable_field(self):
        with self.assertRaises(ValueError): exists(sp.eye(3),sp.eye(3),2,1)
        with self.assertRaises(ValueError): exists(sp.eye(2),sp.eye(2),2,0)
        with self.assertRaises(ValueError): exists(sp.eye(2)*0.5,sp.eye(2),2,1)
        with self.assertRaises(ValueError): exists(sp.eye(2)*sp.Symbol('a'),sp.eye(2),2,1)
        with self.assertRaises(RuntimeError): exists(sp.diag(sp.pi,1),sp.eye(2),2,1)
        with self.assertRaises(ResourceLimitError):
            exists(sp.eye(2),sp.zeros(2),2,1,max_variables=4)
        with self.assertRaises(ResourceLimitError):
            exists(sp.eye(2),sp.zeros(2),2,1,max_matrix_entries=3)

    def test_numerical_policy_does_not_change_decision(self):
        with patch.object(zero_testing,'REQUIRE_SYMBOLIC_CERTIFICATION',False):
            self.assertFalse(exists(sp.diag(0,2,1,3),sp.diag(0,1,2,3),2,2))


if __name__=='__main__': unittest.main()
