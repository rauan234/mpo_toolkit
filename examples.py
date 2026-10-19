"""Small executable examples: python3 -m mpo_toolkit.examples."""
import sympy as sp
from . import MPOTensor, generate_fusion_category
from .test_mpo import standard_representation


def main():
    X, Z = sp.Matrix([[0,1],[1,0]]), sp.diag(1,-1)
    base = MPOTensor(2, [[sp.eye(2),X],[Z,sp.zeros(2)]])
    print('Injective:',base.test_injective(),'blocking length:',base.injectivity_length)

    # Two equivalent copies and a zero sector, in a nonorthogonal gauge.
    S = sp.eye(5)
    S[0,4] = 2
    array = [[S*sp.diag(A,A,sp.zeros(1))*S.inv() for A in row] for row in base.A_list]
    tensor = MPOTensor(2,array)
    D = tensor.direct_sum_decomp()
    print('Block dimensions:',[T.n for T in D.tensors])
    print('Multiplicities:',D.multiplicities,'zero-sector dimension:',D.zero_dimension)
    assert all(D.reconstruct(i,j)==tensor.A_list[i][j] for i in range(2) for j in range(2))

    # Irreducible representations encoded along diagonal physical indices.
    C = generate_fusion_category([standard_representation(3)])
    print('Rep(S_3) dimensions:',[T.n for T in C.simples])
    print('Fusion coefficients:',C.N)
    print('Unit:',C.unit,'duals:',C.duals)
    print('F for three standard representations with standard output:')
    sp.pprint(C.F_symbols[0,0,0,0].matrix)
    print('Pentagon verified:',C.pentagon_verified)


if __name__=='__main__': main()
