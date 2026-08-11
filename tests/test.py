import numpy as np
from KroneckerFactorization.kronecker_factorization import full_kronecker_factorization

def test_kronecker_product():
    for n in range(2,20):
        matrix = np.random.rand((n,n))
        kron_tree = full_kronecker_factorization(matrix=matrix)
        assert np.allclose(matrix,kron_tree.get_matrix())
