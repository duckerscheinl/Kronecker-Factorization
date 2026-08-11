import numpy as np
import sympy


class KroneckerTreeNode:

    def __init__(self, id, parent_id, matrix):
        self.id = id
        self.parent_id = parent_id
        self.matrix = matrix


class KroneckerTree:

    def __init__(self):
        self.leaf_ids = list()
        self.nodes = dict()
        self.n = 0

    def add_leaf(self, leaf: KroneckerTreeNode):
        self.leaf_ids.append(leaf.id)
        self.nodes[leaf.id] = leaf
        self.n += 1

    def add_inner_node(self, node: KroneckerTreeNode):
        self.nodes[node.id] = node
        self.n += 1

    def get_matrix(self):

        matrix = 0
        for leaf_id in self.leaf_ids:
            node = self.nodes[leaf_id]
            leaf_matrix = node.matrix
            parent_id = node.parent_id
            while parent_id != "":
                node = self.nodes[parent_id]
                leaf_matrix = np.kron(node.matrix,leaf_matrix)
                parent_id = node.parent_id
            matrix += leaf_matrix

        return matrix

    def print_terms(self):

        for leaf_id in self.leaf_ids:
            print("---------------------------")
            node = self.nodes[leaf_id]
            with np.printoptions(precision=1):
                print(node.matrix)
            parent_id = node.parent_id
            while parent_id != "":
                node = self.nodes[parent_id]
                with np.printoptions(precision=1):
                    print(node.matrix)
                parent_id = node.parent_id



def svd_trunc(matrix, tol=1e-14):

    u,s,vh = np.linalg.svd(matrix, full_matrices=False)
    idx = np.abs(s)>tol
    u = u[:,idx]
    s = s[idx]
    vh = vh[idx,:]
    return u,s,vh 


def kronecker_factorization(matrix, d_A, d_B):

    tensor = np.reshape(matrix, (d_A, d_B, d_A, d_B))
    tensor = np.transpose(tensor, (0,2,1,3))
    lookup = np.reshape(tensor, (d_A**2,d_B**2))
    u,s,vh = svd_trunc(lookup)
    chi = s.shape[0]
    u = u
    vh = np.diag(s)@vh
    return np.reshape(u,(d_A,d_A,chi)), np.reshape(vh, (chi,d_B,d_B))


def recursive_kronecker_factorization(matrix, parent_id, tree):

    d = matrix.shape[0]
    d_A = sympy.primefactors(d)[0]

    if d == d_A:
        leaf = KroneckerTreeNode(id=tree.n, parent_id=parent_id, matrix=matrix)
        tree.add_leaf(leaf)
        return 

    factors, rest = kronecker_factorization(matrix=matrix, d_A=d_A, d_B=d//d_A)

    for i in range(factors.shape[2]):
        node = KroneckerTreeNode(id=tree.n, parent_id=parent_id, matrix=factors[:,:,i])
        tree.add_inner_node(node)
        recursive_kronecker_factorization(matrix=rest[i,:,:], parent_id=node.id, tree=tree)
    

def full_kronecker_factorization(matrix) -> KroneckerTree:

    tree = KroneckerTree()
    root = KroneckerTreeNode(id=0, parent_id="", matrix=1.0)
    tree.add_inner_node(root)
    recursive_kronecker_factorization(matrix=matrix, parent_id=0, tree=tree)

    return tree