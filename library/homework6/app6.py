import numpy as np
from scipy import stats
from scipy.sparse import random

from library.homework3 import app3
from library.util import read_sparse_matrix_from_file


def generate_random_sparse_matrix(rank):
    rvs = stats.poisson(50, loc=0).rvs
    sparse_matrix = random(rank, rank, density=0.40, data_rvs=rvs)

    d = np.random.randn(rank) + 50
    matrix = {}
    NNMAX = 0
    sparse_matrix = sparse_matrix + sparse_matrix.transpose()
    cols = sparse_matrix.nonzero()[1]
    rows = sparse_matrix.nonzero()[0]
    sparse_matrix = sparse_matrix.tocsr()
    for index in range(0, len(cols)):
        if cols[index] == rows[index]:
            d[cols[index]] += sparse_matrix[rows[index], cols[index]]
        else:
            NNMAX += 1
            if not (rows[index], cols[index]) in matrix:
                matrix[(rows[index], cols[index])] = sparse_matrix[rows[index], cols[index]]
            else:
                matrix[(rows[index], cols[index])] += sparse_matrix[rows[index], cols[index]]
    val = [0]
    col = [-1]
    row = 0
    for item in sorted(matrix.keys()):
        while item[0] != row:
            row += 1
            col.append(-(row + 1))
            val.append(0)
        col.append(item[1] + 1)
        val.append(matrix[item])

    col.append(-(row + 2))
    val.append(0)
    return {'n': rank, 'nn': NNMAX, 'd': d, 'val': val, 'col': col}


def matrix_is_symmetric(matrix, eps):
    row = 0
    saved_values = {}
    for index in range(0, len(matrix['col'])):
        if (matrix['col'][index] < 0):
            row = -matrix['col'][index]
        else:
            col = matrix['col'][index]
            if row < col:
                saved_values[(row, col)] = matrix['val'][index]
            else:
                if ((col, row) in saved_values) == False or abs(saved_values[(col, row)] - matrix['val'][index]) > eps:
                    return False
    return True


def scalar_product(vector1, vector2):
    ans = 0
    for index in range(0, len(vector1)):
        ans += vector1[index] * vector2[index]
    return ans


def generate_eigenvalues_eigenvectors(matrix, eps):
    k_max = 1000000
    x = np.random.randn(matrix['n'])
    v = x / np.linalg.norm(x)
    w = np.array(app3.vector_multiply(matrix, v))
    alpha = scalar_product(w, v)
    k = 0
    while (k <= k_max and np.linalg.norm(w - alpha * v) > matrix['n'] * eps):
        v = w / np.linalg.norm(w)
        w = np.array(app3.vector_multiply(matrix, v))
        alpha = scalar_product(w, v)
        k += 1

    if k > k_max:
        return "Invalid", "Info: Cannot compute eigenvalues"
    return alpha, v


def compute_matrix(u, v, si):
    ans = np.zeros((len(u), len(v)))
    for row in range(0, len(u)):
        for col in range(0, len(v)):
            ans[row, col] = u[row] * v[col] * si
    return ans


def SVD_get_info(matrix, s_index):
    """
    Get info about a matrix using Singular Value Decomposition
    :param matrix: a simple matrix
    :return:  singular values,
              matrix rang,
              condition number (maxSingularValue / minSingulargValue),
              
    """
    U, S, V = np.linalg.svd(matrix)
    print("U: ", U)
    print("S: ", S)
    print("V: ", V)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    s_matrix = np.zeros((rows, cols))

    if rows <= cols:
        value = rows
    else:
        value = cols
    for index in range(0, value):
        s_matrix[index, index] = S[index]

    A_s = np.zeros((rows, cols))
    for index in range(0, s_index):
        A_s = A_s + compute_matrix(U[:, index], V[:, index], S[index])

    return S, len(S.nonzero()[0]), max(S) / min(S[x] for x in range(0, len(S)) if S[x] > 0), np.max(
        np.sum(matrix - U * s_matrix * V, axis=1)), A_s, np.max(np.sum(matrix - A_s, axis=1))


def main_function(p, n):
    eps = 10 ** (-10)
    if (p == n):
        readMatrix = read_sparse_matrix_from_file("inputFileApp6.txt", False)
        randomMatrix = generate_random_sparse_matrix(n)
        if not matrix_is_symmetric(readMatrix, eps):
            return "Invalid matrix: not symmetric"

        result = []
        result.append(generate_eigenvalues_eigenvectors(readMatrix, eps))
        result.append(generate_eigenvalues_eigenvectors(randomMatrix, eps))
        return result


# result = main_function(5, 5)
# print(result)
matrix = np.matrix([[3, 2, 2],
                    [2, 3, -2],
                    [3, 2, 2],
                    [2, 3, -2]])
print(SVD_get_info(matrix, 3))
