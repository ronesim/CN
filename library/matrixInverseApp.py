import numpy as np
from numpy import linalg as LA


def generate_V0(A):
    return A.transpose() / (np.max(np.sum(A, axis=0)) * np.max(np.sum(A, axis=1)))


def add_elem_diagonal(matrix, value):
    for row in range(len(matrix)):
        matrix[row, row] += value
    return np.matrix(matrix)


def remove_elem_diagonal(matrix, value):
    for row in range(len(matrix)):
        matrix[row, row] -= value
    return np.matrix(matrix)


def inverse_schulz_method(A, B, eps, kMax):
    # initialize V0
    V0 = generate_V0(A)
    # V(k+1) = V(k) *(2*I(n) - A*V(k))
    while kMax > 0:
        # compute V1 using V0
        C = B * V0
        # add 2 on C's diagonal and multiply with V0
        V1 = V0 * add_elem_diagonal(C, 2)
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def inverse_Li_method_I(A, B, eps, kMax):
    # initialize V0
    V0 = generate_V0(A)
    # V(k+1) = V(k) *(3*I(n) - A*V(k)(3*I(n) - A*V(k)))
    while kMax > 0:
        # compute V1 using V0
        C = B * V0
        aux_mat = add_elem_diagonal(C, 3)
        V1 = V0 * add_elem_diagonal(remove_elem_diagonal(C, 3) * aux_mat, 3)
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def inverse_Li_method_II(A, B, eps, kMax):
    # initialize V0
    V0 = generate_V0(A)
    # V(k+1) = (I(n) + 1/4 * (I(n) - V(k) *A)(3*I(n) - V(k) *A)^2) *V(k)
    while kMax > 0:
        # compute V1 using V0
        C = V0 * B
        aux_mat = add_elem_diagonal(C, 3) ** 2
        V1 = add_elem_diagonal(remove_elem_diagonal(C, 2) * aux_mat / 4, 1) * V0
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def get_norm(matrix, matrix_inv, matrix_size):
    return [LA.norm((matrix * matrix_inv - np.identity(matrix_size))), np.max(
        np.sum(matrix * matrix_inv - np.identity(matrix_size)), axis=0)]


def main_function(matrix, n, eps, kMax=10000):
    # compute B = -matrix: used in C = a*I(n) - A * V
    B = np.matrix((-1) * matrix)

    matrix_inv_schulz, iter = inverse_schulz_method(matrix, B, eps, kMax)
    matrix_inv_li, iter = inverse_Li_method_I(matrix, B, eps, kMax)
    matrix_inv_li_second, iter = inverse_Li_method_II(matrix, B, eps, kMax)

    return matrix_inv_schulz, kMax - iter, matrix_inv_li, kMax - iter, matrix_inv_li_second, kMax - iter


def non_square_matrix(matrix):
    row = matrix.shape[0]
    col = matrix.shape[1]
    if row > col:
        for index in range(col, row):
            matrix = np.delete(matrix, index, 0)
    else:
        for index in range(row, col):
            matrix = np.delete(matrix, index, 1)

    print(main_function(matrix, matrix.shape[0], eps=10 ** (-10))[0])


matrix = np.matrix([[7, 2, 1, 1],
                    [0, 3, -1, 1],
                    [-3, 4, -2, 1]])
non_square_matrix(matrix)
