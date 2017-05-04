import math

import numpy as np


def solve_system_Gauss_Seidel(A, b, eps):
    XGS = np.zeros(A["n"])
    k_max = 10000  # number of iterations
    while k_max > 0:
        norm = 0.0
        # compute next vector
        index_col = 0
        for index in range(0, A["n"]):
            # first and second sum
            first_sum = 0
            second_sum = 0
            if A["col"][index_col] * (-1) == index + 1:
                index_col += 1
                while A["col"][index_col] > 0:
                    if A["col"][index_col] < index + 1:
                        first_sum += A["val"][index_col] * XGS[A["col"][index_col] - 1]
                    elif A["col"][index_col] > index + 1:
                        second_sum += A["val"][index_col] * XGS[A["col"][index_col] - 1]
                    index_col += 1
            value = (b[index] - first_sum - second_sum) / A["d"][index]
            norm += (value - XGS[index]) ** 2
            XGS[index] = (b[index] - first_sum - second_sum) / A["d"][index]

        if math.sqrt(norm) < eps:
            return XGS, k_max
        if math.sqrt(norm) > 10 ** 8:
            return "no solution", k_max
        k_max -= 1

    return "no solution", k_max


def check_diagonal(matrix):
    return all(matrix["d"])


def check_diagonal_dominance(matrix):
    line = 0
    for index, elem in enumerate(matrix["d"]):
        computed_sum = 0
        if matrix["col"][line] * (-1) == index + 1:
            line += 1
            while matrix["col"][line] > 0:
                computed_sum += matrix["val"][line]
                line += 1
        if abs(elem) <= abs(computed_sum):
            return False
    return True


def compute_norm(sparse_matrix_representation, XGS, b):
    b_aprox = app3.vector_multiply(sparse_matrix_representation, XGS)
    return np.max(abs(np.array(b_aprox) - np.array(b)))


def main_function(sparse_matrix_representation, b, eps):
    if check_diagonal(sparse_matrix_representation):  # if all diagonal elem are non-null
        if check_diagonal_dominance(sparse_matrix_representation) is False:
            return "no solution", "Diagonal is dominant"
        XGS, iterations = solve_system_Gauss_Seidel(sparse_matrix_representation, b, eps)  # solve system
        if type(XGS) == str and XGS == "no solution":
            return "no solution", "Number of iterations: {}".format(10000 - iterations)
        else:
            return XGS, 10000 - iterations
    else:
        return "no solution", "Diagonal has null elements"
