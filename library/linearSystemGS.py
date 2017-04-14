import math

import numpy as np

from library.homework3 import app3


def solveSystemGaussSeidel(A, b, eps):
    XGS = np.zeros(A["n"])
    kMax = 10000  # number of iterations
    while kMax > 0:
        norm = 0.0
        # compute next vector
        indexCol = 0
        for index in range(0, A["n"]):
            # first and second sum
            firstSum = 0
            secondSum = 0
            if A["col"][indexCol] * (-1) == index + 1:
                indexCol += 1
                while A["col"][indexCol] > 0:
                    if A["col"][indexCol] < index + 1:
                        firstSum += A["val"][indexCol] * XGS[A["col"][indexCol] - 1]
                    elif A["col"][indexCol] > index + 1:
                        secondSum += A["val"][indexCol] * XGS[A["col"][indexCol] - 1]
                    indexCol += 1
            value = (b[index] - firstSum - secondSum) / A["d"][index]
            norm += (value - XGS[index]) ** 2
            XGS[index] = (b[index] - firstSum - secondSum) / A["d"][index]

        if math.sqrt(norm) < eps:
            return XGS, kMax
        if math.sqrt(norm) > 10 ** 8:
            return "no solution", kMax
        kMax -= 1

    return "no solution", kMax


def checkDiagonal(matrixA):
    return all(matrixA["d"])


def checkDiagonalDominance(matrixA):
    line = 0
    for index, elem in enumerate(matrixA["d"]):
        computedSum = 0
        if matrixA["col"][line] * (-1) == index + 1:
            line += 1
            while matrixA["col"][line] > 0:
                computedSum += matrixA["val"][line]
                line += 1
        if abs(elem) <= abs(computedSum):
            return False
    return True


def compute_norm(sparse_matrix_representation, XGS, b):
    bAprox = app3.vector_multiply(sparse_matrix_representation, XGS)
    return np.max(abs(np.array(bAprox) - np.array(b)))


def main_function(sparse_matrix_representation, b, eps):
    print("here")
    if checkDiagonal(sparse_matrix_representation):  # if all diagonal elem are non-null
        if checkDiagonalDominance(sparse_matrix_representation) is False:
            return "no solution", "Diagonal is dominant"
        print("here")
        XGS, iterations = solveSystemGaussSeidel(sparse_matrix_representation, b, eps)  # solve system
        if type(XGS) == str and XGS == "no solution":
            return "no solution", "Number of iterations: {}".format(10000 - iterations)
        else:
            return XGS, 10000 - iterations
    else:
        return "no solution", "Diagonal has null elements"
