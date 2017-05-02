import math

import numpy as np
import scipy.linalg


def solve_Choleski(A, n, eps):
    # Compute A = LDL^T Choleski decomposition
    D = np.zeros(n)
    for index in range(0, n):
        # compute D[index] = A[index][index] - sum(D[k] * L[index][k]^2) k=1...index-1
        second_sum = 0
        for k in range(0, index):
            second_sum += D[k] * A.item((index, k)) * A.item((index, k))
        D[index] = A.item((index, index)) - second_sum

        # compute L[i][index] = (a[i][index] - sum(D[k]*L[i][k]*L[p][k])) / D[index] i = index+1...n
        for i in range(index + 1, n):
            second_sum = 0
            for k in range(0, index):
                second_sum += D[k] * A.item((i, k)) * A.item((index, k))  # L is computed using A matrix
            if abs(D[index]) > eps:
                A[i, index] = (A.item((i, index)) - second_sum) / D[index]
            else:
                raise Exception("Divide by zero, Choleski decomposition cannot be computed")
    return A, D


def determinant(diagonal):
    return diagonal.prod()


def solve_system(A, b, diagonal, eps):
    # solve Lz = b
    NMAX = len(b)  # size
    z = np.zeros(NMAX)
    for i in range(0, NMAX):
        # zi = bi - sum(Aij*zj) j = 1...i-1
        second_sum = 0
        for j in range(0, i):
            second_sum += A[i, j] * z[j]
        z[i] = b[i] - second_sum

    # solve Dy = z
    y = np.zeros(NMAX)
    for i in range(0, NMAX):
        # y[i] = z[i] / diagonal[i]
        if abs(diagonal[i]) > eps:
            y[i] = z[i] / diagonal[i]
        else:
            Exception("Divide by zero, the system cannot be solved")

    # solve L(t)x = y
    x_chol = np.zeros(NMAX)
    for i in range(NMAX - 1, -1, -1):
        # xi = bi - sum(Aij*xj) j = i+1...n
        second_sum = 0
        for j in range(i + 1, NMAX):
            second_sum += A[j, i] * x_chol[j]
        x_chol[i] = y[i] - second_sum

    return x_chol


def rebuilt_init(matrix, NMAX):
    Ainit = np.zeros((NMAX, NMAX))
    for i in range(0, NMAX):
        for j in range(0, NMAX):
            if i <= j:
                Ainit[i, j] = matrix[i, j]
            else:
                Ainit[i, j] = matrix[j, i]
    return Ainit


def verify(A, x, b, NMAX):
    z = np.zeros(NMAX)
    for i in range(0, NMAX):
        second_sum = 0
        for j in range(0, NMAX):
            second_sum += A[i, j] * x[j]
        z[i] = second_sum - b[i]
    return math.sqrt(np.sum(z.dot(z)))


def main_function(finalMatrix, matrixDimension, b, eps):
    # matrix is symmetric, compute Choleski decomposition
    A, D = solve_Choleski(finalMatrix, matrixDimension, eps)

    det_A = determinant(D)  # det A = det L * det D * det L(t) = 1 * det D * 1 = det D

    # calculate Ax = b using LDL(t)
    x_chol = solve_system(A, b, D, eps)

    # compute LU decomposition using scipy
    Ainit = scipy.array(rebuilt_init(A, matrixDimension))
    P, L, U = scipy.linalg.lu(Ainit)

    # solve Ainit * x = b using numpy
    b = scipy.array(b)
    solve_system_scipy = scipy.linalg.solve(Ainit, b)

    # verify solution
    norm = verify(Ainit, x_chol, b, matrixDimension)

    return A, D, det_A, x_chol, L, U, solve_system_scipy, norm
