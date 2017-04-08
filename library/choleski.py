import math

import numpy as np


def solveCholeski(A, n, eps):
    # Compute A = LDL^T Choleski decomposition
    D = np.zeros(n)
    for index in range(0, n):
        # compute D[index] = A[index][index] - sum(D[k] * L[index][k]^2) k=1...index-1
        secondSum = 0
        for k in range(0, index):
            secondSum += D[k] * A.item((index, k)) * A.item((index, k))
        D[index] = A.item((index, index)) - secondSum

        # compute L[i][index] = (a[i][index] - sum(D[k]*L[i][k]*L[p][k])) / D[index] i = index+1...n
        for i in range(index + 1, n):
            secondSum = 0
            for k in range(0, index):
                secondSum += D[k] * A.item((i, k)) * A.item((index, k))  # L is computed using A matrix
            if abs(D[index]) > eps:
                A[i, index] = (A.item((i, index)) - secondSum) / D[index]
            else:
                raise Exception("Divide by zero, Choleski decomposition cannot be computed")
    return A, D


def determinant(diagonal):
    return diagonal.prod()


def solveSystem(A, b, diagonal, eps):
    # solve Lz = b
    NMAX = len(b)  # size
    z = np.zeros(NMAX)
    for i in range(0, NMAX):
        # zi = bi - sum(Aij*zj) j = 1...i-1
        sumSecond = 0
        for j in range(0, i):
            sumSecond += A[i, j] * z[j]
        z[i] = b[i] - sumSecond

    # solve Dy = z
    y = np.zeros(NMAX)
    for i in range(0, NMAX):
        # y[i] = z[i] / diagonal[i]
        if abs(diagonal[i]) > eps:
            y[i] = z[i] / diagonal[i]
        else:
            Exception("Divide by zero, the system cannot be solved")

    # solve L(t)x = y
    xChol = np.zeros(NMAX)
    for i in range(NMAX - 1, -1, -1):
        # xi = bi - sum(Aij*xj) j = i+1...n
        sumSecond = 0
        for j in range(i + 1, NMAX):
            sumSecond += A[j, i] * xChol[j]
        xChol[i] = y[i] - sumSecond

    return xChol


def rebuiltInit(matrix, NMAX):
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
        secondSum = 0
        for j in range(0, NMAX):
            secondSum += A[i, j] * x[j]
        z[i] = secondSum - b[i]
    return math.sqrt(np.sum(z.dot(z)))
