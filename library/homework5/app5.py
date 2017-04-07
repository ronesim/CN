import pprint
import sys

import numpy as np
from numpy import linalg as LA


def readFile(filePath):
    with open(filePath) as f:
        n, eps, kMax = [int(x) for x in next(f).split()]  # n - matrix dim, eps - precision, kMax - number of iterations
        A = np.matrix([[float(x) for x in line.split()] for line in f])  # symmetric and well positive defined matrix
    eps = 10 ** (-eps)  # precision
    return A, n, eps, kMax


def generateV0(A):
    return A.transpose() / (np.max(np.sum(A, axis=0)) * np.max(np.sum(A, axis=1)))


def addElemDiagonal(matrix, value):
    for row in range(len(matrix)):
        matrix[row, row] += value
    return np.matrix(matrix)


def removeElemDiagonal(matrix, value):
    for row in range(len(matrix)):
        matrix[row, row] -= value
    return np.matrix(matrix)


def inverseSchulzMethod(A, B, n, eps, kMax):
    # initialize V0
    V0 = generateV0(A)
    # V(k+1) = V(k) *(2*I(n) - A*V(k))
    while kMax > 0:
        # compute V1 using V0
        C = B * V0
        # add 2 on C's diagonal and multiply with V0
        V1 = V0 * addElemDiagonal(C, 2)
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def inverseLiMethodI(A, B, n, eps, kMax):
    # initialize V0
    V0 = generateV0(A)
    # V(k+1) = V(k) *(3*I(n) - A*V(k)(3*I(n) - A*V(k)))
    while kMax > 0:
        # compute V1 using V0
        C = B * V0
        auxMat = addElemDiagonal(C, 3)
        V1 = V0 * addElemDiagonal(removeElemDiagonal(C, 3) * auxMat, 3)
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def inverseLiMethodII(A, B, n, eps, kMax):
    # initialize V0
    V0 = generateV0(A)
    # V(k+1) = (I(n) + 1/4 * (I(n) - V(k) *A)(3*I(n) - V(k) *A)^2) *V(k)
    while kMax > 0:
        # compute V1 using V0
        C = V0 * B
        auxMat = addElemDiagonal(C, 3) ** 2
        V1 = addElemDiagonal(removeElemDiagonal(C, 2) * auxMat / 4, 1) * V0
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


matrix, n, eps, kMax = readFile(sys.argv[1])

# compute B = -matrix: used in C = a*I(n) - A * V
B = np.matrix((-1) * matrix)

matrixInvSchulz, iter = inverseSchulzMethod(matrix, B, n, eps, kMax)
print("Schultz method:")
if type(matrixInvSchulz) == str and matrixInvSchulz == "Divergent":
    print("There is no inverse after {} iterations".format(kMax))
else:
    print("Inverse found after {} iterations".format(kMax - iter))
    print("Norm {}".format(LA.norm((matrix * matrixInvSchulz - np.identity(n)))))
    print("Norm ||A * A^(-1) - I(n)||1 is {}".format(np.max(np.sum(matrix * matrixInvSchulz - np.identity(n)), axis=0)))
    pprint.pprint(matrixInvSchulz)

matrixInvLi, iter = inverseLiMethodI(matrix, B, n, eps, kMax)
print("LI & LI first method:")
if type(matrixInvLi) == str and matrixInvLi == "Divergent":
    print("There is no inverse after {} iterations".format(kMax))
else:
    print("Inverse found after {} iterations".format(kMax - iter))
    print("Norm {}".format(LA.norm((matrix * matrixInvLi - np.identity(n)))))
    print("Norm ||A * A^(-1) - I(n)||1 is {}".format(np.max(np.sum(matrix * matrixInvLi - np.identity(n)), axis=0)))
    pprint.pprint(matrixInvLi)

matrixInvLiII, iter = inverseLiMethodII(matrix, B, n, eps, kMax)
print("LI & LI second method:")
if type(matrixInvLiII) == str and matrixInvLiII == "Divergent":
    print("There is no inverse after {} iterations".format(kMax))
else:
    print("Inverse found after {} iterations".format(kMax - iter))
    print("Norm {}".format(LA.norm((matrix * matrixInvLiII - np.identity(n)))))
    print("Norm ||A * A^(-1) - I(n)||1 is {}".format(np.max(np.sum(matrix * matrixInvLiII - np.identity(n)), axis=0)))
    pprint.pprint(matrixInvLiII)
