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


def inverseSchulzMethod(A, n, eps, kMax):
    # initialize V0
    V0 = generateV0(A)
    # V(k+1) = V(k) *(2*I(n) - A*V(k))
    while kMax > 0:
        # compute V1 using V0
        V1 = V0 * (2 * np.identity(n) + (-1) * A * V0)
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def inverseLiMethodI(A, n, eps, kMax):
    # initialize V0
    V0 = generateV0(A)
    # V(k+1) = V(k) *(3*I(n) - A*V(k)(3*I(n) - A*V(k)))
    while kMax > 0:
        # compute V1 using V0
        V1 = V0 * (3 * np.identity(n) + (((-1) * A * V0) * (3 * np.identity(n) + (-1) * A * V0)))
        # check exit condition
        norm = LA.norm(V1 - V0)
        if norm < eps:
            return V1, kMax
        if norm > 10 ** 10:
            return "Divergent", kMax

        V0 = V1
        kMax -= 1

    return "Divergent", 0


def inverseLiMethodII(A, n, eps, kMax):
    # initialize V0
    V0 = generateV0(A)
    # V(k+1) = (I(n) + 1/4 * (I(n) - V(k) *A)(3*I(n) - V(k) *A)^2) *V(k)
    while kMax > 0:
        # compute V1 using V0
        auxV = 3 * np.identity(n) + (-1) * A * V0
        V1 = V0 * (np.identity(n) + ((np.identity(n) + (-1) * A * V0) * auxV * auxV) / 4)
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

matrixInvSchulz, iter = inverseSchulzMethod(matrix, n, eps, kMax)
print("Schultz method:")
if matrixInvSchulz is "Divergent":
    print("There is no inverse after {} iterations".format(kMax))
else:
    print("Inverse found after {} iterations".format(kMax - iter))
    print("Norm {}".format(LA.norm((matrix * matrixInvSchulz - np.identity(n)))))
    print("Norm ||A * A^(-1) - I(n)||1 is {}".format(np.max(np.sum(matrix * matrixInvSchulz - np.identity(n)), axis=0)))
    pprint.pprint(matrixInvSchulz)

matrixInvLi, iter = inverseLiMethodI(matrix, n, eps, kMax)
print("LI & LI first method:")
if matrixInvLi is "Divergent":
    print("There is no inverse after {} iterations".format(kMax))
else:
    print("Inverse found after {} iterations".format(kMax - iter))
    print("Norm {}".format(LA.norm((matrix * matrixInvLi - np.identity(n)))))
    print("Norm ||A * A^(-1) - I(n)||1 is {}".format(np.max(np.sum(matrix * matrixInvLi - np.identity(n)), axis=0)))
    pprint.pprint(matrixInvLi)

matrixInvLiII, iter = inverseLiMethodII(matrix, n, eps, kMax)
print("LI & LI second method:")
if matrixInvLiII is "Divergent":
    print("There is no inverse after {} iterations".format(kMax))
else:
    print("Inverse found after {} iterations".format(kMax - iter))
    print("Norm {}".format(LA.norm((matrix * matrixInvLiII - np.identity(n)))))
    print("Norm ||A * A^(-1) - I(n)||1 is {}".format(np.max(np.sum(matrix * matrixInvLiII - np.identity(n)), axis=0)))
    pprint.pprint(matrixInvLiII)
