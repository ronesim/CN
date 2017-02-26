import numpy as np


def readFile(filePath):
    with open(filePath) as f:
        n, eps = [int(x) for x in next(f).split()]  # n - matrix dim, eps - precision
        b = np.array([float(x) for x in f.readline().split()])
        A = np.matrix([[float(x) for x in line.split()] for line in f])  # symmetric and well positive defined matrix
    eps = 10 ** (-eps)  # precision
    return A, b, n, eps


def printMatrix(matrix, diagonal, n):
    # print matrix A
    print("A:")
    for row in range(0, n):
        for col in range(0, n):
            if col >= row:
                print(matrix[row, col], end=' ')
            else:
                print(matrix[col, row], end=' ')
        print()

    print("L:")
    for row in range(0, n):
        for col in range(0, n):
            if col < row:
                print(matrix[row, col], end=' ')
            elif col > row:
                print(0, end=' ')
            else:
                print(1, end=' ')
        print()
    print("D:")
    for row in range(0, n):
        for col in range(0, n):
            if row is col:
                print(diagonal[row], end=' ')
            else:
                print(0, end=' ')
        print()


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


def solveSystem(A, b, diagonal):
    return 1  # TO DO (ronesim)


matrix, b, n, eps = readFile("input.txt")
A, D = solveCholeski(matrix, n, eps)
printMatrix(A, D, n)

detA = determinant(D)  # det A = det L * det D * det L(t) = 1 * det D * 1 = det D
print("Determinant A = {}".format(detA))

# calculate Ax = b using LDL(t)
ans = solveSystem(A, b, D)
