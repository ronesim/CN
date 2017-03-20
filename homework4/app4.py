import math
import pickle
import pprint
import sys
import time

import numpy as np

import homework3.app3 as app3


def solveSystemGaussSeidel(A, b):
    XGS = np.zeros(A["n"])
    eps = 10 ** (-16)  # precision
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


if __name__ == '__main__':
    NUMBER_OF_FILES = 6

    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        start_time = time.time()
        savedMatrix = []
        for file in range(1, NUMBER_OF_FILES):
            matrix, b = app3.read_file("m_rar_2017_" + str(file) + ".txt")
            savedMatrix.append((matrix, b))
            print('Reading took {:.2f} for m_rar_2017_{}.txt file seconds'.format(time.time() - start_time, file))
        print('Reading took {:.2f} seconds'.format(time.time() - start_time))
        with open('save.pkl', 'wb') as f_handle:
            f_handle.write(pickle.dumps(savedMatrix))
    else:
        start_time = time.time()
        with open('save.pkl', 'rb') as f_handle:
            savedMatrix = pickle.loads(f_handle.read())
        print('Loading saved parsed files took {:.2f} seconds'.format(time.time() - start_time))

    numberSystem = 0
    for system in savedMatrix:
        # compute XGS for each tuple (A,B)
        if checkDiagonal(system[0]):  # if all diagonal elem are non-null
            XGS, iterations = solveSystemGaussSeidel(system[0], system[1])  # solve system
            if XGS == "no solution":
                print("System number {} cannot be solved".format(numberSystem))
                print("Number of iterations: {}".format(10000 - iterations))
            else:
                print("Solution for system number {} is:".format(numberSystem))
                pprint.pprint(XGS)
                print("Number of iterations: {}".format(10000 - iterations))
                # compute norm
                bAprox = app3.vector_multiply(system[0], XGS)
                print("Norm ||A * XGS - b||inf is {}".format(np.max(abs(np.array(bAprox) - np.array(system[1])))))
        else:
            print("System number {} cannot be solved. Diagonal has null elements".format(numberSystem))

        numberSystem += 1
