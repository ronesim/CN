import numpy as np


def isSquare(matrixDimension, matrix):
    if matrixDimension ** 2 == len(matrix):
        return True
    return False


def isSymmetric(matrix):
    return (matrix.transpose() == matrix).all()


def get_data_from_form(request):
    '''
    
    :param request: form request
    :return: matrix size, eps precision, matrix A, b (Ax = b)
    '''
    return int(request.form['inputMatrixSize']), int(request.form['inputPrecision']), request.form['inputMatrix'], \
           request.form['inputVector']


def refactor_data(b, precision):
    return np.array([float(x) for x in b.split()]), 10 ** (-precision)


def validate_data(type, matrixDimension, elements, b):
    if type is "sparse":
        return len(b) == matrixDimension and matrixDimension <= 10

    if type is "standard":
        return isSquare(matrixDimension, elements) and len(b) == matrixDimension and matrixDimension <= 10


def get_processed_standard_matrix(matrix, matrixDimension):
    # process given matrix
    elements = matrix.split()
    processedMatrix = []
    for rows in range(0, matrixDimension):
        row = []
        for column in range(0, matrixDimension):
            row.append(float(elements[rows * matrixDimension + column]))
        processedMatrix.append(row)
    return np.matrix(processedMatrix)


def get_processed_sparse_matrix(matrix, matrixDimension):
    elements = matrix.split()
    # # READ AND COMPUTE O(nlogn)
    d = [0 for _ in range(matrixDimension)]
    NN = 0
    matrix = {}
    for index in range(0, len(elements), 3):
        value = float(elements[index].split(',')[0])
        row = int(elements[index + 1].split(',')[0])
        col = int(elements[index + 2].split(',')[0])
        if row == col:
            d[row] += value
        else:
            NN += 1
            if not (row, col) in matrix:
                matrix[(row, col)] = value
            else:
                matrix[(row, col)] += value

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
    return {'n': matrixDimension, 'nn': NN, 'd': d, 'val': val, 'col': col}
