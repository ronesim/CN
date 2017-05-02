import numpy as np


def is_square(matrix_size, matrix):
    if matrix_size ** 2 == len(matrix):
        return True
    return False


def is_symmetric(matrix):
    return (matrix.transpose() == matrix).all()


def get_data_from_form(type, request):
    """
    Util function used for getting data from the request
    :param type, request: type (system - has result vector
                          standard - size, matrix and precision
                    form request
    :return: matrix size, eps precision, matrix A, b (Ax = b)
    """
    if type is "system":
        return int(request.form['inputMatrixSize']), int(request.form['inputPrecision']), request.form['inputMatrix'], \
               request.form['inputVector']
    if type is "standard":
        return int(request.form['inputMatrixSize']), int(request.form['inputPrecision']), request.form['inputMatrix']


def refactor_data(b, precision):
    return np.array([float(x) for x in b.split()]), 10 ** (-precision)


def validate_data(type, matrix_size, elements, b):
    if type is "sparse":
        return len(b) == matrix_size and matrix_size <= 10

    if type is "standard":
        return is_square(matrix_size, elements) and len(b) == matrix_size and matrix_size <= 10


def get_processed_standard_matrix(matrix, matrix_size):
    # process given matrix
    elements = matrix.split()
    processed_matrix = []
    for rows in range(0, matrix_size):
        row = []
        for column in range(0, matrix_size):
            row.append(float(elements[rows * matrix_size + column]))
        processed_matrix.append(row)
    return np.matrix(processed_matrix)


def get_processed_sparse_matrix(matrix, matrix_size):
    elements = matrix.split()
    # # READ AND COMPUTE O(nlogn)
    d = [0 for _ in range(matrix_size)]
    NNMAX = 0
    matrix = {}
    for index in range(0, len(elements), 3):
        value = float(elements[index].split(',')[0])
        row = int(elements[index + 1].split(',')[0])
        col = int(elements[index + 2].split(',')[0])
        if row == col:
            d[row] += value
        else:
            NNMAX += 1
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
    return {'n': matrix_size, 'nn': NNMAX, 'd': d, 'val': val, 'col': col}


def read_sparse_matrix_from_file(file_path, check=True):
    with open(file_path, 'r') as handle:
        # read number of elements
        n = int(handle.readline().strip())
        handle.readline()

        d = [0 for _ in range(n)]
        nn = 0
        matrix = {}
        line = handle.readline().strip()
        while line:
            value, i, j = map(lambda x: int(x[1]) if x[0] else float(x[1]), enumerate(line.split(',')))
            if i == j:
                d[i] += value
            else:
                nn += 1
                if not (i, j) in matrix:
                    matrix[(i, j)] = value
                else:
                    matrix[(i, j)] += value
            line = handle.readline().strip()

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

        return {'n': n, 'nn': nn, 'd': d, 'val': val, 'col': col}
