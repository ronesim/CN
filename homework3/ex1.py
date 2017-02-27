import time


def print_matrix(matrix):
    print('\n'.join([' '.join(map(str, row)) for row in matrix]))


def matrix_sum(m1, m2):
    # initialise sum as m1
    n = m1['n']
    nn = m1['nn']
    d = list(m1['d]'])
    val = list(m1['val'])
    col = list(m1['col'])
    col_len = n + nn + 1

    # add second matrix to first matrix
    c1 = 0
    r1 = 0
    c2 = 1
    r2 = -1
    while m2['col'][c1] != -(n+1):
        if m2['col'][c1] < 0:
            row = m2['col'][c1]

        while c2 < col_len and 0 < col[c2] < m2['col'][c1]:
            c2 += 1

        if c1 == c2



def matrix_multiply(matrix1, matrix2):
    pass


def vector_multiply(matrix1, vector):
    pass


def read_file(file_path):
    with open(file_path, 'r') as handle:
        # read number of elements
        n = int(handle.readline().strip())
        handle.readline()

        # read vector
        b = []
        for counter in range(n):
            b.append(float(handle.readline().strip()))
        handle.readline()

        # read matrix and check less than 10 elements pe line
        d = [0 for _ in range(n)]
        val = [0 for _ in range(n + 1)]
        col = [-(k+1) for k in range(n + 1)]
        nn = 0
        line = handle.readline().strip()
        while line:
            value, i, j = map(lambda x: int(x[1]) if x[0] else float(x[1]), enumerate(line.split(',')))
            if i == j:
                d[i] += value
            else:
                nn += 1
                for c, m in enumerate(col):
                    if m == -(i+1):
                        for count, z in enumerate(col[c + 1:]):
                            if z == j + 1:
                                val[count + c + 1] += value
                                break
                            elif z < 0 or z > j + 1:
                                col.insert(count + c + 1, j + 1)
                                val.insert(count + c + 1, value)
                                break
                            elif count == 9:
                                exit("More than 10 non-null values on line {}: {}..."
                                     "".format(i, ' '.join(map(str, col[c + 1:c + 11]))))
                        break
            line = handle.readline().strip()

        return {'n': n, 'nn': nn, 'b': b, 'd': d, 'val': val, 'col': col}

print(read_file('test.txt'))
matrixA = read_file('a.txt')
matrixB = read_file('b.txt')
