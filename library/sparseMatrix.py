import os
import time


def print_matrix(matrix):
    print('\n'.join([' '.join(map(str, row)) for row in matrix]))


def matrix_sum(m1, m2):
    # initialise sum as m1
    n = m1['n']
    nn = m1['nn']
    d = list(m1['d'])
    val = list(m1['val'])
    col = list(m1['col'])
    col_len = n + nn + 1

    # add second matrix to first matrix
    c1 = 1
    c2 = 1
    while m2['col'][c2] != -(n + 1):
        # new line beginnings
        while m2['col'][c2] < 0:
            while col[c1] > 0:
                c1 += 1

            # get to the values
            c1 += 1
            c2 += 1
            if m2['col'][c2] == -(n + 1):
                break

        # follow both lines at the same time
        while c1 < col_len and 0 < col[c1] < m2['col'][c2]:
            c1 += 1

        # add value from second matrix to first
        if col[c1] == m2['col'][c2]:
            val[c1] += m2['val'][c2]
        else:
            col.insert(c1, m2['col'][c2])
            val.insert(c1, m2['val'][c2])
            nn += 1
            col_len += 1

        # move to the next element
        c2 += 1

    # add diagonals
    d = list(map(lambda k: k[0] + k[1], zip(d, m2['d'])))

    return {'n': n, 'nn': nn, 'd': d, 'val': val, 'col': col}


def matrix_multiply(m1, m2):
    # initialise new matrix
    n = len(m1['d'])
    result_matrix = {}

    col_one = 0
    row_one = 0
    while m1['col'][col_one] != -(n + 1):
        # get this line from m1
        row_one += 1
        col_one += 1
        this_line1 = {}
        not_added = True
        while m1['col'][col_one] > 0:
            if not_added and m1['col'][col_one] > row_one:
                not_added = False
                this_line1[row_one] = m1['d'][row_one - 1]
            this_line1[m1['col'][col_one]] = m1['val'][col_one]
            col_one += 1
        if not_added:
            this_line1[row_one] = m1['d'][row_one - 1]

        # get all lines of m2
        col_two = 0
        row_two = 0
        while m2['col'][col_two] != -(n + 1):
            row_two += 1
            col_two += 1
            while m2['col'][col_two] > 0:
                if row_two in this_line1:
                    temp_value = result_matrix.get((row_one, m2['col'][col_two]), 0)
                    result_matrix[(row_one, m2['col'][col_two])] = temp_value + this_line1[row_two] * m2['val'][col_two]
                col_two += 1

        # get all diagonal items of m2
        for col_three in range(n):
            if m2['d'][col_three]:
                if col_three + 1 in this_line1:
                    temp_value = result_matrix.get((row_one, col_three + 1), 0)
                    result_matrix[(row_one, col_three + 1)] = temp_value + this_line1[col_three + 1] * m2['d'][
                        col_three]

    # parse new matrix and return it
    d = [0 for _ in range(n)]
    val = [0]
    col = [-1]
    nn = 0
    last_row = None
    for item in sorted(result_matrix.keys()):
        if not result_matrix[item]:
            continue
        if last_row is None:
            last_row = item[0]
        if item[0] == item[1]:
            d[item[0] - 1] = result_matrix[item]
        else:
            if item[0] != last_row:
                last_row = item[0]
                val.append(0)
                col.append(-item[0])
            val.append(result_matrix[item])
            col.append(item[1])
            nn += 1
    val.append(0)
    col.append(-(n + 1))
    return {'n': n, 'nn': nn, 'd': d, 'val': val, 'col': col}


def vector_multiply(matrix, vector):
    # response vector
    ans = [0 for _ in range(matrix['n'])]

    # iterate over whole matrix
    n = matrix['n']
    row = 1
    for index in range(1, len(matrix['col'])):
        if matrix['col'][index] < 0:
            row += 1
        else:
            ans[row - 1] += matrix['val'][index] * vector[matrix['col'][index] - 1]

    # iterate over matrix diagonal
    for i in range(n):
        ans[i] += matrix['d'][i] * vector[i]

    return ans


def read_file(file_path, check=True):
    with open(file_path, 'r') as handle:
        # read number of elements
        n = int(handle.readline().strip())
        handle.readline()

        # read vector
        b = []
        for counter in range(n):
            b.append(float(handle.readline().strip()))
        handle.readline()

        # # READ AND COMPUTE O(nlogn)
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
        count = 0
        for item in sorted(matrix.keys()):
            while item[0] != row:
                row += 1
                col.append(-(row + 1))
                val.append(0)
                count = 0
            col.append(item[1] + 1)
            val.append(matrix[item])
            count += 1
            if count == 10 and check:
                exit("More than 10 non-null values on line...")
        col.append(-(row + 2))
        val.append(0)
        return {'n': n, 'nn': nn, 'd': d, 'val': val, 'col': col}, b


def main_function(file_names):
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    APP_UPLOADS = os.path.join(APP_ROOT, 'uploads/')
    reading_time = []
    start_time = time.time()
    matA, vecA = read_file(APP_UPLOADS + file_names[0])
    reading_time.append('Reading took {:.2f} seconds'.format(time.time() - start_time))

    matB, vecB = read_file(APP_UPLOADS + file_names[1])
    reading_time.append('Reading took {:.2f} seconds'.format(time.time() - start_time))

    sumM, vecSum = read_file(APP_UPLOADS + file_names[2], check=False)
    reading_time.append('Reading took {:.2f} seconds'.format(time.time() - start_time))

    mulM, vecMul = read_file(APP_UPLOADS + file_names[3], check=False)
    reading_time.append('Reading took {:.2f} seconds'.format(time.time() - start_time))

    computing_time = []

    start_time = time.time()
    computing_time.append('Sum is ok: {}'.format(matrix_sum(matA, matB) == sumM))
    computing_time.append('Matrix sum finished in {:.2f} seconds'.format(time.time() - start_time))

    start_time = time.time()
    product_result = matrix_multiply(matA, matB)
    computing_time.append('Product is ok: {}'.format(product_result == mulM))
    computing_time.append('Matrix product finished in {:.2f} seconds'.format(time.time() - start_time))

    start_time = time.time()
    product_result = vector_multiply(matA, [2017 - k for k in range(2017)])
    computing_time.append('Vector product is ok: {}'.format(product_result == vecA))
    computing_time.append('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    start_time = time.time()
    product_result = vector_multiply(matA, [2017 - k for k in range(2017)])
    computing_time.append('Vector product is ok: {}'.format(product_result == vecA))
    computing_time.append('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    start_time = time.time()
    product_result = vector_multiply(sumM, [2017 - k for k in range(2017)])
    computing_time.append('Vector product is ok: {}'.format(product_result == vecSum))
    computing_time.append('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    start_time = time.time()
    product_result = vector_multiply(mulM, [2017 - k for k in range(2017)])
    computing_time.append('Vector product is ok: {}'.format(product_result == vecMul))
    computing_time.append('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    return (reading_time, computing_time)
