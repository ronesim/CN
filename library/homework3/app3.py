import pickle
import sys
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

    c1 = 0
    line1 = 0
    while m1['col'][c1] != -(n + 1):
        # get this line from m1
        line1 += 1
        c1 += 1
        this_line1 = {}
        not_added = True
        while m1['col'][c1] > 0:
            if not_added and m1['col'][c1] > line1:
                not_added = False
                this_line1[line1] = m1['d'][line1 - 1]
            this_line1[m1['col'][c1]] = m1['val'][c1]
            c1 += 1
        if not_added:
            this_line1[line1] = m1['d'][line1 - 1]

        # get all lines of m2
        c2 = 0
        line2 = 0
        while m2['col'][c2] != -(n + 1):
            line2 += 1
            c2 += 1
            while m2['col'][c2] > 0:
                if line2 in this_line1:
                    temp_value = result_matrix.get((line1, m2['col'][c2]), 0)
                    result_matrix[(line1, m2['col'][c2])] = temp_value + this_line1[line2] * m2['val'][c2]
                c2 += 1

        # get all diagonal items of m2
        for c3 in range(n):
            if m2['d'][c3]:
                if c3 + 1 in this_line1:
                    temp_value = result_matrix.get((line1, c3 + 1), 0)
                    result_matrix[(line1, c3 + 1)] = temp_value + this_line1[c3 + 1] * m2['d'][c3]

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


def vector_multiply(m1, vector):
    # response vector
    r_vec = [0 for _ in range(m1['n'])]

    # iterate over whole matrix
    n = m1['n']
    c1 = 0
    line1 = 1
    while m1['col'][c1] != -(n + 1):
        c1 += 1
        if m1['col'][c1] < 0:
            line1 += 1
        else:
            r_vec[line1 - 1] += m1['val'][c1] * vector[m1['col'][c1] - 1]

    # iterate over matrix diagonal
    for i in range(n):
        r_vec[i] += m1['d'][i] * vector[i]

    return r_vec


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

        # TODO vlupoaie remove unused code
        # READ and COMPUTE (O(n^2)) matrix and check less than 10 elements per line
        # d = [0 for _ in range(n)]
        # val = [0 for _ in range(n + 1)]
        # col = [-(k + 1) for k in range(n + 1)]
        # nn = 0
        # line = handle.readline().strip()
        # while line:
        #     value, i, j = map(lambda x: int(x[1]) if x[0] else float(x[1]), enumerate(line.split(',')))
        #     if i == j:
        #         d[i] += value
        #     else:
        #         nn += 1
        #         for c, m in enumerate(col):
        #             if m == -(i + 1):
        #                 for count, z in enumerate(col[c + 1:]):
        #                     if z == j + 1:
        #                         val[count + c + 1] += value
        #                         break
        #                     elif z < 0 or z > j + 1:
        #                         col.insert(count + c + 1, j + 1)
        #                         val.insert(count + c + 1, value)
        #                         break
        #                     elif count == 9:
        #                         if check:
        #                             exit("More than 10 non-null values on line {}: {}..."
        #                                  "".format(i, ' '.join(map(str, col[c + 1:c + 11]))))
        #                         else:
        #                             print("More than 10 non-null values on line {}: {}..."
        #                                   "".format(i, ' '.join(map(str, col[c + 1:c + 11]))))
        #                 break
        #     line = handle.readline().strip()

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


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'reload':
        start_time = time.time()
        print('Reading matrix A')
        matA, vecA = read_file('a.txt')
        print(matA["col"])
        print('Reading took {:.2f} seconds'.format(time.time() - start_time))
        print('Reading matrix B')
        matB, vecB = read_file('b.txt')
        print('Reading took {:.2f} seconds'.format(time.time() - start_time))
        print('Reading matrix SUM')
        sumM, vecSum = read_file('aplusb.txt', check=False)
        print('Reading took {:.2f} seconds'.format(time.time() - start_time))
        print('Reading matrix MUL')
        mulM, vecMul = read_file('aorib.txt', check=False)
        print('Reading took {:.2f} seconds'.format(time.time() - start_time))
        with open('save.pkl', 'wb') as f_handle:
            x = (matA, vecA, matB, vecB, sumM, vecSum, mulM, vecMul)
            f_handle.write(pickle.dumps(x))
    else:
        start_time = time.time()
        with open('save.pkl', 'rb') as f_handle:
            matA, vecA, matB, vecB, sumM, vecSum, mulM, vecMul = pickle.loads(f_handle.read())
        print('Loading saved parsed files took {:.2f} seconds'.format(time.time() - start_time))

    print('\nComputing matrix sum...')
    start_time = time.time()
    print('Sum is ok: {}'.format(matrix_sum(matA, matB) == sumM))
    print('Matrix sum finished in {:.2f} seconds'.format(time.time() - start_time))

    print('\nComputing matrix product...')
    start_time = time.time()
    product_result = matrix_multiply(matA, matB)
    print('Product is ok: {}'.format(product_result == mulM))
    print('Matrix product finished in {:.2f} seconds'.format(time.time() - start_time))

    print('\nComputing vector product... matA * v = vecA')
    start_time = time.time()
    product_result = vector_multiply(matA, [2017 - k for k in range(2017)])
    print('Vector product is ok: {}'.format(product_result == vecA))
    print('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    print('\nComputing vector product... matB * v = vecB')
    start_time = time.time()
    product_result = vector_multiply(matA, [2017 - k for k in range(2017)])
    print('Vector product is ok: {}'.format(product_result == vecA))
    print('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    print('\nComputing vector product... sumM * v = vecSum')
    start_time = time.time()
    product_result = vector_multiply(sumM, [2017 - k for k in range(2017)])
    print('Vector product is ok: {}'.format(product_result == vecSum))
    print('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))

    print('\nComputing vector product... mulM * v = vecMul')
    start_time = time.time()
    product_result = vector_multiply(mulM, [2017 - k for k in range(2017)])
    print('Vector product is ok: {}'.format(product_result == vecMul))
    print('Vector product finished in {:.2f} seconds'.format(time.time() - start_time))
