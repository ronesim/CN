def isSquare(matrixDimension, matrix):
    if matrixDimension ** 2 == len(matrix):
        return True
    return False


def isSymmetric(matrix):
    return (matrix.transpose() == matrix).all()
