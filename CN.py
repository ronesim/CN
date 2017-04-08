import numpy as np
import scipy.linalg
from flask import Flask, render_template, request

from library import util, choleski

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choleskiTool')
def choleskiDecomposition():
    return render_template('choleski/choleskiDec.html')


@app.route('/submitCholeski', methods=['POST'])
def submitCholeski():
    # get information from form
    matrixDimension = int(request.form['inputMatrixSize'])  # matrix size
    precision = int(request.form['inputPrecision'])  # eps precision
    matrix = request.form['inputMatrix']  # matrix A
    b = request.form['inputVector']  # Ax = b

    # validate matrix and vector using matrix size
    b = np.array([float(x) for x in b.split()])
    eps = 10 ** (-precision)
    if util.isSquare(matrixDimension, matrix.split()) and len(b) == matrixDimension:
        # process given matrix
        elements = matrix.split()
        processedMatrix = []
        for rows in range(0, matrixDimension):
            row = []
            for column in range(0, matrixDimension):
                row.append(float(elements[rows * matrixDimension + column]))
            processedMatrix.append(row)
        finalMatrix = np.matrix(processedMatrix)

        if util.isSymmetric(finalMatrix):
            # matrix is symmetric, compute Choleski decomposition
            A, D = choleski.solveCholeski(finalMatrix, matrixDimension, eps)

            detA = choleski.determinant(D)  # det A = det L * det D * det L(t) = 1 * det D * 1 = det D

            # calculate Ax = b using LDL(t)
            xChol = choleski.solveSystem(A, b, D, eps)

            # compute LU decomposition using scipy
            Ainit = scipy.array(choleski.rebuiltInit(A, matrixDimension))
            P, L, U = scipy.linalg.lu(Ainit)

            # solve Ainit * x = b using numpy
            b = scipy.array(b)
            solveSystemScipy = scipy.linalg.solve(Ainit, b)

            # verify solution
            norm = choleski.verify(Ainit, xChol, b, matrixDimension)

            return render_template('choleski/choleskiResult.html', NMAX=matrixDimension, Amatrix=A, D=D, detA=detA,
                                   xChol=xChol, Llu=L, Ulu=U,
                                   solveSystemScipy=solveSystemScipy, norm=norm, normComp=norm < eps)
        else:
            return "Matrix is not symmetric"
    else:
        return "Matrix is not symmetric"


if __name__ == '__main__':
    app.run()
