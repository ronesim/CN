from flask import Flask, render_template, request

from library import util, choleski, linearSystemGS

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
    matrixDimension, precision, matrix, b = util.get_data_from_form(request)

    # validate matrix and vector using matrix size
    b, eps = util.refactor_data(b, precision)

    if util.validate_data("standard", matrixDimension, matrix.split(), b):
        finalMatrix = util.get_processed_standard_matrix(matrix, matrixDimension)

        if util.isSymmetric(finalMatrix):
            A, D, detA, xChol, L, U, solveSystemScipy, norm = choleski.main_function(finalMatrix, matrixDimension, b,
                                                                                     eps)
            return render_template('choleski/choleskiResult.html', NMAX=matrixDimension, Amatrix=A, D=D, detA=detA,
                                   xChol=xChol, Llu=L, Ulu=U, solveSystemScipy=solveSystemScipy, norm=norm,
                                   normComp=norm < eps)
        else:
            return "Matrix is not symmetric"  # TODO create template
    else:
        return "Matrix is not symmetric"  # TODO create template


@app.route('/sparseMatrix')
def sparseMatrix():
    return render_template('sparseMatrix/sparseMatrix.html')


@app.route('/linearSystem')
def linearSystem():
    return render_template('linearSystem/linearSystem.html')


@app.route('/submitLinearSystem', methods=['POST'])
def submitLinearSystem():
    # get information from form
    matrixDimension, precision, matrix, b = util.get_data_from_form(request)

    # validate matrix and vector using matrix size
    b, eps = util.refactor_data(b, precision)

    if util.validate_data("sparse", matrixDimension, matrix.split(), b):
        sparse_matrix_representation = util.get_processed_sparse_matrix(matrix, matrixDimension)
        XGS, info = linearSystemGS.main_function(sparse_matrix_representation, b, eps)
        print(XGS)
        if type(XGS) == str and XGS == "no solution":
            return render_template('linearSystem/linearSystemResult.html', NMAX=matrixDimension,
                                   sparse_matrix_representation=sparse_matrix_representation, XGS="No Solution",
                                   info=info)
        # compute norm
        norm = linearSystemGS.compute_norm(sparse_matrix_representation, XGS, b)
        return render_template('linearSystem/linearSystemResult.html', NMAX=matrixDimension,
                               sparse_matrix_representation=sparse_matrix_representation, XGS=XGS, norm=norm, info=info)

    else:
        return "Invalid matrix"  # TODO create template
if __name__ == '__main__':
    app.run()
