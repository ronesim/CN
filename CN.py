from flask import Flask, render_template, request

from library import util, choleskiApp, linearSystemGSApp, matrixInverseApp

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choleskiTool')
def choleski_decomposition():
    return render_template('choleski/choleskiDec.html')


@app.route('/submitCholeski', methods=['POST'])
def submit_choleski():
    # get information from form
    matrix_size, precision, matrix, b = util.get_data_from_form("system", request)

    # validate matrix and vector using matrix size
    b, eps = util.refactor_data(b, precision)

    if util.validate_data("standard", matrix_size, matrix.split(), b):
        final_matrix = util.get_processed_standard_matrix(matrix, matrix_size)

        if util.is_symmetric(final_matrix):
            A, D, detA, xChol, L, U, solveSystemScipy, norm = choleskiApp.main_function(final_matrix, matrix_size, b,
                                                                                        eps)
            return render_template('choleski/choleskiResult.html', NMAX=matrix_size, Amatrix=A, D=D, detA=detA,
                                   xChol=xChol, Llu=L, Ulu=U, solveSystemScipy=solveSystemScipy, norm=norm,
                                   normComp=norm < eps)
        else:
            return "Matrix is not symmetric"  # TODO create template
    else:
        return "Matrix is not symmetric"  # TODO create template


@app.route('/sparseMatrix')
def sparse_matrix():
    return render_template('sparseMatrix/sparseMatrix.html')


@app.route('/linearSystem')
def linear_system():
    return render_template('linearSystem/linearSystem.html')


@app.route('/submitLinearSystem', methods=['POST'])
def submit_linear_system():
    # get information from form
    matrix_size, precision, matrix, b = util.get_data_from_form("system", request)

    # validate matrix and vector using matrix size
    b, eps = util.refactor_data(b, precision)

    if util.validate_data("sparse", matrix_size, matrix.split(), b):
        sparse_matrix_representation = util.get_processed_sparse_matrix(matrix, matrix_size)
        XGS, info = linearSystemGSApp.main_function(sparse_matrix_representation, b, eps)
        if type(XGS) == str and XGS == "no solution":
            return render_template('linearSystem/linearSystemResult.html', NMAX=matrix_size,
                                   sparse_matrix_representation=sparse_matrix_representation, XGS="No Solution",
                                   info=info)
        # compute norm
        norm = linearSystemGSApp.compute_norm(sparse_matrix_representation, XGS, b)
        return render_template('linearSystem/linearSystemResult.html', NMAX=matrix_size,
                               sparse_matrix_representation=sparse_matrix_representation, XGS=XGS, norm=norm, info=info)

    else:
        return "Invalid matrix"  # TODO create template


@app.route('/matrixInverse')
def matrix_inverse():
    return render_template('matrixInverse/matrixInverse.html')


@app.route('/submitMatrixInverse', methods=['POST'])
def submit_matrix_inverse():
    # get information from form
    matrix_size, precision, matrix = util.get_data_from_form("standard", request)
    eps = 10 ** (-precision)

    if util.is_square(matrix_size, matrix.split()) and matrix_size < 10:
        final_matrix = util.get_processed_standard_matrix(matrix, matrix_size)
        inv_list = list(matrixInverseApp.main_function(final_matrix, matrix_size, eps))

        norms = list()
        norms.append(matrixInverseApp.get_norm(final_matrix, inv_list[0], matrix_size))
        norms.append(matrixInverseApp.get_norm(final_matrix, inv_list[2], matrix_size))
        norms.append(matrixInverseApp.get_norm(final_matrix, inv_list[4], matrix_size))

        return render_template('matrixInverse/matrixInverseResult.html', NMAX=matrix_size, init_matrix=final_matrix,
                               inv_matrixes=inv_list, norms=norms)


if __name__ == '__main__':
    app.run()
