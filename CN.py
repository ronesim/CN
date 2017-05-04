import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from library import util, choleskiApp, linearSystemGSApp, matrixInverseApp, sparseMatrix, eingenvaluesSVD, functionsApp

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'library/uploads')
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
        final_matrix = util.get_processed_standard_matrix(matrix, matrix_size, matrix_size)

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


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/submitSparseMatrix', methods=['POST'])
def submit_sparse_matrix():
    file_a = request.files['aMatrix']
    file_b = request.files['bMatrix']
    file_aplusb = request.files['aplusbMatrix']
    file_aorib = request.files['aoribMatrix']

    count = 0
    if file_a and allowed_file(file_a.filename):
        filename = secure_filename(file_a.filename)
        file_a.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        count += 1
    if file_b and allowed_file(file_b.filename):
        filename = secure_filename(file_b.filename)
        file_b.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        count += 1
    if file_aplusb and allowed_file(file_aplusb.filename):
        filename = secure_filename(file_aplusb.filename)
        file_aplusb.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        count += 1
    if file_aorib and allowed_file(file_aorib.filename):
        filename = secure_filename(file_aorib.filename)
        file_aorib.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        count += 1

    if count == 4:
        result = sparseMatrix.main_function(
            [file_a.filename, file_b.filename, file_aplusb.filename, file_aorib.filename])
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_a.filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_b.filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_aplusb.filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file_aorib.filename))
        return render_template('sparseMatrix/sparseMatrixResult.html', result_time=result[0], solve_time=result[1])
    return "Invalid files"


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
        final_matrix = util.get_processed_standard_matrix(matrix, matrix_size, matrix_size)
        inv_list = list(matrixInverseApp.main_function(final_matrix, matrix_size, eps))

        norms = list()
        norms.append(matrixInverseApp.get_norm(final_matrix, inv_list[0], matrix_size))
        norms.append(matrixInverseApp.get_norm(final_matrix, inv_list[2], matrix_size))
        norms.append(matrixInverseApp.get_norm(final_matrix, inv_list[4], matrix_size))

        return render_template('matrixInverse/matrixInverseResult.html', NMAX=matrix_size, init_matrix=final_matrix,
                               inv_matrixes=inv_list, norms=norms)


@app.route('/eigenvaluesSVD')
def eigenvalues_SVD():
    return render_template('eigenvaluesSVD/eigenvaluesSVD.html')


@app.route('/submitEigenvaluesSVD', methods=['POST'])
def submit_eigenvalues_SVD():
    row = request.form['inputRow']
    column = request.form['inputColumn']
    matrix = request.form['inputMatrix']

    row = int(row)
    column = int(column)

    result = eingenvaluesSVD.main_function(min(row, column), min(row, column))
    final_matrix = util.get_processed_standard_matrix(matrix, row, column)
    result_info = eingenvaluesSVD.SVD_get_info(final_matrix, 2)
    return render_template('eigenvaluesSVD/eigenvaluesSVDResult.html', NMAX=min(row, column), eigenvalues=result,
                           svd_info=result_info, row=row, col=column)


@app.route('/functions')
def functions():
    return render_template('functions/functions.html')


@app.route('/submitFunctions', methods=['POST'])
def submit_functions():
    coeff = request.form['inputCoeff']
    coeff = coeff.split()

    coeff = [float(x) for x in coeff]
    result = functionsApp.main_function(coeff)

    return render_template('functions/functionsResult.html', result=result, lg=len(result[1]))



if __name__ == '__main__':
    app.run()
