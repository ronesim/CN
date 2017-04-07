from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choleskiTool')
def choleskiDecomposition():
    return render_template('choleskiDec.html')


@app.route('/submitCholeski', methods=['POST'])
def submitCholeski():
    matrixDimension = request.form['inputMatrixSize']
    precision = request.form['inputPrecision']
    matrix = request.form['inputMatrix']
    print(matrixDimension + precision + matrix)
    return "TODO"


if __name__ == '__main__':
    app.run()
