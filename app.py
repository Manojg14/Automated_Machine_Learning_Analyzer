from flask import Flask, render_template, request
from model import run_pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    target_column = request.form['target']

    if file:
        result = run_pipeline(file, target_column)
        return render_template('index.html', result=result)

    return "Error uploading file"


if __name__ == '__main__':
    app.run(debug=True)