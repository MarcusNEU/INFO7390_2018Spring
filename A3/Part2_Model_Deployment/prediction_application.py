from flask import Flask, request, render_template
import data_prediction
from common.custom_expections import BaseError

app = Flask(__name__)
UPLOAD_FOLDER = './upload_files'
OUTPUT_FOLDER = './static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/', methods=['GET'])
def upload_file():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def main():
    try:
        file_uploaded = request.files['upload_file']
        preview_parameter = data_prediction.data_processing(file_uploaded, UPLOAD_FOLDER)
        output_path = data_prediction.form_download_file(OUTPUT_FOLDER, preview_parameter[0], preview_parameter[1])
        return render_template('download.html', output_column=preview_parameter[0], output_row=preview_parameter[1],
                               total_rows=preview_parameter[2], output_path=output_path)
    except BaseError as e:
        print e.message
        raise BaseError(code=e.code, message=e.message)


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", message=e.message)

if __name__ == '__main__':
    app.run()
