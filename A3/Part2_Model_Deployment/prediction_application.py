from flask import Flask, request, render_template
import data_prediction
from common.custom_expections import BaseError
import logging
import os
from werkzeug import exceptions
import traceback

def setLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(os.getcwd(), 'log.txt'))
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s.%(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger

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
        metrics_parameter = data_prediction.unpickle_error_metrics()
        output_path = data_prediction.form_download_file(OUTPUT_FOLDER, preview_parameter[0], preview_parameter[1],
                                                         metrics_parameter[0], metrics_parameter[1])
        return render_template('download.html', metrics_column=metrics_parameter[0], metrics_row=metrics_parameter[1],
                               output_column=preview_parameter[0], output_row=preview_parameter[1],
                               total_rows=preview_parameter[2], output_path=output_path)
    except BaseError as e:
        print e.message
        setLogger().exception(e.message)
        raise BaseError(code=e.code, message=e.message)
    except:
        import traceback
        traceback.print_exc()
        return render_template("no_file.html")


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", message=e.message)

if __name__ == '__main__':
    app.run()

