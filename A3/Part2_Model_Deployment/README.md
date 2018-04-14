## Web Application for Machine Learning on Blog Feedback Dataset

#### This is the 2nd part of this assignment which is designed for application deployment. Flask is used to create the application.

<pre><code>
@app.route('/', methods=['GET'])
def upload_file():
    ***
</pre></code>

Access http://159.65.231.188:5000 will send a GET request to the application and a upload page will be returned where users can upload their file for prediction. Noticed that only json file will be accpeted.

---
<pre><code>
@app.route('/', methods=['POST'])
def main():
    try:
        calling function: data processing
        calling function: unpickle error metrics
        calling function: form download file
        ruturn: download page with preview tables
    except BaseError as e:
        print e.message
        raise BaseError(code=e.code, message=e.message)
    except:
        import traceback
        return render_template("no_file.html")
</pre></code>

After receiving a file from user, the application will run these three fuctions implemented in the lower layer. The code in upper layer is abstract without details as the content of "prediction_application.py" shows. If everything goes well, the code will run to return line and return the download page for the user with a preview of results and download button. Otherwise, errors like user uploaded a wrong file (not json), the uploaded json file is not valid for our model (not align with the data in our dataset), poor connection to S3 or any other deployment problem like unable to load the models will be captured here as a 'BaseError'.

---
<pre><code>
class BaseError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
</pre></code>

'BaseError' is nothing special but the common errors just like the ones listed in the last paragraph. And the 'self.message' will allow us to feedback the customized content which help users to understand what is wrong like "File Form Error! Only json File is Accepted!" instead of "XXX is not in the allowed_file". Other words, this class is designed to capture the common mistakes that users might make and guide the users to fix it. Since not uploading any files will cause ValueError in the back-end, another template for handling this kind of problem is simply created here to make things easier and will not return '400' error.

---
<pre><code>
@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", message=e.message)
</pre></code>

One last thing in the upper layer is the error handler. Only capturing the error will not do any helo so that we need a hanlder to actually do something: first it displays a pop-up window telling users what is going on here, then it returns the upload file page to users in the first stage.

---
In conclusion, the whole use case of the application will be looked like this in the upper layer:

![Aaron Swartz](https://raw.githubusercontent.com/MarcusNEU/INFO7390_2018Spring/master/A3/Part2_Model_Deployment/FlowChart.png)
---
The uploaded file and download file formed in the application will be stored in "upload" and "static/output" directory respectively. The implementation details of the application's fuctions are in the lower layer file ("data_prediction.py"):
https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/A3/Part2_Model_Deployment/data_prediction.py
