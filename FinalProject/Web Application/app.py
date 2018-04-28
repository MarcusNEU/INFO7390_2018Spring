import os
from flask import Flask, render_template, redirect, url_for,flash,request
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, current_user
from flask_login import LoginManager
from flask_login import logout_user
from forms import LoginForm,UserForm
from flask_login import UserMixin
from flask_migrate import Migrate, MigrateCommand
from flask_script import Manager
import classification
import logging
from custom_expections import BaseError
import pandas as pd
from sklearn.svm import SVC
import scipy
from sklearn.ensemble import RandomForestClassifier

basedir = os.path.abspath(os.path.dirname(__file__))

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
OUTPUT_FOLDER = './static/output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = 'hard to guess string'
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)

# manage session
login_manager = LoginManager()
login_manager.session_protection = 'strong'
login_manager.login_view = 'login'
login_manager.init_app(app=app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Post(db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    date = db.Column(db.DateTime)
    description = db.Column(db.String(64))
    type = db.Column(db.Enum('purchases', 'payments', 'fees'))
    amount = db.Column(db.Numeric)

    def __repr__(self):
        return '<Role %r>' % self.id


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    users = db.relationship('User', backref='role', lazy='dynamic')

    def __repr__(self):
        return '<Role %r>' % self.name


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    password_hash = db.Column(db.String(128))
    posts = db.relationship('Post', backref='user', lazy='dynamic')

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def get_transaction(self):
        return Post.query.join(User, User.id == Post.user_id).filter(Post.user_id == self.id)

    def __repr__(self):
        return '<User %r>' % self.username


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user is not None and user.verify_password(form.password.data):
            login_user(user, form.remember_me.data)
            print'sucess'
            return redirect(url_for('index'))
            setLogger().info(form.username.date + 'login in ')
        flash('Wrong password')
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out')
    #setLogger().info(current_user.username + 'logout')
    return redirect(url_for('welcome'))


@app.route('/', methods=['GET', 'POST'])
def welcome():
    return render_template('welcome.html')


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = UserForm()
    result = 'Normal'
    if form.validate_on_submit():
        amount = form.amount.data
        time = form.date.data
        type_id = form.type.data
        id = current_user.id
        if type_id == 2:
            type = 'payments'
        elif type_id == 1:
            type = 'purchases'
        else:
            type = 'fees'
        if form.amount.data < 5:
            result = 'Fraud'
        flash(result)
        transaction = Post(date=time, type=type, amount=amount, user_id=id)
        if Post.query.filter_by(date=time, amount=amount).first():
            print 'transaction already exist'
        else:
            db.session.add(transaction)
            db.session.commit()
    return render_template('index.html', form=form)


@app.route('/choose', methods=['GET', 'POST'])
@login_required
def upload_file():
    try:
        if request.method == 'POST':
            file_uploaded = request.files['upload_file']
            preview_parameter = classification.data_processing(file_uploaded)
            output_path = classification.form_download_file(OUTPUT_FOLDER, preview_parameter[1])
            return render_template('download.html', output_row=preview_parameter[1], total_rows=preview_parameter[2]
                               , output_path=output_path, output_column=preview_parameter[0])
        else:
            return render_template('upload_test.html')
    except BaseError as e:
        print e.massage
        setLogger().exception(e.message)
        raise BaseError(code=e.code, message=e.message)
    except:
        import traceback
        traceback.print_exc()
        return render_template("no_file.html")


@app.route('/uploadforuser', methods=['GET', 'POST'])
@login_required
def user_upload_file():
    try:
        if request.method == 'POST':
            file_uploaded = request.files['upload_file']
            preview_parameter = classification.data_processing_user(file_uploaded)
            output_path = classification.form_download_file(OUTPUT_FOLDER, preview_parameter[1])
            return render_template('download_for_user.html', output_row=preview_parameter[1], total_rows=preview_parameter[2]
                               , output_path=output_path, output_column=preview_parameter[0])
        else:
            return render_template('upload_test.html')
    except BaseError as e:
        print e.massage
        setLogger().exception(e.message)
        raise BaseError(code=e.code, message=e.message)
    except:
        import traceback
        traceback.print_exc()
        return render_template("no_file.html")


if __name__ == '__main__':
    #manager.run()
    app.run("0.0.0.0", debug=True)