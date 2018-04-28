from flask_wtf import FlaskForm
from wtforms import FloatField, SelectField, DateField, StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField('User Name', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember me', default=False)
    submit = SubmitField('Log In')


class UserForm(FlaskForm):
    date = DateField('Date',validators=[DataRequired()])
    description = StringField('Location')
    type = SelectField('Type', coerce=str, choices=[(u'1', u'purchases'), (u'2', u'payments'), (u'3', u'fees')])
    amount = FloatField('Amount', validators=[DataRequired()])
    submit = SubmitField('Detect!')



#class UserForm(FlaskForm):
#   transaction = SubmitField('Find Your Transaction')