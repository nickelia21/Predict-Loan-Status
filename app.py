from flask import Flask, render_template, url_for, session, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return redirect(url_for('calculator'))

app.config['SECRET_KEY'] = 'mysecretkey'

def test_algo(variable_1, variable_2, variable_3, variable_4, variable_5, variable_6):
	dummy = variable_1 * variable_2 * variable_3 * variable_4 * variable_5 * variable_6
	return dummy

class CalcForm(FlaskForm):
    variable_1 = IntegerField()
    variable_2  = IntegerField()
    variable_3 = IntegerField()
    variable_4 = IntegerField()
    variable_5 = IntegerField()
    variable_6 = IntegerField()
    submit = SubmitField('Submit')

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():

	form = CalcForm()
 
	if form.validate_on_submit():

		session['variable_algo'] = test_algo(form.variable_1.data, form.variable_2.data, 
			form.variable_3.data, form.variable_4.data, form.variable_5.data, form.variable_6.data)

		return redirect(url_for('result'))

	return render_template('calculator.html', form=form)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

