from flask import Flask, render_template, url_for, redirect, request, session
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, validators
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
    # variable_1 = IntegerField('variable 1', [validators.NumberRange(min=0, max=99, message='Must be between 0 and 99!'),
    #                                     validators.InputRequired('Required Field!')])
    variable_1 = IntegerField('Variable 1', [validators.InputRequired('Required Field!')])
    variable_2 = IntegerField('Variable 2', [validators.InputRequired('Required Field!')])
    variable_3 = IntegerField('Variable 3', [validators.InputRequired('Required Field!')])
    variable_4 = IntegerField('Variable 4', [validators.InputRequired('Required Field!')])
    variable_5 = IntegerField('Variable 5', [validators.InputRequired('Required Field!')])
    variable_6 = IntegerField('Variable 6', [validators.InputRequired('Required Field!')])
    submit = SubmitField('Submit')


@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    form = CalcForm()
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    # output = round(prediction[0], 2)

    # return render_template('result.html', prediction_text='Loan status: $ {}'.format(output))

    if form.validate_on_submit():
        session['variable_algo'] = test_algo(form.variable_1.data, form.variable_2.data, form.variable_3.data,
                                        form.variable_4.data, form.variable_5.data, form.variable_6.data)
        return redirect(url_for('result'))

    return render_template('calculator.html', form=form)


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
