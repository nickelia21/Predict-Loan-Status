# Predict-Loan-Status-Flask-App
Launch machine learning models into production using flask, docker etc.

## Machine Learning in Finance Class Project 5
## - Our model predicts the loan status for an individual given user inputs

# 1. Predict Loan Status

## Environment and tools
1. scikit-learn
2. pandas
3. numpy
4. flask

## Installation

`pip3 install scikit-learn pandas numpy flask`

`python3 model.py`

`python3 app.py`

![Logo](logo.png)

# 2. Predict House Prices

Download our dataset from [here](www.kaggle.com/dataset/f9d3194d14ba42445cdcc507e5d89f0c375b29695ad1cdb4b4a42a65e22c443b).

## Environment and tools
1. scikit-learn
2. pandas
3. numpy
4. flask
5. docker

## Installation

`curl -X POST -H "Content-Type: application/json" -d @to_predict_json.json http://localhost:8080/predict_price`

where `to_predict.json` contains:

`{"grade":9.0,"lat":37.45,"long":12.09,"sqft_living":1470.08,"waterfront":0.0,"yr_built":2008.0}`

Output:

```
{
  "predict cost": 1022545.34768284
}
```

## Citing

```
@misc{Nick:2020,
  Author = {Nick Elia},
  Title = {Predict-Loan-Status-Flask-App},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/nickelia21/Predict-Loan-Status.git}}
}
```