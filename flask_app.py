
# Machine learning course project deployment
import os
import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle

from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse

#from PyInstaller.utils.hooks import collect_submodules
#hidden_imports = collect_submodules('h5py')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return 'This is a machine learning course!'

@app.route('/random_forest',methods=['GET'])
def new_fund():
    #Get arguments
    parser=reqparse.RequestParser()
    parser.add_argument("s1")
    parser.add_argument("s2")
    parser.add_argument("s3")
    parser.add_argument("s4")
    parser.add_argument("s5")
    parser.add_argument("s6")

    args = parser.parse_args()

    # Implement ML code to production

    df_creditcarddata = pd.read_csv("/home/KrishnanProf/mysite/UCI_Credit_Card.csv")

    # Assigning labels to features to make interpretation easier
    # And fix datatypes
    GenderMap = {2:'female', 1:'male'}
    MarriageMap = {1:'married', 2:'single', 3:'other', 0: 'other'}
    EducationMap = {1:'graduate school', 2:'university', 3:'high school', 4:'others', 5:'unknown', 6:'unknown', 0:'unknown'}


    df_creditcarddata['SEX'] = df_creditcarddata.SEX.map(GenderMap)
    df_creditcarddata['MARRIAGE'] = df_creditcarddata.MARRIAGE.map(MarriageMap)
    df_creditcarddata['EDUCATION'] = df_creditcarddata.EDUCATION.map(EducationMap)
    df_creditcarddata['PAY_0'] = df_creditcarddata['PAY_0'].astype(str)
    df_creditcarddata['PAY_2'] = df_creditcarddata['PAY_2'].astype(str)
    df_creditcarddata['PAY_3'] = df_creditcarddata['PAY_3'].astype(str)
    df_creditcarddata['PAY_4'] = df_creditcarddata['PAY_4'].astype(str)
    df_creditcarddata['PAY_5'] = df_creditcarddata['PAY_5'].astype(str)
    df_creditcarddata['PAY_6'] = df_creditcarddata['PAY_6'].astype(str)

    # Split the target variables
    predictor= df_creditcarddata.iloc[:, df_creditcarddata.columns != 'default.payment.next.month']
    target= df_creditcarddata.iloc[:, df_creditcarddata.columns == 'default.payment.next.month']

    # save all categorical columns in list
    categorical_columns = [col for col in predictor.columns.values if predictor[col].dtype == 'object']

    # dataframe with categorical features
    df_categorical = predictor[categorical_columns]

    # dataframe with numerical features
    df_numeric = predictor.drop(categorical_columns, axis=1)

    # Using pandas.get_dummies function to Convert categorical variable into dummy/indicator variables
    dummy_code_cat_vars  = pd.get_dummies(df_categorical,drop_first=True)

    # using concat function we merging two dataframe for furthere analysis
    df_predictor = pd.concat([df_numeric, dummy_code_cat_vars], axis=1)

    df_predictor=pd.concat([df_numeric["BILL_AMT1"],df_numeric["BILL_AMT2"],df_numeric["BILL_AMT3"],df_numeric["BILL_AMT4"],
                            df_numeric["BILL_AMT5"],df_numeric["BILL_AMT6"]], axis=1)


    #Let us now split the dataset into train & test
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train, y_test = train_test_split(df_predictor, target, test_size = 0.30, random_state=0)
    print("x_train ",X_train.shape)
    print("x_test ",X_test.shape)
    print("y_train ",y_train.shape)
    print("y_test ",y_test.shape)

    X_train.head()


    # Create sample to test in production
    X_test.to_csv("production_sample.csv",index=None)

    # Standarize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))

    # Load the ML model
    filename = '/home/KrishnanProf/mysite/finalized_model.model'
    loaded_model = pickle.load(open(filename,'rb'))

    df_new = [[args["s1"],args["s2"],args["s3"], args["s4"],args["s5"],args["s6"]]]
    X_test_scaled_new = pd.DataFrame(scaler.transform(df_new))
    predicted = loaded_model.predict(X_test_scaled_new)
    output = {}
    # Just a simple hack for now.
    print(predicted)
    output['Predicted class']= float(predicted[0])
    return jsonify(output)













