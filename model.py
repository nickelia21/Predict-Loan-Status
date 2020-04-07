# Import packages
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import pickle

from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.activations import relu, elu
from keras.wrappers.scikit_learn import KerasClassifier
#from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

pd.set_option('max_columns', None)

# Concatenate all CSVs into one DataFrame
df1 = pd.read_csv("LoanStats3a_securev1.csv")
df2 = pd.read_csv("LoanStats3b_securev1.csv")
df3 = pd.read_csv("LoanStats3c_securev1.csv")
df4 = pd.read_csv("LoanStats3d_securev1.csv")
df = pd.concat([df1, df2, df3, df4])

# Drop duplicates
df.drop_duplicates(subset='id', inplace=True)
df.drop_duplicates(subset='member_id', inplace=True)

# Keep columns with sufficient data
df = df[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
         'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'pymnt_plan',
         'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
         'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
         'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
         'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'last_fico_range_high',
         'last_fico_range_low', 'acc_now_delinq', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']]

df.dropna(inplace=True)

# Adjusting non-categorical variables into integers/floats
df['term'] = df['term'].map(lambda x: x.rstrip(' months'))
df['term'] = df['term'].astype(float)
df['int_rate'] = df['int_rate'].map(lambda x: x.rstrip('%'))
df['int_rate'] = df['int_rate'].astype(float)
df['emp_length'] = df['emp_length'].str.replace('< 1', '1')
df['emp_length'] = df['emp_length'].map(lambda x: x.rstrip('+ years'))
df['emp_length'] = df['emp_length'].astype(float)
df['revol_util'] = df['revol_util'].map(lambda x: x.rstrip('%'))
df['revol_util'] = df['revol_util'].astype(float)

# Replace 'dates since' features (non-categorical) with days since 1/1/1970 and convert to int
# These columns were converted to date type in Excel workbook
df['issue_d'] = pd.to_datetime(df['issue_d']) - pd.datetime(1970, 1, 1)
df['issue_d'] = df['issue_d'].astype(str).str[:5]
df['issue_d'] = df['issue_d'].astype(int)
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d']) - pd.datetime(1970, 1, 1)
df['last_pymnt_d'] = df['last_pymnt_d'].astype(str).str[:5]
df['last_pymnt_d'] = df['last_pymnt_d'].astype(int)
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d']) - pd.datetime(1970, 1, 1)
df['last_credit_pull_d'] = df['last_credit_pull_d'].astype(str).str[:5]
df['last_credit_pull_d'] = df['last_credit_pull_d'].astype(int)

# Creating discrete value for whether loan status is default or chargeback
oneOptions = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
df['default_or_chargeback'] = df['loan_status'].apply(lambda x: 1 if x in oneOptions else 0)
df.drop('loan_status', axis=1, inplace=True)

df.drop(['funded_amnt', 'funded_amnt_inv'], axis=1, inplace=True)
df.drop(['installment', 'fico_range_low', 'out_prncp_inv', 'total_pymnt_inv',
         'total_rec_prncp', 'last_fico_range_low'], axis=1, inplace=True)

# Selecting features that will be known at the beginning of a loan origination
df = df[['loan_amnt', 'term', 'int_rate',
         'grade', 'sub_grade', 'emp_length',
         'home_ownership', 'annual_inc',
         'verification_status', 'purpose',
         'addr_state', 'dti', 'delinq_2yrs',
         'fico_range_high', 'inq_last_6mths',
         'open_acc', 'pub_rec', 'revol_bal',
         'revol_util', 'total_acc', 'last_fico_range_high',
         'acc_now_delinq', 'delinq_amnt', 'pub_rec_bankruptcies',
         'default_or_chargeback']]

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Split the target variables
predictor = df.iloc[:, df.columns != 'default_or_chargeback']
target = df.iloc[:, df.columns == 'default_or_chargeback']

# Save all categorical columns in list
categorical_columns = [col for col in predictor.columns.values if predictor[col].dtype == 'object']

# Dataframe with categorical features
df_categorical = predictor[categorical_columns]

# Dataframe with numerical features
df_numeric = predictor.drop(categorical_columns, axis=1)

# Using pandas.get_dummies function to convert categorical variable into dummy/indicator variables
dummy_code_cat_vars = pd.get_dummies(df_categorical, drop_first=True)

# Using concat function we merging two DataFrame for further analysis
df_predictor = pd.concat([df_numeric, dummy_code_cat_vars], axis=1)

# Split the dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(df_predictor, target, test_size=0.20, random_state=3,
                                                    stratify=target)

# Transform y_train and y_test into Series
y_train = pd.Series(y_train.index)
y_test = pd.Series(y_test.index)

print("x_train ", X_train.shape)
print("x_test ", X_test.shape)
print("y_train ", y_train.shape)
print("y_test ", y_test.shape)

# Standardize features
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))

X_train_scaled.columns = X_train.columns.values
X_test_scaled.columns = X_test.columns.values
X_train_scaled.index = X_train.index.values
X_test_scaled.index = X_test.index.values

X_train = X_train_scaled
X_test = X_test_scaled
X_train.head()

y_train['default_or_chargeback'].value_counts()


# MODEL

# Utility Functions

def plot_decision_boundary(func, X, y, figsize=(9, 6)):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    c = func(ab)
    cc = c.reshape(aa.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, ax = plt.subplots(figsize=figsize)
    contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)

    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(amin, amax)
    plt.ylim(bmin, bmax)


def plot_multiclass_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()], verbose=0)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0]) - 0.1, max(X[:, 0]) + 0.1))
    plt.ylim((min(X[:, 1]) - 0.1, max(X[:, 1]) + 0.1))
    plt.legend()


def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))


def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])


def plot_confusion_matrix(model, X, y):
    y_pred = model.predict_classes(X, verbose=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_pred)), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)
    plt.title('Confusion matrix of the classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')


def plot_compare_histories(history_list, name_list, plot_accuracy=True):
    dflist = []
    for history in history_list:
        h = {key: val for key, val in history.history.items() if not key.startswith('val_')}
        dflist.append(pd.DataFrame(h, index=history.epoch))

    historydf = pd.concat(dflist, axis=1)

    metrics = dflist[0].columns
    idx = pd.MultiIndex.from_product([name_list, metrics], names=['model', 'metric'])
    historydf.columns = idx

    plt.figure(figsize=(6, 8))

    ax = plt.subplot(211)
    historydf.xs('loss', axis=1, level='metric').plot(ylim=(0, 1), ax=ax)
    plt.title("Loss")

    if plot_accuracy:
        ax = plt.subplot(212)
        historydf.xs('acc', axis=1, level='metric').plot(ylim=(0, 1), ax=ax)
        plt.title("Accuracy")
        plt.xlabel("Epochs")

    plt.tight_layout()


def make_sine_wave():
    c = 3
    num = 2400
    step = num / (c * 4)
    np.random.seed(0)
    x0 = np.linspace(-c * np.pi, c * np.pi, num)
    x1 = np.sin(x0)
    noise = np.random.normal(0, 0.1, num) + 0.1
    noise = np.sign(x1) * np.abs(noise)
    x1 = x1 + noise
    x0 = x0 + (np.asarray(range(num)) / step) * 0.3
    X = np.column_stack((x0, x1))
    y = np.asarray([int((i / step) % 2 == 1) for i in range(len(x0))])
    return X, y


def make_multiclass(N=500, D=2, K=3):
    """
    N: number of points per class
    D: dimensionality
    K: number of classes
    """
    np.random.seed(0)
    X = np.zeros((N * K, D))
    y = np.zeros(N * K)
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        # radius
        r = np.linspace(0.0, 1, N)
        # theta
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    return X, y


# Addressing overfitting
# Dropout

model = Sequential()
model.add(Dense(12, input_shape=(148,), activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(36, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.1))
model.add(Dense(36, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(24, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(1, activation="sigmoid"))

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
ann = model.fit(x=X_train, y=y_train, verbose=0, epochs=50, callbacks=[early_stop], validation_split=0.3)
plot_loss(ann)

# Hyperparameter tuning
p = {'activation1': [relu, elu],
     # 'activation2':[relu, elu],
     'optimizer': ['Adam', "RMSprop"],
     # 'losses': ['logcosh', keras.losses.binary_crossentropy],
     'first_hidden_layer': [10, 8, 6],
     # 'batch_size': [100, 1000, 10000],
     'epochs': [10, 30]
     }

# AUC curve
y_pred_proba = model.predict(X_test).ravel()
fpr, tpr, t = metrics.roc_curve(y_test, y_pred_proba)  # 
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="AUC=" + str(auc))
plt.legend()
plt.show()

# Get classification report
y_pred_ann = model.predict_classes(X_test)
print(classification_report(y_test, y_pred_ann))

print(confusion_matrix(y_test, y_pred_ann))
plot_confusion_matrix(model, X_test, y_test)


# Function to create model, required for KerasClassifier
def create_model(first_hidden_layer, activation1):
    model = Sequential()
    model.add(Dense(first_hidden_layer, input_shape=(148,), activation=activation1))
    model.add(Dropout(0.1))
    model.add(Dense(36, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(36, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    # ann = model.fit(x=X_train, y=y_train, verbose=0, epochs=params['epochs'],
    #                 callbacks=[talos.utils.early_stopper(epochs=params['epochs'], mode='moderate', monitor='val_loss')],
    #                 validation_data=[x_val, y_val])
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=10, verbose=0)

# Define hyperparameter dict
p = {'activation1': [relu, elu],
     # 'activation2':[relu, elu],
     # 'optimizer': ['Adam', "RMSprop"],
     # 'losses': ['logcosh', keras.losses.binary_crossentropy],
     'first_hidden_layer': [10, 8, 6],
     # 'batch_size': [100, 1000, 10000],
     # 'epochs': [10, 30]
     }

grid = RandomizedSearchCV(estimator=model, param_distributions=p, n_iter=20, n_jobs=4, cv=3, random_state=111)
grid_result = grid.fit(X_train, y_train)

X_train.head()

# Build the random forest classifier and estimate
classifier = RandomForestClassifier(random_state=0, n_estimators=100, \
                                    criterion='entropy', max_leaf_nodes=30, n_jobs=-1)
model_RF = classifier.fit(X_train, y_train)

# Within test sample accuracy
acc_train_rf = round(classifier.score(X_train, y_train), 2) * 100
print(" Model accuracy within training data is : " + str(acc_train_rf) + "%")

y_pred_RF = model_RF.predict(X_test)
print(classification_report(y_test, y_pred_RF))

# Get feature importances
feature_importances = pd.Series(classifier.feature_importances_, index=X_train.columns)
feature_importances.nlargest(20).plot(kind='barh')
plt.show()

# Tune the hyperparameters of the RF estimator
# before HP tuning
classifier = RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy', max_leaf_nodes=30, n_jobs=-1)
model_RF = classifier.fit(X_train, y_train)

# With HPP tuning
# Create the HP grid
param_grid_new = {
    'bootstrap': [True],
    'max_depth': [80, 90],
    'max_features': [2],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100]
}

# Create a RF model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_new, cv=3, verbose=1)

# Fir the grid search model to data
grid_search_rf.fit(X_train, y_train)
print(grid_search_rf.best_params_)

best_grid = grid_search_rf.best_estimator_
print("Grid search accuracy:", round(best_grid.score(X_train, y_train) * 100, 2), "%")

print("Test sample Grid search accuracy:", round(best_grid.score(X_test, y_test) * 100, 2), "%")

AdaBoost = AdaBoostClassifier(n_estimators=100,
                              base_estimator=DecisionTreeClassifier(max_depth=1),
                              random_state=0)

model_AB = AdaBoost.fit(X_train, y_train)

# train accuracy
acc_adaboost = round(AdaBoost.score(X_train, y_train) * 100, 2)
print("Model accuracy in the training sample is: ", acc_adaboost, "%")

y_pred_AB = model_AB.predict(X_test)
print(classification_report(y_test, y_pred_AB))

X_test.head()
print(y_pred_AB)

cnf_matrix = confusion_matrix(y_test, y_pred_AB)
print(cnf_matrix)

cf_df = pd.DataFrame(cnf_matrix, columns=['0', '1'], index=['0', '1'])
plt.figure(figsize=(7, 5))
sns.set(font_scale=1.4)
sns.heatmap(cf_df, annot=True, fmt='5.0f')
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# # Implement XGBoost
# xg_model = XGBClassifier()
# xg_model.fit(X_train, y_train)
#
# # train accuracy
# acc_xgboost = round(xg_model.score(X_train, y_train) * 100, 2)
# print("Model accuracy in training sample is: ", acc_xgboost, "%")
#
# # test accuracy
# acc_xgboost_test = round(xg_model.score(X_test, y_test) * 100, 2)
# print("Model accuracy in test sample is: ", acc_xgboost_test, "%")
#
# y_pred_xg = xg_model.predict(X_test)
# print(classification_report(y_test, y_pred_xg))
#
# cnf_matrix = confusion_matrix(y_test, y_pred_xg)
# print(cnf_matrix)
#
# cf_df = pd.DataFrame(cnf_matrix, columns=['0', '1'], index=['0', '1'])
# plt.figure(figsize=(7, 5))
# sns.set(font_scale=1.4)
# sns.heatmap(cf_df, annot=True, fmt='5.0f')
# plt.title('Confusion matrix of the classifier')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# Load model into pickle file
pickle.dump(model_AB, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print('Model test prediction:', round(model.predict(1)[0], 3))
