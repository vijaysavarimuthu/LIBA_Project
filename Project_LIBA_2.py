import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC ,SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
import os
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns',None)

#----------------------------------------------------------------------------------------------
#Reading CSV File

df = pd.read_csv(r"C:\Users\Gladis M\Desktop\anand temp\LIBA-DS\Project\heart_2022_with_nans.csv")
df.head()

#----------------------------------------------------------------------------------------------

# Removing Duplicates
duplicated=df.duplicated()
print(duplicated.sum())

#---------------------------------------------------------------------------------------------
# # Title of the web app
# st.title('Machine Learning Input App')
#
# # Sidebar for user input
# st.sidebar.header('User Input Parameters')
#
# # Function to get user input
# def get_user_input():
#     sepal_length = st.text_input('Sepal length', '5.4')
#     sepal_width = st.text_input('Sepal width', '3.4')
#     petal_length = st.text_input('Petal length', '1.3')
#     petal_width = st.text_input('Petal width', '0.2')
#     data = {'sepal_length': float(sepal_length),
#             'sepal_width': float(sepal_width),
#             'petal_length': float(petal_length),
#             'petal_width': float(petal_width)}
#     features = pd.DataFrame(data, index=[0])
#     return features
#
# # Load the iris dataset
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Train the model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
#
# # Get user input
# user_input = get_user_input()
#
# # Display the user input
# st.subheader('User Input:')
# st.write(user_input)
#
# # Make predictions
# prediction = model.predict(user_input)
#
# # Display the prediction
# st.subheader('Prediction:')
# st.write(iris.target_names[prediction[0]])
------------------------------------------------------------------------------------------------------------------------------------------------

