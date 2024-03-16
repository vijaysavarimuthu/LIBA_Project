import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from sklearn.metrics import precision_score,recall_score
import os
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns',None)

#----------------------------------------------------------------------------------------------
#Reading CSV File

df = pd.read_csv(r"C:\Users\Gladis M\Desktop\anand temp\LIBA-DS\Project\heart_2022_with_nans.csv")

#----------------------------------------------------------------------------------------------

# Removing Duplicates
duplicated=df.duplicated()


#---------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#READING THE DATASET
df = pd.read_csv(r"C:\Users\Gladis M\Desktop\anand temp\LIBA-DS\Project\heart_2022_with_nans.csv")
#datasets.head()
#------------------------------------------------------------------------------------
#FILL MISSING VALUES
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df),
columns=df.columns)
#print(df.isnull().sum())
#-------------------------------------------------------------------------------------------
#SPLIT CATEGORICAL & NUMIRICAL COLUMNS
cat_data=df.select_dtypes(include='object')
num_data=df.select_dtypes(exclude='object')


#-------------------------------------------------------------------------------------
label_encoder = preprocessing.LabelEncoder()
for c in cat_data:
    df[c]= label_encoder.fit_transform(df[c])
    df[c].unique()
#---------------------------------------------------------------------------------------------
# Title of the web app
st.title('Machine Learning Input App')



X = df[['GeneralHealth','SleepHours','RemovedTeeth','HadDiabetes','DifficultyWalking','SmokerStatus','WeightInKilograms']]
y = df[['HadHeartAttack']]
class_names = df[['HadHeartAttack']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Function to get user input
def get_user_input():
    GeneralHealth = st.text_input(' GeneralHealth Enter 0 or 1')
    SleepHours = st.text_input('Enter sleephours','0')
    RemovedTeeth = st.text_input('RemovedTeeth Enter 0 or 1')
    HadDiabetes = st.text_input('HadDiabetes Enter 0 or 1')
    DifficultyWalking = st.text_input(' DifficultyWalking Enter 0 or 1')
    SmokerStatus = st.text_input('SmokerStatus Enter 0 or 1')
    WeightInKilograms = st.text_input('WeightInKilograms','0')
    SleepHours = float(SleepHours)
    data = {'GeneralHealth': str(GeneralHealth),
            'SleepHours': float(SleepHours),
            'RemovedTeeth': str(RemovedTeeth),
            'HadDiabetes': str(HadDiabetes),
            'DifficultyWalking':str(DifficultyWalking),
            'SmokerStatus': str(SmokerStatus),
            'WeightInKilograms': float(WeightInKilograms)}
    features = pd.DataFrame(data, index=[0])
    return features
#

# Define the Streamlit app
def main():

    st.title("Heart Attack Prediction")
    user_input = get_user_input()
    st.write(user_input)
    if st.button("Predict"):
        #print(user_input)
        # Update the model with new parameters
        # Display the user input
        #st.subheader('User Input:')
        y_pred = dt_classifier.predict(user_input)
        #st.write("Button clicked!")

        st.write('Heart Attack:',y_pred)
if __name__ == "__main__":
    main()