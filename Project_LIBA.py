import numpy as np
import pandas as pd
import streamlit as st
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
from sklearn.metrics import precision_score,recall_score
import os
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns',None)

#----------------------------------------------------------------------------------------------
#Reading CSV File

df = pd.read_csv("heart_2022_with_nans.csv")

#----------------------------------------------------------------------------------------------

# Removing Duplicates
duplicated=df.duplicated()


#---------------------------------------------------------------------------------------------


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



X = df[['HadAngina','SleepHours','HadStroke','AgeCategory','DifficultyWalking','SmokerStatus','WeightInKilograms']]
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
    HadAngina = st.text_input(' HadAngina Enter 0 or 1')
    SleepHours = st.text_input('Enter sleephours','0')
    HadStroke = st.text_input('HadStroke Enter 0 or 1')
    AgeCategory = st.text_input('AgeCategory')
    DifficultyWalking = st.text_input(' DifficultyWalking Enter 0 or 1')
    SmokerStatus = st.text_input('SmokerStatus Enter 0 or 1')
    WeightInKilograms = st.text_input('WeightInKilograms','0')

    data = {'HadAngina': str(HadAngina),
            'SleepHours': float(SleepHours),
            'HadStroke': str(HadStroke),
            'AgeCategory': str(AgeCategory),
            'DifficultyWalking':str(DifficultyWalking),
            'SmokerStatus': str(SmokerStatus),
            'WeightInKilograms': float(WeightInKilograms)}
    features = pd.DataFrame(data, index=[0])
    return features
#

# Define the Streamlit app
def main():

    st.title("Heart Attack Prediction")
    with st.sidebar:
            st.write("Instruction to enter the data")
            st.write("Hadangina",":" ,"HadStroke",":" ,"DifficultyWalking",":::", "No",":","0","-","Yes",":","1")
            st.write("SmokerStatus",":","Current smoker - now smokes every day",":","0","-","Current smoker - now smokes some days",":","1","-","Former smoker",":","2","-","Never smoked",":","3")
            st.write("AgeCategory",":","18 to 24",":","0","-","25 to 29",":","1","-","30 to 34",":","2","-","35 to 39",":","3","-","40 to 44",":","4","-","45 to 49",":","5","-","50 to 54",":","6","-","55 to 59",":","7","-","60 to 64",":","8","-","65 to 69",":","9","-","70 to 74",":","10","-","75 to 79",":","11","-","80orolder",":","12")

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
