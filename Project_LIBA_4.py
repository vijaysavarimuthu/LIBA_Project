import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBOOST": xgb.XGBClassifier(),
    "Logistics": LogisticRegression()
}
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
#print("categorical features: ", cat_data.columns.to_list())

#-------------------------------------------------------------------------------------
label_encoder = preprocessing.LabelEncoder()
for c in cat_data:
    df[c]= label_encoder.fit_transform(df[c])
    df[c].unique()
#print(df.head())
df.to_csv('encoder1.csv')
#----------------------------------------------------------------------------------------
X = df[['GeneralHealth','SleepHours','RemovedTeeth','HadDiabetes','DifficultyWalking','SmokerStatus','WeightInKilograms']]
y = df[['HadHeartAttack']]
class_names = df[['HadHeartAttack']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    #print(X_test)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Train the classifier
    score = clf.score(X_test, y_test)  # Evaluate its performance
    print(f"{name}: {score}")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print(f'Accuracy_score: {accuracy_score(y_test, y_pred)}')
    print(f'Precission_score: {precision_score(y_test, y_pred)}')
    print(f'Recall_score: {recall_score(y_test, y_pred)}')