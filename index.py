import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr


# Load the dataset
data=pd.read_csv(r"C:\xampp\htdocs\project\diabetes.csv")


# Split the dataset into features (X) and target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Data Augmentation using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)


# Train the model on the augmented training data
model = LogisticRegression()
model.fit(X_train_smote, y_train_smote)


from gradio import Slider, Number
def predict_diabetes(Preganancies,Glucose,BloodPressure,BMI,DiabetesPedigreeFunction,Age):
  # Set other attributes to zero
    SkinThickness = 0
    Insulin = 0
    # Scale the input values
    input_data = scaler.transform([[Preganancies,Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    # Make prediction
    prediction = model.predict(input_data)[0]
    return "you have HIGH CHANCE of being DIABETIC in the future maintain a healthy lifestyle and consult the physician.you dhould reduce the intake of sugar and increase the fiber content in your diet " if prediction == 1 else "you have LESS CHANCE of being DIABETIC continue to maintain a healthy lifestyle"
from gradio import Slider, Number
inputs = [
    gr.Slider(0, 100, "no.of.preganancies(number)"),
    gr.Number(label="Blood Glucose (mg/dL)"),
    gr.Number(label="Blood Pressure (mmHg)"),
    gr.Number(label="BMI"),
     gr.Number(label="Diabetes Pedigree Function(number)"),
    gr.Slider(0, 100, "age(number)")
]




# Specify the number output component
outputs = ["text"]
description = "For educational purposes only. Please consult a medical professional for personalized medical advice."




# Create the interface and launch it
interface = gr.Interface(fn=predict_diabetes, inputs=inputs, outputs=outputs,
    description=description)
interface.launch(debug=True)
