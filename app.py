
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.title('Shopping Purchase Prediction (Decision Tree)')
df = pd.read_csv('shopping_patterns_dataset.csv')

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

X = df.drop('bought', axis=1)
y = df['bought']
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X, y)

age = st.number_input('Age', 18, 80, 25)
gender = st.selectbox('Gender', ['Male','Female'])
income = st.number_input('Income', 10000, 200000, 50000)
time_spent = st.number_input('Time Spent (mins)', 1.0, 60.0, 5.0)
pages_viewed = st.number_input('Pages Viewed', 1, 30, 5)
gender_value = 1 if gender=='Male' else 0

data = np.array([[age, gender_value, income, time_spent, pages_viewed]])

if st.button('Predict'):
    pred = dt.predict(data)[0]
    st.success('Customer will BUY!' if pred==1 else 'Customer will NOT buy.')
