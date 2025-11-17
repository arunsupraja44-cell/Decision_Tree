import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
    .main {
        background-color: #F3F4F6;
    }
    .title {
        background: linear-gradient(90deg, #EC4899, #8B5CF6);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
    }
    .subheader-box {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #8B5CF6;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
    }
    .predict-box {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }
    .success {
        color: #10B981;
    }
    .error {
        color: #EF4444;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<div class='title'>🛍️ Shopping Purchase Prediction</div>", unsafe_allow_html=True)

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv("shopping_patterns_dataset.csv")

le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])

X = df.drop("bought", axis=1)
y = df["bought"]

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.info("This app predicts whether a customer will buy a product based on shopping behavior.")

show_info = st.sidebar.checkbox("Show Dataset Info")

if show_info:
    st.sidebar.write(df.head())
    st.sidebar.write("Shape:", df.shape)

# ---------------------- USER INPUTS ----------------------
st.markdown("<div class='subheader-box'><h4>🧾 Enter Customer Details</h4></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    income = st.number_input("Income (₹)", 10000, 200000, 50000)
    time_spent = st.number_input("Time Spent on Website (minutes)", 1.0, 60.0, 5.0)
    pages_viewed = st.number_input("Pages Viewed", 1, 40, 5)

gender_value = 1 if gender == "Male" else 0

input_data = np.array([[age, gender_value, income, time_spent, pages_viewed]])

# ---------------------- PREDICTION ----------------------
if st.button("🔮 Predict Purchase", use_container_width=True):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown("<div class='predict-box success'>🛒 Customer WILL Buy!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='predict-box error'>❌ Customer will NOT Buy.</div>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("<br><hr><p style='text-align:center;color:gray;'>✨ Developed with Supraja❤️ using Streamlit</p>", unsafe_allow_html=True)
