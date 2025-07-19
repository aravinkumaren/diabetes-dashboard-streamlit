import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

# Prediction function
def predict_diabetes(model, input_data):
    prediction = model.predict([input_data])
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Main dashboard logic
def main():
    st.set_page_config(page_title="Diabetes Dashboard", layout="centered")
    st.title("Diabetes Prediction and Analysis Dashboard")

    # Load data
    data = load_data()

    # Sidebar navigation
    menu = ["Overview", "Data Exploration", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Overview":
        st.subheader("Project Overview")
        st.write("""
        This dashboard is built to analyze and predict the likelihood of diabetes 
        based on diagnostic data. Using machine learning (Logistic Regression), 
        we aim to provide a quick prediction tool along with data insights.
        """)

        st.write("### Sample of the dataset")
        st.dataframe(data.head())

        st.write("### Summary Statistics")
        st.dataframe(data.describe())

    elif choice == "Data Exploration":
        st.subheader("Correlation Heatmap")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif choice == "Prediction":
        st.subheader("Enter Patient Data for Prediction")

        # User Inputs
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
        insulin = st.slider("Insulin", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.slider("Age", 18, 100, 30)

        input_data = [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]

        # Train model
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model trained with accuracy: {acc:.2f}")

        # Prediction
        if st.button("Predict"):
            result = predict_diabetes(model, input_data)
            st.subheader("Prediction Result:")
            st.info(f"The model predicts: **{result}**")

    st.caption("Developed using Streamlit and scikit-learn")

# Entry point
if __name__ == "__main__":
    main()
