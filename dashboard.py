import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Dashboard", layout="wide")

# Title
st.title("🩺 Diabetes Prediction and Analysis")

# Load data
df = pd.read_csv("data/Healthcare-Diabetes.csv")

st.markdown("## 📊 Filter Data")

# Filters in two columns (not sidebar)
col1, col2 = st.columns(2)

with col1:
    age_range = st.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (25, 50))

with col2:
    selected_preg = st.selectbox("Number of Pregnancies", sorted(df["Pregnancies"].unique()))

# Apply filters
filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) & (df["Pregnancies"] == selected_preg)]

st.markdown("---")
st.markdown("### 🧮 Summary Metrics")

# Metrics in three columns
m1, m2, m3 = st.columns(3)

m1.metric("Avg Glucose", round(filtered_df["Glucose"].mean(), 2))
m2.metric("Avg BMI", round(filtered_df["BMI"].mean(), 2))
m3.metric("Diabetes Cases", int(filtered_df["Outcome"].sum()))

st.markdown("---")
st.markdown("### 🔍 Glucose vs BMI")

# Scatter Plot
fig, ax = plt.subplots()
ax.scatter(filtered_df["Glucose"], filtered_df["BMI"], alpha=0.7, color="teal")
ax.set_xlabel("Glucose")
ax.set_ylabel("BMI")
ax.set_title("Glucose vs BMI (Filtered Data)")
st.pyplot(fig)

st.markdown("---")
st.markdown("### 🤖 Predict Diabetes Using Logistic Regression")

# Model training (on full dataset)
X = df.drop(["Id", "Outcome"], axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Expandable section for model info
with st.expander("📈 View Model Accuracy"):
    st.write(f"Model Accuracy on Test Data: **{round(accuracy * 100, 2)}%**")

st.caption("Developed using Streamlit and scikit-learn")

if __name__ == "__main__":
    main(st.title("Diabetes Prediction and Analysis Dashboard"))
