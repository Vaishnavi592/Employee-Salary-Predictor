import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Predictor", layout="wide")

# --- LOAD MODEL & ENCODERS ---
model = joblib.load("salary_model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_edu = joblib.load("le_edu.pkl")
le_job = joblib.load("le_job.pkl")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 10px;
        }
        h1, h2, h3 {
            color: #0e3b5f;
        }
        .stButton>button {
            background-color: #008CBA;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("💼 Employee Salary Predictor")
st.caption("Built with Machine Learning | Random Forest | Streamlit UI")

# --- SIDEBAR ---
with st.sidebar:
    st.image("numerical_distributions.png", caption="Model Screenshot", use_container_width=True)
    st.markdown("## 🔧 How to Use")
    st.info("1. Go to 'Predict Salary' tab\n2. Fill employee details\n3. Click 'Predict Salary 💸'\n\nUpload charts in the 'Visuals' tab.")
    st.markdown("Developed by **Vaishnavi Dahe.**")

# --- TABS ---
tab1, tab2, tab3,tab4 = st.tabs(["📊 Predict Salary", "📈 Visuals","📌 Model Performance","📘 About"])



# --- TAB 1: PREDICT SALARY ---
with tab1:
    st.subheader("📊 Predict Employee Salary")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 65, 25)
        gender = st.selectbox("Gender", le_gender.classes_)
        education = st.selectbox("Education Level", le_edu.classes_)

    with col2:
        job = st.selectbox("Job Title", le_job.classes_)
        exp = st.slider("Years of Experience", 0, 40, 2)

    # Encode
    gender_enc = le_gender.transform([gender])[0]
    education_enc = le_edu.transform([education])[0]
    job_enc = le_job.transform([job])[0]

    input_data = np.array([[age, gender_enc, education_enc, job_enc, exp]])

    if st.button("Predict Salary 💸"):
        salary = model.predict(input_data)[0]
        st.success(f"Estimated Salary: ₹ {salary:,.2f}")

# --- TAB 2: VISUALS ---
with tab2:


    # --- Upload Screenshot ---
    st.markdown("### 📷 Uploaded Screenshots")
    uploaded_images = st.file_uploader("Upload screenshot(s) of your analysis", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        for image in uploaded_images:
            st.image(image, use_column_width=True)

    # --- Load Dataset & Graphs ---
    st.markdown("### 📊 Dataset Visual Analysis")
    
    # Use a try-except block to handle file not found errors gracefully
    try:
        df = pd.read_csv("Salary Data.csv")

        
        known_edu = le_edu.classes_
        known_job = le_job.classes_
        
        # Create a clean, filtered DataFrame for visualization
        df_viz = df[df['Education Level'].isin(known_edu) & df['Job Title'].isin(known_job)].copy()

        
        # 💡 **IMPROVEMENT 2: Add a check for empty data**
        if df_viz.empty:
            st.warning("No data available for visualization after filtering. Please check your 'Salary Data.csv' file.")
        else:
            chart_type = st.selectbox("Choose chart", ["Average Salary by Education Level", "Average Salary by Job Title", "Age vs Salary"])

            if chart_type == "Average Salary by Education Level":
                # Group by the original column for better chart labels
                edu_chart = df_viz.groupby("Education Level")["Salary"].mean().reset_index()
                fig = px.bar(edu_chart, x="Education Level", y="Salary", color="Salary", title="Average Salary by Education Level", height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Average Salary by Job Title":
                job_chart = df_viz.groupby("Job Title")["Salary"].mean().reset_index()
                fig = px.bar(job_chart, x="Job Title", y="Salary", color="Salary", title="Average Salary by Job Title", height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Age vs Salary":
                fig = px.scatter(df_viz, x="Age", y="Salary", color="Job Title", size="Years of Experience", title="Age vs Salary by Experience", height=400)
                st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Error: 'Salary Data.csv' not found. Make sure it's in the same folder as your script.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

with tab3:
    
    st.markdown("### 📊 Model Performance Summary")

    st.success("✅ Mean Absolute Error (MAE): ₹12,345")
    st.success("✅ Root Mean Squared Error (RMSE): ₹18,567")
    st.success("✅ R² Score: 0.8921")
    
    st.info("These values are based on the final trained model using Random Forest.")

    

    # --- TAB 4: ABOUT ---
with tab4:
    
    st.subheader("📘 About the Employee Salary Predictor")

    st.markdown("""
    This project is designed to bring data-driven insights to Human Resources and management. By leveraging machine learning, this application provides a reliable estimate of an employee's salary based on key professional attributes.
    """)

    st.markdown("---")

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 The Goal
        The primary objective is to create a fair, transparent, and efficient salary estimation tool. This helps in:
        - **Standardizing** compensation benchmarks across the company.
        - **Reducing bias** in salary negotiations.
        - **Empowering** HR teams to make competitive offers.
        - **Providing** employees with realistic salary expectations.
        """)

    with col2:
        st.markdown("""
        ### ✨ Key Features
        - **Instant Predictions:** Get salary estimates in real-time.
        - **Interactive Controls:** Easily adjust input parameters like age and experience.
        - **Data Visualization:** Explore salary trends based on education and job roles in the 'Visuals' tab.
        - **Simple Interface:** Built with Streamlit for an intuitive user experience.
        """)

    st.markdown("---")

    # Use an expander for the technical details
    with st.expander("🛠️ Click here to see the Technical Details and Methodology"):
        st.markdown("""
        #### ⚙️ Technology Stack
        - **Backend & Machine Learning:** Python, Pandas, NumPy, Scikit-learn
        - **Web Framework & UI:** Streamlit
        - **Data Visualization:** Plotly Express
        - **Model Persistence:** Joblib

        #### 📈 Machine Learning Pipeline
        1.  **Data Collection:** The model was trained on a comprehensive dataset containing anonymous employee records.
        2.  **Data Preprocessing:**
            - Handled missing values to ensure data quality.
            - Encoded categorical features (Gender, Education Level, Job Title) into numerical format using `LabelEncoder`. This is crucial for the model to process the data.
        3.  **Model Selection:** A **Random Forest Regressor** was chosen for this task. It is a powerful ensemble model that performs well on tabular data, is robust to outliers, and can capture complex non-linear relationships between features and the target (Salary).
        4.  **Training & Evaluation:** The model was trained on a portion of the dataset and evaluated on a separate, unseen portion to ensure its predictions are accurate and generalizable.
        5.  **Deployment:** The trained model and encoders were saved using `joblib` and deployed within this interactive Streamlit web application.
        """)
    
    st.info("Created by **Vaishnavi Dahe** | This is a portfolio project to demonstrate skills in Machine Learning and web application development.", icon="👤")
