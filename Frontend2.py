import streamlit as st
import pandas as pd
import joblib
model = joblib.load("C:\\Users\\aadit\\OneDrive\\Desktop\\knn_heart_model.pkl")
scaler = joblib.load("C:\\Users\\aadit\\OneDrive\\Desktop\\heart_scaler.pkl")
expected_columns = joblib.load("C:\\Users\\aadit\\OneDrive\\Desktop\\heart_columns.pkl")
st.title("Heart Stroke Prediction by Aaditya")
st.markdown("Provide the following details to check your heart stroke risk:")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])  
if st.button("Predict"):
    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }


    # Create input dataframe
    input_df = pd.DataFrame([raw_input])


    # Handle expected_columns: It's crucial that `expected_columns` is the list of column names
    # the model was trained on. If `expected_columns` itself is a scaler object, that's an erro
    # The previous `if hasattr(expected_columns, 'get_feature_names_out'):` was trying to extra
    # feature names from a scaler, implying `expected_columns` might have been incorrectly load
    # Here, we assume `expected_columns` was correctly loaded as a list/array of column names.
    # If `expected_columns` is not a list/array of strings, further debugging of the `heart_col
    all_model_columns = expected_columns


    # Create a DataFrame with all expected columns, initialized to 0
    final_input_df = pd.DataFrame(0, index=[0], columns=all_model_columns)


    # Populate the final_input_df with actual input values
    for col in input_df.columns:
        if col in final_input_df.columns:
            final_input_df[col] = input_df[col].values


    # Scale the numerical features
    scaled_input = scaler.transform(final_input_df)


    # Make prediction
    prediction = model.predict(scaled_input)


    # Display result
    if prediction[0] == 1:
        st.write("There is a high risk of heart stroke. Please consult a doctor.")
    else:
        st.write("You have a low risk of heart stroke.")


# Run this command in your terminal instead:
# streamlit run app.py & npx localtunnel --port 8501





