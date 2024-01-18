import numpy as np
import pandas as pd
import pickle
import joblib
import streamlit as st
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\wanji\Desktop\Omdena\Omdena-Machakos-Unemployment\Feature_Engineered_Dataset (1).csv")
    return df

# Will only run once if already cached
df = load_data()

# Load the saved models as well as the scaler
new_path = "C:/Users/wanji/Desktop/Omdena/Omdena-Machakos-Unemployment/"

# Model files
model_files = ["RandomForestRegressor.pkl", "catboostcoreCatBoostRegressorobjectatxfbb.pkl", "ExtraTreesRegressor.pkl"]

# Load models using joblib
models = []

for model_file in model_files:
    try:
        with open(new_path + model_file, "rb") as file:
            model = joblib.load(file)
            models.append(model)
            print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")

# Save models using joblib
if len(models) == len(model_files):
    with open(new_path + "RandomForestRegressor.pkl", "wb") as model_file:
        joblib.dump(models[0], model_file)

    with open(new_path + "CatBoostRegressor.pkl", "wb") as model_file:
        joblib.dump(models[1], model_file)

    with open(new_path + "ExtraTreesRegressor.pkl", "wb") as model_file:
        joblib.dump(models[2], model_file)

    print("Models saved successfully.")
else:
    print("Error: Not all models were successfully loaded. Models not saved.")


#define the features
features = ['Real_GDP_Ksh',
 'Population_Growth',
 'Female_Labor_Participation',
 'Male_Labor_Participation',
 'Education_Expenditure_Ksh',
 'Inflation',
 'Dollar_Rate',
 'Labor_Total_Population_Ratio',
 'Urban_Population_Growth_Income_Per_Capita_Growth_Ratio']

def main(scaler = None):
    def predict_unemployment(model,input_data):
        if len(input_data) > 0:
            return model.predict(input_data)
        else:
                return "No data found"


    st.markdown("<h2 style='text-align: center; color: black;'>Predict the Unemployment Rate in Kenya </h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left; color: black;'>1. Enter a numeric value for each of the following features:</h1>", unsafe_allow_html=True)


    # Input data
    Real_GDP_Ksh = st.text_input("Real GDP (Ksh)")
    Population_Growth = st.text_input("Population Growth Percentage")
    Female_Labor_Participation = st.text_input("Female Labor Force Percentage")
    Male_Labor_Participation = st.text_input("Male Labor Force Percentage")
    Education_Expenditure_Ksh = st.text_input("Education Expense (Ksh)")
    Inflation = st.text_input("Inflation Rate")
    Dollar_Rate = st.text_input("Dollar Rate")
    Labor_Total_Population_Ratio = st.text_input("Labor to Total Population Ratio")
    Urban_Population_Growth_Income_Per_Capita_Growth_Ratio = st.text_input("Urban Population to Per Capita Income Growth Ratio")

    # Put all the features together in an array
    input_data = np.array([Real_GDP_Ksh, Population_Growth, Female_Labor_Participation,
                  Male_Labor_Participation, Education_Expenditure_Ksh, Inflation,
                  Dollar_Rate, Labor_Total_Population_Ratio,
                  Urban_Population_Growth_Income_Per_Capita_Growth_Ratio])#.reshape(1, -1)
    # convert to a dataframe
    input_df = pd.DataFrame([input_data], columns=features)
    input_df = input_df.apply(pd.to_numeric, errors='coerce')

    # Fit and scale/transform the entire DataFrame
    df_scaled = scaler.transform(input_df)
    

    # Selectbox for choosing the model
    st.markdown("<h6 style='text-align: left; color: black;'>2. Select a model:</h1>", unsafe_allow_html=True)
    model_name = st.selectbox("", ["Random Forest", "CatBoost", "Extra Trees"])
    
    # Button for picking the model
    if st.button("Predict the unemployment Rate"):
        st.info(f"Using {model_name}")
        st.info(f"Scaling Data")

        # Get the selected model
        selected_model = models[0]  # Default to the first model
        if model_name == "CatBoost":
            selected_model = models[1]
        elif model_name == "Extra Trees":
            selected_model = models[2]
        # Prediction
        prediction = predict_unemployment(selected_model, df_scaled)
        st.success(f"Predicted Unemployment Rate: {prediction[0]:.2f}")
        st.write("This is all Folks!!")
        


    st.markdown("""---""")
    st.markdown("<h5 style='text-align: center; color: black;'>Bonus: Analyzing Temporal Relationships Among Scaled Features</h1>", unsafe_allow_html=True)

    scaling = MinMaxScaler()

    # Scale the selected column
    df_scaled = df.copy()  # Make a copy of the original DataFrame
    df_scaled= pd.DataFrame(scaling.fit_transform(df),columns=df.columns)

    # Select two columns to plot
    selected_column_1 = st.selectbox("Select the first column to plot", df_scaled.columns[1:])
    selected_column_2 = st.selectbox("Select the second column to plot", df_scaled.columns[1:])

    # Melt the DataFrame to long format
    df_melted = df_scaled.melt(id_vars='Year', var_name='Variable', value_name='Value')

    # Filter the melted DataFrame based on the selected columns
    df_selected = df_melted[df_melted['Variable'].isin([selected_column_1, selected_column_2])]

    # Create a line chart
    chart = alt.Chart(df_selected).mark_line().encode(
        x='Year',
        y='Value',
        color='Variable',
        tooltip=['Year', 'Variable', 'Value']
    ).properties(
        width=700,
        height=400
    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

from app2 import main

if __name__ == '__main__':
    main()


