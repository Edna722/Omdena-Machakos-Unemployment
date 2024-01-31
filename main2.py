from flask import Flask, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder= 'templates')

# Load the data and models
def load_data():
    df = pd.read_csv("C:\\Users\\wanji\\Desktop\\Omdena\\Omdena-Machakos-Unemployment\\Feature_Engineered_Dataset (1).csv")  # Update with the correct path
    return df

# Load models using joblib
new_path = "C:/Users/wanji/Desktop/Omdena/Omdena-Machakos-Unemployment/"
scaler_path = "C:/Users/wanji/Desktop/Omdena/Omdena-Machakos-Unemployment/scaler.pkl"

# Load scaler
scaler = joblib.load(scaler_path)

model_files = [
    "catboostcoreCatBoostRegressorobjectatxfbb.pkl",
    "CatBoostRegressor.pkl",
    "ExtraTreesRegressor.pkl",
    "RandomForestRegressor.pkl",
]

models = []

for model_file in model_files:
    try:
        with open(new_path + model_file, "rb") as file:
            model = joblib.load(file)
            models.append(model)
            print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")

# ... (other functions and model loading)

# Define your features
features = [
    'Real_GDP_Ksh',
    'Population_Growth',
    'Female_Labor_Participation',
    'Male_Labor_Participation',
    'Education_Expenditure_Ksh',
    'Inflation',
    'Dollar_Rate',
    'Labor_Total_Population_Ratio',
    'Urban_Population_Growth_Income_Per_Capita_Growth_Ratio'
]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html', features=features)  # Pass features to the template

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for feature in features:
        input_data[feature] = request.form[feature]

    input_df = pd.DataFrame([input_data])
    input_df = input_df.apply(pd.to_numeric, errors='coerce')

    df_scaled = scaler.transform(input_df)

    model_name = request.form['model_name']
    selected_model = models[0]

    if model_name == "CatBoost":
        selected_model = models[0]
    elif model_name == "Extra Trees":
        selected_model = models[1]
    elif model_name == "Random Forest":
        selected_model = models[2]

    prediction = predict_unemployment(selected_model, df_scaled)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
