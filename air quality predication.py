import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Sample dataset (can be replaced with actual AQI data)
def load_dataset():
    # Simulated dataset: CO, NOx, SO2, O3, PM2.5, PM10, AQI Level
    data = {
        'CO': np.random.uniform(0.2, 5.0, 500),
        'NOx': np.random.uniform(10, 200, 500),
        'SO2': np.random.uniform(1, 30, 500),
        'O3': np.random.uniform(10, 120, 500),
        'PM2.5': np.random.uniform(5, 200, 500),
        'PM10': np.random.uniform(10, 300, 500),
    }

    df = pd.DataFrame(data)

    # Generating AQI level labels based on PM2.5 (simplified logic)
    def aqi_level(pm25):
        if pm25 <= 50:
            return "Good"
        elif pm25 <= 100:
            return "Moderate"
        elif pm25 <= 150:
            return "Unhealthy for Sensitive Groups"
        elif pm25 <= 200:
            return "Unhealthy"
        else:
            return "Very Unhealthy"

    df['AQI_Level'] = df['PM2.5'].apply(aqi_level)
    return df

# Step 2: Train the model
def train_model(df):
    X = df[['CO', 'NOx', 'SO2', 'O3', 'PM2.5', 'PM10']]
    y = df['AQI_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Print metrics
    y_pred = clf.predict(X_test)
    print("Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    return clf

# Step 3: Get user input
def get_user_input():
    print("\nEnter the following pollutant concentrations:")
    CO = float(input("CO (mg/m³): "))
    NOx = float(input("NOx (ppb): "))
    SO2 = float(input("SO2 (ppb): "))
    O3 = float(input("O3 (ppb): "))
    PM25 = float(input("PM2.5 (µg/m³): "))
    PM10 = float(input("PM10 (µg/m³): "))

    return pd.DataFrame([[CO, NOx, SO2, O3, PM25, PM10]],
                        columns=['CO', 'NOx', 'SO2', 'O3', 'PM2.5', 'PM10'])

# Step 4: Predict AQI level
def predict_aqi(model, user_input_df):
    prediction = model.predict(user_input_df)
    print(f"\nPredicted Air Quality Level: {prediction[0]}")

# Main function
def main():
    df = load_dataset()
    model = train_model(df)
    user_input_df = get_user_input()
    predict_aqi(model, user_input_df)

if _name_ == "_main_":
    main()
