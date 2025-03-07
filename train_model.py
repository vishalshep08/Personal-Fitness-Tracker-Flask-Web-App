import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Sample dataset (Replace this with your actual dataset)
data = {
    "Duration": [30, 45, 60, 90, 120],
    "Heart_Rate": [80, 85, 90, 100, 110],
    "Body_Temp": [36.5, 37, 37.5, 38, 38.5],
    "Gender": [1, 0, 1, 0, 1],  # 1 for Male, 0 for Female
    "Age": [25, 30, 35, 40, 45],
    "BMI": [22, 24, 26, 28, 30],
    "Calories_Burned": [200, 300, 400, 600, 800],
}

df = pd.DataFrame(data)

# Features & target
X = df.drop(columns=["Calories_Burned"])
y = df["Calories_Burned"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "calories_model.pkl")

print("✅ Model trained and saved as 'calories_model.pkl'")
