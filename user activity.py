import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
df = pd.read_csv("user_activity_large_dataset.csv")

# Step 2: Data Preprocessing
# Convert 'date' to datetime format and extract useful features
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek  # Extract day of the week

# Define the target variable (binary classification)
df["task_completed"] = (df["task_completion_time (minutes)"] <= 45).astype(int)

# Select features and target
X = df.drop(columns=["user_id", "date", "task_completion_time (minutes)", "task_completed"])
y = df["task_completed"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Model Development
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

# Step 5: Generate Predictions
# Sample input for prediction
sample_data = np.array([[120, 8000, 250, 7.5, 8, 300, 70, 2.5, 5, 2]])  # Example feature values
sample_data_scaled = scaler.transform(sample_data)

# Generate prediction
sample_prediction = model.predict(sample_data_scaled)

# Interpret result
prediction_result = "Task Likely to be Completed" if sample_prediction[0] == 1 else "Task Unlikely to be Completed"
print("Sample Prediction:", prediction_result)
