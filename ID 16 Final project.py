import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = 'experiment_01.csv'
data = pd.read_csv(file_path)

# Define target variable and features
target = "M1_CURRENT_FEEDRATE"
X = data.drop(columns=[target, "Machining_Process"])  # Exclude target and categorical column
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)

# Visualization with regression line
plt.scatter(y_test, y_pred, alpha=0.7, label='Predictions')
plt.xlabel("Actual Feed Rate")
plt.ylabel("Predicted Feed Rate")
plt.title("Actual vs Predicted Feed Rate")

# Add linear regression line
slope, intercept = np.polyfit(y_test, y_pred, 1)  # Fit a linear regression line
line = slope * y_test + intercept
plt.plot(y_test, line, color='red', label='Regression Line')

plt.legend()
plt.show()
