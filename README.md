# Data-science-with-python-Task-6-Time-Series-Analysis
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load and Inspect Dataset
data = pd.read_csv("C:/Users/hp/Documents/heart_disease.csv")  

print(data.head())  # Display first few rows

print("\nMissing Values:")
print(data.isnull().sum())

# Step 3: Preprocess Data
X = data.drop(columns=['Heart Disease'])  # Features
y = data['Heart Disease']  # Target

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Step 4: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 5: Evaluate Model Performance
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["No Disease", "Disease"], 
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()# Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Step 3: Load and Inspect Dataset
data = pd.read_csv("C:/Users/hp/Documents/sales_data.csv", parse_dates=['Date'], index_col='Date')

# Explicitly set frequency to daily to suppress warning
data.index = pd.DatetimeIndex(data.index).to_period('D')

# Display first few rows
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Visualize Sales Trends
plt.figure(figsize=(10, 5))
plt.plot(data.index.to_timestamp(), data['Sales'], label="Sales Data", color='Blue')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend Over Time")
plt.legend()
plt.show()

# Step 5: Check Stationarity (Dickey-Fuller Test)
result = adfuller(data['Sales'])
print(f"\nADF Statistic: {result[0]}")
print(f"P-Value: {result[1]}")

if result[1] > 0.05:
    print("The data is non-stationary. Differencing is needed.")
else:
    print("The data is stationary.")

# Step 6: Build ARIMA Model (p=2, d=1, q=2)
model = ARIMA(data['Sales'], order=(2, 1, 2))
model_fit = model.fit()

# Forecast next 10 periods
forecast = model_fit.forecast(steps=10)
print("\nForecasted Sales for next 10 periods:\n", forecast)

# Step 7: Visualize Forecast
# Convert period index to timestamp for plotting
future_dates = pd.date_range(start=data.index[-1].to_timestamp(), periods=11, freq='D')[1:]

plt.figure(figsize=(10, 5))
plt.plot(data.index.to_timestamp(), data['Sales'], label="Actual Sales", color='green')
plt.plot(future_dates, forecast, label="Forecast", color='black')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecast using ARIMA")
plt.legend()
plt.show()
