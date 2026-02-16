import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

matplotlib.use("TkAgg")
data = pd.read_csv("house_price.csv")

data = data.fillna(data.mean(numeric_only=True))
data = pd.get_dummies(data, drop_first=True)

X = data.drop("MEDV", axis=1)
y = data["MEDV"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original number of features:", X_train.shape[1])

pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Reduced number of features after PCA:", X_train_pca.shape[1])

model = LinearRegression()
model.fit(X_train_pca, y_train)

predictions = model.predict(X_test_pca)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("R2 Score:", r2)
print("Mean Squared Error:", mse)

plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("output.png")


joblib.dump(model, "house_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

