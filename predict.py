import joblib
import numpy as np

model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def convert(value, min_val, max_val):
    return (value / 100) * (max_val - min_val) + min_val

print("Enter all values between 0 and 100")

crim = convert(float(input("Crime rate: ")), 0, 90)
zn = convert(float(input("Residential land: ")), 0, 100)
indus = convert(float(input("Industrial proportion: ")), 0, 30)

chas_input = float(input("Near river (0-100): "))
chas = 1 if chas_input > 50 else 0

nox = convert(float(input("Pollution level: ")), 0.3, 0.9)
rm = convert(float(input("Room rating: ")), 3, 9)
age = convert(float(input("House age: ")), 0, 100)
dis = convert(float(input("Distance rating: ")), 1, 12)
rad = convert(float(input("Highway access: ")), 1, 24)
tax = convert(float(input("Tax level: ")), 180, 720)
ptratio = convert(float(input("School quality inverse: ")), 12, 22)
b = convert(float(input("Population index: ")), 0, 400)
lstat = convert(float(input("Lower status %: ")), 1, 40)

new_house = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])

new_house_scaled = scaler.transform(new_house)
new_house_pca = pca.transform(new_house_scaled)

prediction = model.predict(new_house_pca)

print("Predicted House Price:", prediction[0])

