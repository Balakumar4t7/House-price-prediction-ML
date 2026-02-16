import joblib
import numpy as np

model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

new_house = np.array([[0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]])

new_house_scaled = scaler.transform(new_house)
new_house_pca = pca.transform(new_house_scaled)

prediction = model.predict(new_house_pca)

print("Predicted House Price:", prediction[0])
