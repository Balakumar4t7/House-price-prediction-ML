# 🏠 House Price Prediction using PCA and Linear Regression

## 📌 Project Overview
This project builds a Machine Learning model to predict house prices using:

- Principal Component Analysis (PCA) for dimensionality reduction
- Linear Regression for prediction
- Model evaluation using R² Score and Mean Squared Error

The model is trained on a housing dataset and can predict prices for new input values.

---

## 📊 Dataset Description

The dataset used is the Boston Housing dataset, containing 13 features related to housing characteristics.

### Key Features:
- CRIM – Crime rate
- ZN – Residential land proportion
- INDUS – Industrial area proportion
- CHAS – Charles River dummy variable
- NOX – Nitric oxide concentration
- RM – Average number of rooms
- AGE – Proportion of old houses
- DIS – Distance to employment centers
- RAD – Accessibility to highways
- TAX – Property tax rate
- PTRATIO – Pupil-teacher ratio
- B – Proportion related to population
- LSTAT – Percentage of lower status population

### Target Variable:
- MEDV – Median house value (Price)

---

## ⚙️ Project Workflow

1. Load dataset using Pandas.
2. Handle missing values.
3. Split data into training and testing sets.
4. Standardize features using StandardScaler.
5. Apply PCA to reduce 13 features to 5 principal components.
6. Train Linear Regression model.
7. Evaluate model using:
   - R² Score
   - Mean Squared Error
8. Save trained model using Joblib.
9. Predict new house prices using `predict.py`.

---

## 📈 Output Explanation

### 1️⃣ R² Score
Measures how well the model explains variance in house prices.

- Value closer to 1 → Better model performance
- In this project: ~0.58 (moderate accuracy)

### 2️⃣ Mean Squared Error (MSE)
Measures average squared difference between actual and predicted prices.

Lower MSE → Better prediction accuracy.

### 3️⃣ Scatter Plot
Shows Actual Price vs Predicted Price.
If predictions are accurate, points align closely along a straight line.

---
## 🚀 How to Run

### Step 1: Train Model

```bash
python house_model.py
```

This will:
- Train the model  
- Display R² score and Mean Squared Error  
- Save model files (.pkl)  

---

### Step 2: Predict New House Price

```bash
python predict.py
```

This uses the trained model to predict house price for new input values.

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  

Author: Balakumar
