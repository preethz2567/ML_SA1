# ML_SA1
## AIM :

To analyze the Fuel Consumption dataset and build linear regression models to study the relationship between different features (like CYLINDERS, ENGINESIZE, FUELCONSUMPTION_COMB) and CO2 emissions, and evaluate models across different train-test splits.

```
Program Developed by : PREETHI D
Register Number : 212224040250
```

Equipments Required:

1.Hardware – PCs
2.Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:

1.Import necessary libraries like pandas, matplotlib, and sklearn.

2.Load the dataset (FuelConsumption.csv) and preview the data.

3.Plot scatter plots for:

  -Cylinders vs CO2 Emissions (green)
  
  -Cylinders vs CO2 Emissions & Engine Size vs CO2 Emissions
  
  -Cylinders, Engine Size, Fuel Consumption vs CO2 Emissions
  
4.Train a Linear Regression model using:

  -Independent variable: CYLINDERS, Dependent: CO2EMISSIONS
  
  -Independent variable: FUELCONSUMPTION_COMB, Dependent: CO2EMISSIONS

5.Evaluate models by calculating accuracy (R² score).

6.Perform model training on different train-test ratios (80/20, 70/30, 60/40) and record accuracy for each.

## PROGRAM / OUTPUT:

1.Create a scatter plot between cylinder vs Co2Emission (green color)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("FuelConsumption.csv")

plt.figure(figsize=(8,6))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Cylinders vs CO2 Emissions")
plt.grid(True)
plt.show()

![image](https://github.com/user-attachments/assets/8e988363-8c40-442a-9e2b-a83dacbb93b1)

2.Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors

plt.figure(figsize=(8,6))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinders')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.xlabel("Cylinders / Engine Size")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Cylinders & Engine Size vs CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

![image](https://github.com/user-attachments/assets/21e3e7ec-d51c-4e14-b0b7-6a1051c1dd46)

3.Using scatter plot compare data   cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors

plt.figure(figsize=(8,6))
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinders')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='red', label='Fuel Consumption')
plt.xlabel("Cylinders / Engine Size / Fuel Consumption")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Various Features vs CO2 Emissions")
plt.legend()
plt.grid(True)
plt.show()

![image](https://github.com/user-attachments/assets/e8c39caa-208e-4205-8778-8f5e40c64988)

4.Train your model with independent variable as cylinder and dependent variable as Co2Emission

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X1 = df[['CYLINDERS']]
y = df['CO2EMISSIONS']

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=0)

model1 = LinearRegression()
model1.fit(X1_train, y_train)
y_pred1 = model1.predict(X1_test)

print("Model 1 (Cylinders -> CO2) Accuracy (R2 Score):", r2_score(y_test, y_pred1))

![image](https://github.com/user-attachments/assets/8b1ac3cf-fc61-4def-a4e1-6486d9bd7ef7)

5.Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission

X2 = df[['FUELCONSUMPTION_COMB']]

X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)

model2 = LinearRegression()
model2.fit(X2_train, y_train)
y_pred2 = model2.predict(X2_test)

print("Model 2 (Fuel Consumption -> CO2) Accuracy (R2 Score):", r2_score(y_test, y_pred2))

![image](https://github.com/user-attachments/assets/4d5f21e6-2ebc-441a-9a7b-b2817fd7a068)

6.Train your model on different train test ratio and train the models and note down their accuracies

ratios = [0.2, 0.3, 0.4]
for r in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=r, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Train-Test Split {int((1-r)*100)}%/{int(r*100)}% => Accuracy (R2): {r2_score(y_test, y_pred):.4f}")

![image](https://github.com/user-attachments/assets/06022336-05d6-4404-9669-81fe18931e39)

## RESULT:

Thus, the program to create scatter plots and implement Linear Regression models to predict CO2 Emissions based on CYLINDERS and FUELCONSUMPTION_COMB was successfully written, executed, and evaluated for accuracy across different train-test splits.
