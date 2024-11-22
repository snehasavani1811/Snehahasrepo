
'''import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

correlations = data.corr()['target'].sort_values(ascending=False)
print("Correlation of features with target:\n", correlations)

"""
Findings:
- The strongest correlations with the target are:
  1. S5 (0.565)
  2. BMI (0.586)
  3. BP (0.441)

- Since BMI and S5 are already part of the model, the next variable to include is BP (Blood Pressure), as it has a significant positive correlation with diabetes progression.
"""

plt.scatter(data['bp'], data['target'], alpha=0.5)
plt.title("Scatter plot of Blood Pressure (BP) vs Diabetes Progression")
plt.xlabel("Blood Pressure (BP)")
plt.ylabel("Diabetes Progression (Target)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data[['bmi', 's5', 'bp']], y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

"""
Findings:
- Adding BP as a variable improves the model's explanatory power.
- R-squared value indicates how well the model predicts diabetes progression.
- BP has a strong theoretical and statistical justification for inclusion.
"""





from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

X_bmi_s5 = data[['bmi', 's5']]
X_bmi_s5_bp = data[['bmi', 's5', 'bp']]

# Train-test split for both models
X_train_baseline, X_test_baseline, y_train, y_test = train_test_split(X_bmi_s5, y, test_size=0.2, random_state=42)
X_train_extended, X_test_extended, _, _ = train_test_split(X_bmi_s5_bp, y, test_size=0.2, random_state=42)

model_baseline = LinearRegression()
model_extended = LinearRegression()

model_baseline.fit(X_train_baseline, y_train)
y_pred_baseline = model_baseline.predict(X_test_baseline)

model_extended.fit(X_train_extended, y_train)
y_pred_extended = model_extended.predict(X_test_extended)

mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

mse_extended = mean_squared_error(y_test, y_pred_extended)
r2_extended = r2_score(y_test, y_pred_extended)

print("Baseline Model (BMI and S5):")
print(f"Mean Squared Error (MSE): {mse_baseline:.2f}")
print(f"R-squared (R²): {r2_baseline:.2f}")

print("\nExtended Model (BMI, S5, and BP):")
print(f"Mean Squared Error (MSE): {mse_extended:.2f}")
print(f"R-squared (R²): {r2_extended:.2f}")

"""
Findings:
- Adding BP improves the R² score slightly, showing better explanatory power for diabetes progression.
- The MSE also decreases, indicating that the model's predictions are closer to actual values with BP included.
- While the improvement may not be very large, it demonstrates that BP contributes additional information to the model.
"""
'''


import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

correlations = data.corr()['target'].sort_values(ascending=False)
print("Correlation of features with the target:\n", correlations)

"""
Findings:
- Top correlated features with the target are:
  1. BMI (0.586)
  2. S5 (0.565)
  3. BP (0.441)
  4. S4 (0.430)
  5. S6 (0.382)

We will progressively add S4 and S6 to the model and evaluate its performance.
"""

X_baseline = data[['bmi', 's5']]
X_extended = data[['bmi', 's5', 'bp']]
X_full = data[['bmi', 's5', 'bp', 's4', 's6']]

X_train_base, X_test_base, y_train, y_test = train_test_split(X_baseline, y, test_size=0.2, random_state=42)
X_train_ext, X_test_ext, _, _ = train_test_split(X_extended, y, test_size=0.2, random_state=42)
X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_base = LinearRegression()
model_ext = LinearRegression()
model_full = LinearRegression()

model_base.fit(X_train_base, y_train)
y_pred_base = model_base.predict(X_test_base)

model_ext.fit(X_train_ext, y_train)
y_pred_ext = model_ext.predict(X_test_ext)

model_full.fit(X_train_full, y_train)
y_pred_full = model_full.predict(X_test_full)

mse_base = mean_squared_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

mse_ext = mean_squared_error(y_test, y_pred_ext)
r2_ext = r2_score(y_test, y_pred_ext)

mse_full = mean_squared_error(y_test, y_pred_full)
r2_full = r2_score(y_test, y_pred_full)

print("Baseline Model (BMI and S5):")
print(f"Mean Squared Error (MSE): {mse_base:.2f}")
print(f"R-squared (R²): {r2_base:.2f}\n")

print("Extended Model (BMI, S5, BP):")
print(f"Mean Squared Error (MSE): {mse_ext:.2f}")
print(f"R-squared (R²): {r2_ext:.2f}\n")

print("Full Model (BMI, S5, BP, S4, S6):")
print(f"Mean Squared Error (MSE): {mse_full:.2f}")
print(f"R-squared (R²): {r2_full:.2f}\n")

"""
Findings:
- Adding BP to the baseline model improves performance (lower MSE, higher R²).
- Adding more variables like S4 and S6 (Full Model) further improves performance, though the improvement diminishes as we add more features.
- This indicates that while adding variables helps, the gain decreases with features that have weaker correlations or redundant information.
"""


