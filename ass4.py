'''import pandas as pd
file_path = 'data_banknote_authentication.csv'
data = pd.read_csv(file_path)
print("First 5 rows of the dataset:")
print(data.head())
X = data.drop(columns=['class'])
y = data['class']
print("\nShape of X (features):", X.shape)
print("Shape of y (target):", y.shape)


import pandas as pd
from sklearn.model_selection import train_test_split
file_path = 'data_banknote_authentication.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
print("Training set shapes: X_train =", X_train.shape, ", y_train =", y_train.shape)
print("Testing set shapes: X_test =", X_test.shape, ", y_test =", y_test.shape)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
file_path = 'data_banknote_authentication.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
svc = SVC(kernel='linear', random_state=20)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
file_path = 'data_banknote_authentication.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
svc = SVC(kernel='linear', random_state=20)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
file_path = 'data_banknote_authentication.csv'
data = pd.read_csv(file_path)
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
svc_rbf = SVC(kernel='rbf', random_state=20)
svc_rbf.fit(X_train, y_train)
y_pred_rbf = svc_rbf.predict(X_test)
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
print("Confusion Matrix (RBF Kernel):")
print(conf_matrix_rbf)
print("\nClassification Report (RBF Kernel):")
print(classification_report(y_test, y_pred_rbf))


Predict on the testing data and compute the confusion matrix and classification report
y_pred_rbf = svc_rbf.predict(X_test)
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
class_report_rbf = classification_report(y_test, y_pred_rbf)
print("Linear Kernel SVM:")
print("Confusion Matrix:\n", conf_matrix_linear)
print("Classification Report:\n", class_report_linear)
print("\nRBF Kernel SVM:")
print("Confusion Matrix:\n", conf_matrix_rbf)
print("Classification Report:\n", class_report_rbf)



import pandas as pd
file_path = 'weight-height.csv'
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm']
y = data['Weight_kg']
print("Feature (Height in cm):")
print(X.head())
print("\nTarget (Weight in kg):")
print(y.head())




import pandas as pd
from sklearn.model_selection import train_test_split
file_path = 'weight-height.csv'
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm']
y = data['Weight_kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size (X_train): {len(X_train)}")
print(f"Testing set size (X_test): {len(X_test)}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
file_path = 'weight-height.csv'
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm'].values.reshape(-1, 1)
y = data['Weight_kg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
min_max_scaler = MinMaxScaler()
X_train_normalized = min_max_scaler.fit_transform(X_train)
X_test_normalized = min_max_scaler.transform(X_test)
standard_scaler = StandardScaler()
X_train_standardized = standard_scaler.fit_transform(X_train)
X_test_standardized = standard_scaler.transform(X_test)
print("Sample of Normalized Training Data:")
print(X_train_normalized[:5])
print("\nSample of Standardized Training Data:")
print(X_train_standardized[:5])



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
file_path = 'weight-height.csv'
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm'].values.reshape(-1, 1)
y = data['Weight_kg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Value for KNN Regression on Unscaled Data: {r2:.4f}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
file_path = 'weight-height.csv'
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm'].values.reshape(-1, 1)
y = data['Weight_kg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)
knn_normalized = KNeighborsRegressor(n_neighbors=5)
knn_normalized.fit(X_train_normalized, y_train)
y_pred_normalized = knn_normalized.predict(X_test_normalized)
r2_normalized = r2_score(y_test, y_pred_normalized)
print(f"R2 Value for KNN Regression on Normalized Data: {r2_normalized:.4f}")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
file_path = 'weight-height.csv'
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm'].values.reshape(-1, 1)
y = data['Weight_kg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)
knn_standardized = KNeighborsRegressor(n_neighbors=5)
knn_standardized.fit(X_train_standardized, y_train)
y_pred_standardized = knn_standardized.predict(X_test_standardized)
r2_standardized = r2_score(y_test, y_pred_standardized)
print(f"R2 Value for KNN Regression on Standardized Data: {r2_standardized:.4f}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
file_path = 'weight-height.csv'  # Ensure the file is in the same directory as this script
data = pd.read_csv(file_path)
data['Height_cm'] = data['Height'] * 2.54
data['Weight_kg'] = data['Weight'] * 0.453592
X = data['Height_cm'].values.reshape(-1, 1)
y = data['Weight_kg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_unscaled = KNeighborsRegressor(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_unscaled = knn_unscaled.predict(X_test)
r2_unscaled = r2_score(y_test, y_pred_unscaled)
scaler_normalized = MinMaxScaler()
X_train_normalized = scaler_normalized.fit_transform(X_train)
X_test_normalized = scaler_normalized.transform(X_test)
knn_normalized = KNeighborsRegressor(n_neighbors=5)
knn_normalized.fit(X_train_normalized, y_train)
y_pred_normalized = knn_normalized.predict(X_test_normalized)
r2_normalized = r2_score(y_test, y_pred_normalized)
scaler_standardized = StandardScaler()
X_train_standardized = scaler_standardized.fit_transform(X_train)
X_test_standardized = scaler_standardized.transform(X_test)
knn_standardized = KNeighborsRegressor(n_neighbors=5)
knn_standardized.fit(X_train_standardized, y_train)
y_pred_standardized = knn_standardized.predict(X_test_standardized)
r2_standardized = r2_score(y_test, y_pred_standardized)
print("Comparison of Models in Terms of R2 Values:")
print(f"Unscaled Data: R2 = {r2_unscaled:.4f}")
print(f"Normalized Data: R2 = {r2_normalized:.4f}")
print(f"Standardized Data: R2 = {r2_standardized:.4f}")'''




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
file_path = 'C:\Users\pacman\PycharmProjects\Snehahasrepo/suv.csv'
data = pd.read_csv(file_path)
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def evaluate_decision_tree(criterion, X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    return cm, cr
print("Decision Tree with Entropy Criterion:")
cm_entropy, cr_entropy = evaluate_decision_tree('entropy', X_train_scaled, X_test_scaled, y_train, y_test)
print("Confusion Matrix:\n", cm_entropy)
print("Classification Report:\n", cr_entropy)
print("Decision Tree with Gini Criterion:")
cm_gini, cr_gini = evaluate_decision_tree('gini', X_train_scaled, X_test_scaled, y_train, y_test)
print("Confusion Matrix:\n", cm_gini)
print("Classification Report:\n", cr_gini)







