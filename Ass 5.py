
import pandas as pd
df = pd.read_csv('bank.csv', delimiter=';')
print(df.head())
print(df.info())

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print(df2.head())

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
print(df3.head())

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
correlation_matrix = df3.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

y = df3['y'].apply(lambda x: 1 if x == 'yes' else 0)
X = df3.drop(columns=['y'])
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  # 75/25 split
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
logistic_cm = confusion_matrix(y_test, y_pred_logistic)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Confusion Matrix:")
print(logistic_cm)
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_cm = confusion_matrix(y_test, y_pred_knn)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN Confusion Matrix:")
print(knn_cm)
print(f"KNN Accuracy: {knn_accuracy:.2f}")


from sklearn.metrics import classification_report
print("Comparison of Models:")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
print(f"KNN Accuracy: {knn_accuracy:.2f}")
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))
print("\nClassification Report for KNN:")
print(classification_report(y_test, y_pred_knn))
