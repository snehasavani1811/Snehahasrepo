'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
df2 = pd.read_csv("iris.csv")
print(df2.head())
x = df2.drop('species', axis=1)
y = df2['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='poly', degree=2)
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
df = pd.read_csv('weight-height.csv',skiprows=0,delimiter=',')

x=2.54*df[['Height']]
x_mm = MinMaxScaler().fit_transform(x)
x_std = StandardScaler().fit_transform(x)
x = np.array(x)
x_std = np.array(x_std)
x_mm = np.array(x_mm)

plt.subplot(1,3,1)
plt.hist(x_mm,30)
plt.xlabel("Height")
plt.title("original")
plt.subplot(1,3,2)
plt.hist(x_mm,30)
plt.title("Normalized")
plt.subplot(1,3,3)
plt.hist(x_std,30)
plt.title("Standardized")
plt.show()

x_mm2 = (x-np.min(x))/(np.max(x)-np.min(x))
print("diff=",np.max(np.abs(x_mm2-x_mm2)))
x_std2 = (x-np.mean(x))/np.std(x)
print("diff=",np.mean(np.abs(x_std2-x_std2)))




import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('Admission_Predict.csv',skiprows=0,delimiter=",")
print(df)
x = df[["CGPA",'GRE Score']]
y = df[["Chance of Admit "]]
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
x_train_norm = MinMaxScaler().fit_transform(x_train)
x_test_norm = MinMaxScaler().fit_transform(x_test)
x_train_std = StandardScaler().fit_transform(x_train)
x_test_std = StandardScaler().fit_transform(x_test)
print(x_train_norm)
print(x_test_norm)
print(x_train_std)
print(x_test_std)

lm = neighbors.KNeighborsRegressor(n_neighbors=5)
lm.fit(x_train, y_train)
predictions = lm.predict(x_test)
print("R2 =",lm.score(x_test,y_test))

lm.fit(x_train_norm, y_train)
print("R2 (norm) =",lm.score(x_test_norm,y_test))

lm.fit(x_train_std, y_train)
print("R2 (std) =",lm.score(x_test_std,y_test))
'''

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

df = pd.read_csv("data_banknote_authentication.csv")
print(df.head())
x = df.drop('class', axis=1)
y = df['class']
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
dot_data = tree.export_graphviz(classifier, out_file=None,
            feature_names = x_train.columns,class_names = "class",
            filled = True, rounded = True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("dtree")