'''
import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7,8,9],[10,11,12]]])
print('last element from 2nd dim: ', arr[1,0,-3])

import numpy as np
a= np.array([[1,2],[3,4]])
b= np.array([[5,6],[7,8]])
print(a)
print(b)
c= a+b
print(c)
print(a)
print(b)
d= a*b
print(d)
e = np.matmul(a,b)
print(e)
print(a**2)

import numpy as np
a = np.array([[2,1], [-4,3]])
b = np.array([11,3])
x = np.linalg.solve(a,b)
print(x)

Ainv = np.linalg.inv(a)
print(np.matmul(Ainv,b))



import matplotlib.pyplot as plt
import numpy as np
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([1,3,5,7,9,11,13,15,17,19])
plt.scatter(x,y)
plt.plot(x,y)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = np.array([0,1,2,3,4,5])
y = np.array([1,5,9,13,17,21])
plt.scatter(x,y)
plt.plot(x,y)
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("linreg_data.csv",skiprows=0,names=["x","y"])
print(data)
xpd = data["x"]
ypd = data["y"]
n = xpd.size
plt.scatter(xpd,ypd)
plt.show()
xbar = np.mean(xpd)
ybar = np.mean(ypd)
term1 = np.sum(xpd*ypd)
term2 = np.sum(xpd**2)
b = (term1-n*xbar*ybar)/(term2-n*xbar*xbar)
a = ybar - b*xbar
x = np.linspace(0,2,100)
y = a+b*x
plt.plot(x,y,color="black")
plt.scatter(xpd,ypd)
plt.scatter(xbar,ybar,color="red")
plt.show()


import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
my_data = np.genfromtxt('linreg_data.csv',delimiter=',')
xp = my_data[:,0]
yp = my_data[:,1]
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)
regr = linear_model.LinearRegression()
regr.fit(xp,yp)
print(regr.coef_,regr.intercept_)
xval= np.full((1,1),0.5)
yval =regr.predict(xval)
print(yval)


xval = np.linspace(-1,2,20).reshape(-1,1)
yval = regr.predict(xval)
plt.plot(xval,yval)
plt.scatter(xp,yp,color='black')
plt.show()
from sklearn import metrics
yhat = regr.predict(xp)
print('Mean) Absolute Error:', metrics.mean_absolute_error(yp,yhat))
print('Mean Squared Error:', metrics.mean_squared_error(yp,yhat))
print ('Root Mean Squared Error:', metrics.root_mean_squared_error(yp,yhat))
print('R2 value:', metrics.r2_score(yp,yhat))


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data_pd = pd.read_csv("quadreg_data.csv",skiprows=0,names=["x","y"])
print(data_pd)

xpd = np.array(data_pd["x"])
ypd = np.array(data_pd["y"])
xpd = xpd.reshape(-1,1)
ypd = ypd.reshape(-1,1)
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(xpd)
pol_reg = LinearRegression()
pol_reg.fit(X_poly,ypd)
plt.scatter(xpd,ypd, color='red')
xval = np.linspace(-1,1,10).reshape(-1,1)
plt.plot(xval,pol_reg.predict(poly_reg.transform(xval)), color="blue")
plt.show()
print(pol_reg.coef_)
print("c=",pol_reg.intercept_)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data= load_diabetes(as_frame=True)
print(data)
print(data.keys())
df = data.frame
print(df.head())
plt.hist(df["target"],25)
plt.xlabel("target")
plt.show()
sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()
from matplotlib import pyplot as plt

from codecs import xmlcharrefreplace_errors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.linear_model import LinearRegression
data = load_diabetes(as_frame=True)
print(data)
print(data.keys())
df = data.frame
print(df.head())
plt.subplot(1,2,1)
plt.scatter(df['bmi'], ['target'])
plt.xlabel("bmi")
plt.ylabel("target")
plt.subplot(1,2,2)
plt.scatter(df['a5'], df['target'])
plt.xlabel("a5")
plt.ylabel("target")
plt.show()
x = pd.DataFrame(df[['bmi','a5']], columns=['bmi','a5'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(xmlcharrefreplace_errors, y, test_size=0.2, random_state=5)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_log_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
y_test_predict = lm.predict(X_test)
rmse_test = np.sqrt(mean_squared_log_error(y_test, y_test_predict))
r2_test = r2_score(y_test, y_test_predict)
print(rmse_test, r2)
print(r2_test,rmse_test)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
print(data.keys())

print(data.DESCR)

df = data['frame']
print(df)

plt.hist(df["target"],25)
plt.xlabel("target")
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')

plt.subplot(1,2,2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.show()

X = pd.DataFrame(df[['bmi', 's5']], columns=['bmi', 's5'])
y = df['target']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print(f"RMSE = {rmse}, R2 = {r2}")

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)
print(f"RMSE (test) = {rmse_test}, R2 (test)= {r2_test}")

print(X_test, y_test_predict)



from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df =pd.read_csv("ridgereg_data.csv")
x = df[['x']]
y = df[['y']]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, test_size=0.2)

for alp in [0,10,20,30,50,100,1000]:
    rr = Ridge(alpha=alp)
    rr.fit(x_train, y_train)
    plt.scatter(x_train, y_train)
    plt.plot(x_train, rr.predict(x_train),color="red")
    plt.title("alpha="+str(alp))
    plt.show()

alphas = np.linspace(0,4,50)
r2values = []
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(x_train, y_train)
    r2test = r2_score(y_test, rr.predict(x_test))
    r2values.append(r2test)
    plt.plot(alphas,r2values)
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("exams.csv", skiprows=0, delimiter=",")
print(df)
x=df.iloc[:, 0:2]
y=df.iloc[:, -1]

admit_yes = df.loc[y == 1]
admit_no = df.loc[y == 0]

plt.scatter(admit_yes.iloc[:, 0], admit_yes.iloc[:, 1],label="admit yes")
plt.scatter(admit_no.iloc[:, 0], admit_no.iloc[:, 1],label="admit no")
plt.xlabel("exam1")
plt.ylabel("exam2")
plt.legend()
plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("iris.csv")
print(df)

x = df.iloc[:, 0:4].values
y = df.iloc[:, -4].values
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.20)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

metrics.ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)
plt.show()

print(classification_report(y_test, y_pred))
error = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    error.append(np.mean(y_pred != y_test))

plt.plot(range(1, 20), error,marker='o', markersize=10)
plt.xlabel('k')
plt.ylabel('Mean Error')
plt.show()