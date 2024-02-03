import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data =pd.read_csv('insurance.csv')

data['sex']=data['sex'].map({'male':1,'female':0})
data['smoker']=data['smoker'].map({'yes':1,'no':0})
data['region']=data['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})

plt.figure(figsize=(13,7))
sns.histplot(data.age,bins=20,kde=False,color='blue')

#univariate analysis
#BMI distribution
plt.figure(figsize=(13,7))
sns.histplot(data.bmi,bins=20,kde=True,color='purple')
plt.title('BMI distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency of each BMI')
plt.show()
#charges
plt.figure(figsize=(13,7))
sns.histplot(data.charges ,bins=10,kde=False,color='red')
plt.title('Charges distribution')
plt.xlabel('Charges')
plt.ylabel('distribution of charges')
plt.show()
#multivariate analysis
plt.figure(figsize=(13,7))
sns.scatterplot(x='age',y='charges',data=data[(data.smoker==1)])
plt.title("Scatterplot for charges of Smokers")
plt.xlabel('Age')
plt.ylabel('Medical Expenses')
plt.show()
import warnings
warnings.filterwarnings('ignore')
#feature selection
data.drop('region',axis=1,inplace=True)

#Linear Regression
X=data.drop('charges',axis=1)
y=data.charges

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=2)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
from sklearn import metrics
print('MAE: ',metrics.mean_absolute_error(y_test,y_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R2 Score:",metrics.r2_score(y_test,y_pred))

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)

X_train_poly=poly_reg.fit_transform(X_train)

X_train,X_test,y_train,y_test=train_test_split(X_train_poly,y_train,test_size=0.2,random_state=42)

#train
plr=LinearRegression()
plr.fit(X_train,y_train)

y_pred_plr=plr.predict(X_test)
#model evaluation
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_plr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_plr)))
print("R2 Score: ", metrics.r2_score(y_test, y_pred_plr))

age=int(input("enter age: "))
sex=int(input("enter sex (male:1,female:0): "))
bmi=int(input("enter bmi: "))
children=int(input("enter the number of children you have: "))
smoker=int(input("are you a smoketr (no:0,yes:1): "))
charge=reg.predict([[age,sex,bmi,children,smoker]])
print('The charge of this patient is $',charge[0])