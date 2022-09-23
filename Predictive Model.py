from statistics import LinearRegression
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn

st.title('A Term Deposit Predictive Model')
data=pd.read_excel('Term_Deposit_Investment Bank.xlsx', sheet_name = 'Retail Segment')
data

data.info()


data.columns
#There are no missing values in the 6 columns. 4 numeric and 2 categorical


data.describe()


fig = px.histogram(data,['DEPOSIT_AMOUNT'], nbins = 40, color_discrete_sequence=['rgb(102,17,0)'],
                  marginal ='box', title ='Term Deposits in KES M')
fig


fig = px.histogram(data,['INTEREST_RATE'], nbins = 40, color_discrete_sequence=['rgb(102,17,0)'],
                  marginal ='box', title ='Interest rates in %')
fig


#Feature Engineering
#Feature Preprocessing- removing outliers


q1 =data['DEPOSIT_AMOUNT'].quantile(0.25)
q3 =data['DEPOSIT_AMOUNT'].quantile(0.75)
IQR = data['DEPOSIT_AMOUNT'].quantile(0.75) -data['DEPOSIT_AMOUNT'].quantile(0.25) 
whisker_1 = q1-(1.5*IQR)
whisker_3 = q3+(1.5*IQR)
whisker_1, whisker_3

data['DEPOSIT_AMOUNT'].loc[data['DEPOSIT_AMOUNT']>22581407.8250000037]=q3
data.head(60)


q1 =data['INTEREST_RATE'].quantile(0.25)
q3 =data['INTEREST_RATE'].quantile(0.75)
IQR = data['INTEREST_RATE'].quantile(0.75) -data['INTEREST_RATE'].quantile(0.25) 
whisker_1 = q1-(1.5*IQR)
whisker_3 = q3+(1.5*IQR)
whisker_1, whisker_3

data['INTEREST_RATE'].loc[data['INTEREST_RATE']>9.25] =q3
data.tail(60)
#Feature engineering
#Feature addition

data['DATE_DAY'] = pd.to_datetime(data['DATE_DAY'])
data['DATE_DAY']

data['YEAR']= pd.DatetimeIndex(data['DATE_DAY']).year
data['YEAR']

data['MONTH']= pd.DatetimeIndex(data['DATE_DAY']).month
data['MONTH']

data

data['INTEREST_AMOUNT'] = data['DEPOSIT_AMOUNT'] * (data['INTEREST_RATE']/100)
data['INTEREST_AMOUNT']

data

data_new =data[['DEPOSIT_AMOUNT','INTEREST_RATE','YEAR','MONTH','INTEREST_AMOUNT']]
data_new

#Feature selection using pearson's coefficient of corelation

cor = data_new.corr()
cor

cor_target = abs(cor["DEPOSIT_AMOUNT"])
cor_target

relevant_features = cor_target[cor_target > 0.4]
relevant_features


#Algorithm selection
##Detective or predictive, modeling vs just dashboarding

#Regression model, is used to predict the deposit
# y= deposit amounts, x1= interest rates, x2 = foracid


#Dividing into Train and Test

x=data_new.drop('DEPOSIT_AMOUNT', axis=1)
y= data_new['DEPOSIT_AMOUNT']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state=42)

y_test.shape


np.random.seed(42)
from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(x_train, y_train)


model.predict(x_test)

y_pred = model.predict(x_test)

model.score(x_train, y_train)

model.score(x_test, y_test)


#Model Evaluation

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)




