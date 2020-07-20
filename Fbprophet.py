# -*- coding: utf-8 -*-
import pandas as pd
import fbprophet
import matplotlib.pyplot as plt
%matplotlib inline

#Reading from csv file
df = pd.read_csv('airline_passengers.csv')

#Displayint 5 rows of records
df.head()

#Displayting last 5 rows of records

df.tail()


#Plotting the data

df.plot()


#Renaming our date column to ds and target column as y because its required by fbprophet

df.columns=['ds','y']
df.head()


#Dropping the last 144 row as its invalid data

df.drop(144,axis=0,inplace=True)
df.tail()


#Converting our first columns into datatime as its string type

df['ds']=pd.to_datetime(df['ds'])
df.tail()


#Importing prophet

from fbprophet import Prophet

dir(Prophet)


#Initialize the Model

model=Prophet()
model.fit(df)


#Create future dates of 365 days

future_dates=model.make_future_dataframe(periods=365)
future_dates.head()


#Prediction using future dates

prediction = model.predict(future_dates)

prediction.head()

#Plot the predicted projection

model.plot(prediction)


#Visualize Each Components[Trends,yearly]

model.plot_components(prediction)



from fbprophet.diagnostics import cross_validation
df_cv=cross_validation(model,initial='730 days', period='180 days', horizon='365 days')
df_cv.head()


#Evaluating MSE,RMSE,MAE etc

from fbprophet.diagnostics import performance_metrics
df_p=performance_metrics(df_cv)
df_p.head()

#Plotting RMSE

from fbprophet.plot import plot_cross_validation_metric
fig=plot_cross_validation_metric(df_cv, metric='rmse')





