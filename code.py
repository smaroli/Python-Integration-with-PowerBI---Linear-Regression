# 'dataset' holds the input data for this script
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas
from sklearn.metrics import mean_squared_error, r2_score

x = pandas.DataFrame(dataset.loc[:,'demographics.asianPopulation','demographics.whitePopulation']
y = pandas.DataFrame(dataset.loc[:,'weeklySales'])
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)
yPrediction = linearRegressor.predict(xTest)
py = pandas.DataFrame(yPrediction)
py.columns = ['predicted_y']


#use table view. for training and valid, select all data that is not null for weekly sales
for test, select all data that is null for weekly sales

while combining, cmbine tables based on both tables.

asianPopulation = CALCULATE(SUM(demographics[asianPopulation]),FILTER(demographics, AND ('demographics'[demyear] = 'pred_demographics'[salesyear], demographics[locationID] = 'pred_demographics'[locationID] )))




# 'dataset' holds the input data for this script
import pandas as pd
from sklearn.linear_model import LinearRegression
train = pd.DataFrame(dataset[dataset.weeklySales.notnull()])
test = pd.DataFrame(dataset[dataset.weeklySales.isnull()])
linearRegressor = LinearRegression()
linearRegressor.fit(train[['weather_agg.Snow_Avg','weather_agg.Fog_Avg','weather_agg.Fog_Avg','weather_agg.Thunder_Avg','weather_agg.Temp_Avg','weather_agg.Wind_Avg']], train['weeklySales'])
yPrediction = linearRegressor.predict(test[['weather_agg.Snow_Avg','weather_agg.Fog_Avg','weather_agg.Fog_Avg','weather_agg.Thunder_Avg','weather_agg.Temp_Avg','weather_agg.Wind_Avg']])
test['predicted_sales'] = yPrediction
#py = pd.DataFrame(yPrediction)
#py.columns = ['predicted_y']







