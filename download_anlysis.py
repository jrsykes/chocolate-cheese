import cbpro
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import io
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

stock = 'BTC-EUR'

def add_original_feature(df, df_new):
	df_new['open'] = df['Open']
	df_new['open-1'] = df['Open'].shift(1)
	df_new['close-1'] = df['Close'].shift(1)
	df_new['high-1'] = df['High'].shift(1)
	df_new['low-1'] = df['Low'].shift(1)
	df_new['volume-1'] = df['Volume'].shift(1)

def add_avg_price(df, df_new):
	df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
	df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
	df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
	df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
	df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
	df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']

def add_avg_volume(df, df_new):
	df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
	df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
	df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
	df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
	df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
	df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']

def add_std_price(df, df_new):
	df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
	df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
	df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
	df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
	df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
	df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']

def add_std_volume(df, df_new):
	df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
	df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
	df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
	df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
	df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
	df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']

def add_return_feature(df, df_new):
	df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
	df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
	df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
	df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
	df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
	df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
	df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)

def generate_features(df):
	"""
	Generate features for stock/index based on historical price and performance
	@param df: dataframe with columns "Open", "Close", "High", "Low", "Volume", "Adjusted Close"
	@return: dataframe, data set with new features
	"""
	df_new = pd.DataFrame()
	# 6 original features
	add_original_feature(df, df_new)
	# 31 generated features
	add_avg_price(df, df_new)
	add_avg_volume(df, df_new)
	add_std_price(df, df_new)
	add_std_volume(df, df_new)
	add_return_feature(df, df_new)
	# the target
	df_new['Close'] = df['Close']
	df_new = df_new.dropna(axis=0)
	return df_new

Period1 = 0
Period2 = round(time.time())

error_message = b'{\n    "finance": {\n        "error": {\n            "code": "Unauthorized",\n            "description": "Invalid cookie"\n        }\n    }\n}\n'
s = error_message
while s == error_message:
	try:
		url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock + '?period1=' + str(Period1) + '&period2=' + str(Period2) + '&interval=1d&events=history&crumb=jsB7FEYdF2g'
		s = requests.get(url).content
	except:
		pass

data_raw = pd.read_csv(io.StringIO(s.decode('utf-8')))

data = generate_features(data_raw)



data_length = (data.shape[0])

start_train = 1
end_train = round(data_length*0.8)
start_test = round(data_length*0.8+1)
end_test= data_length

data_train = data.iloc[start_train:end_train]
data_test = data.iloc[start_test:end_test]


x_train = data_train.drop('Close', axis=1).values
y_train = data_train['Close'].values
x_test = data_test.drop('Close', axis=1).values
y_test = data_test['Close'].values




x_scaled_train = scaler.fit_transform(x_train)
x_scaled_test = scaler.transform(x_test)

pred_list = []

for i in range(20):

	param_grid = {"alpha": [1e-5, 3e-5, 1e-4], "eta0": [0.01, 0.03, 0.1]}
	lr = SGDRegressor()
	grid_search = GridSearchCV(lr, param_grid, cv = 5, scoring='r2')
	grid_search.fit(x_scaled_train, y_train)
	lr_best = grid_search.best_estimator_
	predictions_lr = lr_best.predict(x_scaled_test)

	pred_list.append(predictions_lr[predictions_lr.shape[0]-1])

print ('Max estimate: +', round(max(pred_list)/data['Close'][data.shape[0]-1]*100-100), '%\n€', round(max(pred_list)))

print ('Min estimate: ', round(min(pred_list)/data['Close'][data.shape[0]-1]*100-100), '%\n€', round(min(pred_list)))

print ('Price: €',round(data['Close'][data.shape[0]-1]))


print("MSE: {0:.3f}".format(mean_squared_error(y_test, predictions_lr)))
print("MAE: {0:.3f}".format(mean_absolute_error(y_test, predictions_lr)))
print("R^2: {0:.3f}".format(r2_score(y_test, predictions_lr)))

days = []
day = data_length
for i in range(predictions_lr.shape[0]):
	days.insert(0, day)
	day -= 1

all_days = []
day = 0
for i in range(data.shape[0]):
	day += 1
	all_days.append(day)

plt.clf()
plt.scatter(all_days, data['Close'], marker='o', c='b')
plt.scatter(days, predictions_lr, marker='*', c='k')
plt.xlabel('Time (days)')
plt.ylabel('Price') 
plt.show()