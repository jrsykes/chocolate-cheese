import threading
import cbpro
import datetime
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


key = ''
b64secret = ''
passphrase = ''

public_client = cbpro.PublicClient()
auth_client = cbpro.AuthenticatedClient(key, b64secret, passphrase)
accounts = auth_client.get_accounts()

coin_list = ['ZRX', 'BTC', 'XRP', 'XLM', 'LTC', 'ETH', 'ETC', 'EOS', 'BCH']

def five_minute_timer():
	threading.Timer(300.0, five_minute_timer).start()

#########
# Looper
				
	for value in coin_list:
		coin = str(value)
		print ('\n********************************\n\n' + value + '\n\n********************************\n')

		exchange = coin + '-EUR'
	

#########
# Net balance calculator

		active_coin_accounts = []
		for line in accounts:
			account = line
			for key,value in account.items():
				if str(value) in coin_list:
					active_coin_accounts.append(line)

		balance_list = []
		for item in active_coin_accounts:
			for key,value in item.items():
				if 'balance' in key:
					balance_list.append(value)

		balance_dictionary = dict(zip(coin_list, balance_list))

		
		net_balance = 0
		for key, value in balance_dictionary.items():
			coin_balance = float(value)
			ticker = (public_client.get_product_ticker(product_id=(key+'-EUR')))
			for key,value in ticker.items():
				if 'price' in key:
					coin_spot = float(value)
			net_balance = net_balance + coin_balance * coin_spot
			net_balance = float("{0:.8f}".format(net_balance))
		
#########
# EUR available balance

		currency = 'EUR'
		for item in accounts:
			if currency in str(item):
				for key,value in item.items():
					if 'available' in key:
						EUR_available_balance = float(value)
						EUR_available_balance = float("{0:.8f}".format(EUR_available_balance))

		net_balance = net_balance + EUR_available_balance

#########
# coin available balance

		for item in accounts:
			if coin in str(item):
				for key,value in item.items():
					if 'available' in key:
						coin_available_balance = float(value)

		

#########
# coin spot value

		ticker = (public_client.get_product_ticker(product_id=exchange))
		for key,value in ticker.items():
			if 'price' in key:
				coin_spot = float(value)
				coin_spot_len = str(len(value))

#########
# Time

		local_machine_time = datetime.datetime.now()

#########
# Historic data & buy, sell definition

		historic_data = public_client.get_product_historic_rates(exchange, granularity=86400)
		historic_data_list = []

		for item in historic_data:
			historic_data_list.append(float(item[3]))
	

		sell_point = float("{0:.8f}".format(np.mean(historic_data_list)))
		buy_point = float("{0:.8f}".format(((np.mean(historic_data_list)) - (np.std(historic_data_list)))))



		current_coin_order_count = 0
		for item in auth_client.get_orders():
			order = item
			for key,value in order.items():
				if coin in str(value):
					current_coin_order_count = current_coin_order_count + 1			

#########
# Regression line definition

		n_days = 0
		for item in historic_data_list:
			n_days = n_days + 1
		day_list = []
		last_day = n_days

		while n_days > 0:
			day_list.append(n_days)
			n_days = n_days - 1

		x = np.array(day_list).reshape(-1,1)
		y = np.array(historic_data_list).reshape(-1,1)
		model = LinearRegression().fit(x,y)
		
		poly_reg = PolynomialFeatures(degree = 4)
		x_poly = poly_reg.fit_transform(x)

		pol_reg = LinearRegression()
		pol_reg.fit(x_poly, y)
	
		predicted_value = pol_reg.predict(poly_reg.fit_transform([[last_day]]))

#########
# Buying
		print ('\nSearching for buy position at ' + str(local_machine_time))

		if float(predicted_value) > coin_spot*1.03 and model.coef_ > 0:
			print ('Buy position found')

			if (coin_available_balance * coin_spot) <= (net_balance * 0.2):


				if EUR_available_balance <= (net_balance * 0.2) - (coin_available_balance * coin_spot):
					buy_amount = EUR_available_balance
				else:
					buy_amount = (net_balance * 0.2) - (coin_available_balance * coin_spot)
		 	
			
				if current_coin_order_count == 0:
					size = str(round(buy_amount, 4))
					
					print ('Buying € ' + str(buy_amount) + ' worth of ' + coin)
					auth_client.place_limit_order(product_id=exchange, 
		        	                    side='buy', 
		        	                    price=coin_spot, 
		          	                  	size=size)
				else:
					print ('Buy order open')
			else:
				print ('Coin wallet >= 20 per cent of capital')		
		else:
			print ('Buy position not found')
				
#########
# Selling

		if coin_available_balance > 0:
			print ('\nSearching for sell position at ' + str(local_machine_time))		
			size = coin_available_balance
			if coin_spot > predicted_value*0.02:
				print ('Selling ' + str(coin_available_balance) + ' ' + coin)
				auth_client.place_limit_order(product_id=exchange, 
                        	    	side='sell', 
                          	   	 price=coin_spot, 
                          	    	size=size)			
			else:	
				print ('Sell position not found \n' + '\nSpot price:\t' + '€ ' + str(coin_spot) + '\n' + 'Sell point:\t' + '€ ' + str(predicted_value*0.02)[0:7])	

		time.sleep(10)
	print ('\n' + 'End of iteration')
	
five_minute_timer()









