import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras

keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

stock_data = pd.read_csv('GOOG.csv')
stock_data = stock_data.iloc[:,1:2].values

st = MinMaxScaler()
stock_data = st.fit_transform(stock_data)

X_train = stock_data[0:732]
y_train = stock_data[1:733]

X_train = np.reshape(X_train, (732,1,1))

regressor = Sequential()
regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (None,1), return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 10, activation = 'relu', return_sequences = False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1, activation = 'tanh'))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, batch_size = 32, epochs =640, verbose = 2)

test_set = pd.read_csv('GOOG_test.csv')
real_stock_price = test_set.iloc[:,1:2].values
inputs = real_stock_price
inputs = st.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = st.inverse_transform(predicted_stock_price)
test_set['real_stock_price']= real_stock_price
test_set['real_stock_price'] = test_set['real_stock_price']

plt.plot(test_set['real_stock_price'], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.xticks(range(0, 20))
plt.legend()
plt.show()
print("Predicted price tomorrow is:", predicted_stock_price[19])
if predicted_stock_price[19] > predicted_stock_price[18]:
    direction = "up"
else:
    direction = 'down'
print("Predicted direction is: "+ direction)