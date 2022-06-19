# NSE-TATAGLOBAL.csv

#import packages
import pandas as pd
import numpy as np
import random

#to plot within notebook
import matplotlib.pyplot as plt

#for normalizing data
from sklearn.preprocessing import MinMaxScaler

#read the file
df = pd.read_csv('NSE-TATAGLOBAL.csv')

operation = ["+","-"]

def Average(lst):
	if operation[random.randint(0,1)] == "+":
		return round(sum(lst) / len(lst) + random.randint(1,10),2)
	else:
		return round(sum(lst) / len(lst) - random.randint(1,10),2)

#print the head
# print(df.head())

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Close Price history')
plt.show()

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    # new_data['Total Trade Quantity'][i] = data['Total Trade Quantity'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# print(new_data)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

varlid1 = []

tempTable = []

for i in range(887,len(dataset)):
	varlid1.append(dataset[i][0])
	

count = 0

for i in range(887,2094):
	temp_data = Average(varlid1[count:len(varlid1)])
	varlid1.append(temp_data)
	tempTable.append([temp_data])
	count += 1

tempTable = np.array(tempTable)

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(100,len(train)):
    x_train.append(scaled_data[i-100:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='adam',  metrics = ['accuracy'])

model.fit(
    x_train,
    y_train,
    epochs=200, batch_size=200, verbose=2,
    validation_split=0.05)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 100:].values

print(inputs[10:10])

print(tempTable[10:10])

inputs = tempTable.reshape(-1,1)
inputs  = scaler.fit_transform(inputs)

X_test = []
for i in range(100,inputs.shape[0]):
    X_test.append(inputs[i-100:i,0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))



# print(X_test.shape)

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

# for plotting
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price

plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.show()