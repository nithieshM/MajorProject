import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D
from sklearn.metrics import r2_score
from sklearn import metrics

# Set up the page title
st.set_page_config(page_title='Stock Price Forecasting', page_icon=':chart_with_upwards_trend:')

# Set up the sidebar
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Enter a stock symbol:', 'AAPL')
start_date = st.sidebar.date_input('Start date:', value=pd.to_datetime('2009-01-01'))
end_date = st.sidebar.date_input('End date:', value=pd.to_datetime('2022-12-31'))

# Download the stock data
data = yf.download(ticker, start_date, end_date)
df = pd.DataFrame(data)
df.dropna(inplace=True)

# Show column wise %ge of NaN values they contain
st.subheader('Data')
st.write(df.head())
st.write(df.describe())
st.write('Column wise %ge of NaN values they contain:')
for i in df.columns:
    st.write(i, ' - ', df[i].isna().mean()*100)

# Choose correlated columns
cormap = df.corr()
st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cormap, annot=True)
st.pyplot(fig)

def get_correlated_col(cor_data, threshold):
    """Function to get columns that are correlated with the target"""
    feature=[]
    value=[]
    for i, index in enumerate(cor_data.index):
        if abs(cor_data[index]) > threshold:
            feature.append(index)
            value.append(cor_data[index])
    df = pd.DataFrame(data=value, index=feature, columns=['corr value'])
    return df.index

top_correlated_cols = get_correlated_col(cormap['Close'], 0.60)
df = df[top_correlated_cols]
X = df.drop(['Close'], axis=1)
y = df['Close']

# Scale the data
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

# Train the Linear Regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
acc_linreg = r2_score(y_test, y_pred_linreg)

# Train the CNN model
X_train_cnn = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
cnn_model = Sequential()
cnn_model.add(Conv1D(32, kernel_size=(3,), padding='same', activation='relu', input_shape=(X_train_cnn.shape[1],1)))
cnn_model.add(Conv1D(64, kernel_size=(3,), padding='same', activation='relu'))
cnn_model.add(Conv1D(128, kernel_size=(5,), padding='same', activation='relu'))
cnn_model.add(Flatten())
cnn_model.add(Dense(50,activation='relu'))
cnn_model.add(Dense(1))
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train_cnn, y_train, epochs=50, verbose=0)

st.subheader('Linear Regression Model')
st.write(f'Accuracy: {acc_linreg}')
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='True')
plt.plot(y_pred_linreg, label='Linear Regression')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot()

st.subheader('CNN Model')
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = y_pred_cnn.reshape(-1)
acc_cnn = r2_score(y_test, y_pred_cnn)
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='True')
plt.plot(y_pred_cnn, label='CNN')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot()
st.write(f'Accuracy: {acc_cnn}')
