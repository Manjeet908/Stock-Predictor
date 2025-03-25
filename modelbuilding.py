from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=100))
    model.add(Dropout(0.3))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
