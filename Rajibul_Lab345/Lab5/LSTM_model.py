from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input


def lstm_model(units, batch_size, input_size, input_dimension):
    model = Sequential()

    model.add(LSTM(units, batch_input_shape=(batch_size, input_size, input_dimension), stateful=True, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units, stateful=True, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units, stateful=True, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.summary()

    return model




