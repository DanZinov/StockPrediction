import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
import random
import quandl
# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, 
                test_size=0.2, feature_columns=['open', 'high', 'low', 'close', 'adjclose', 'volume',
       'adjclose_v', 'value_gas', 'value_silver', 'value_gold', 'value_usd',
       'PMI', 'Production', 'New Orders', 'Backlog of Orders',
       'Supplier Deliveries', 'Inventories', 'Customers Inventories',
       'Employment', 'Prices', 'New Export Orders', 'Imports']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    
    ind = []
    quandl.ApiConfig.api_key = 'hXxA3xSampghdgkVeSJC'
    pmi = pd.read_excel("pmi.xlsx", index_col=0)
    
       
    # Cleaning, processing and transforming the gold data
    gold = quandl.get("WGC/GOLD_DAILY_USD")
    gold.columns = ["value_gold"]
    
    gold_ind = []
    gold_data = []
    for i in range(1, len(gold.index)):
        if(gold.index[i-1].day != gold.index[i].day-1):
            for k in range(gold.index[i].day - gold.index[i-1].day):
                try:
                    gold_ind.append(datetime(gold.index[i-1].year, gold.index[i-1].month, gold.index[i-1].day+k))
                    gold_data.append(gold.value_gold[i-1])
                except:
                    pass
        else:
            gold_ind.append(gold.index[i-1])
            gold_data.append(gold.value_gold[i-1])
    
    
    gold = pd.DataFrame(data = gold_data, index = gold_ind, columns = ["value_gold"])
    # Cleaning, processing and transforming the USD data    
    gold_temp = quandl.get("WGC/GOLD_DAILY_USD")
    gold_temp.columns = ["value_gold"]
    
    usd_value = quandl.get("FRED/TWEXB")
    usd_value.columns = ["value_usd"]
    data = []
    for i in range(len(list(usd_value.value_usd))-1):
        for k in range(5):
            data.append(list(usd_value.value_usd)[i])
    data.append(list(usd_value.value_usd)[-1])
    new_usd = pd.DataFrame(data, index=gold_temp.index[gold_temp.index.get_loc('1995-01-04'):gold_temp.index.get_loc('2020-01-01')+1])
    new_usd.columns = ["value_usd"]
    
    new_usd_ind = []
    new_usd_data = []
    for i in range(1, len(new_usd.index)):
        if(new_usd.index[i-1].day != new_usd.index[i].day-1):
            for k in range(new_usd.index[i].day - new_usd.index[i-1].day):
                try:
                    new_usd_ind.append(datetime(new_usd.index[i-1].year, new_usd.index[i-1].month, new_usd.index[i-1].day+k))
                    new_usd_data.append(new_usd.value_usd[i-1])
                except:
                    pass
        else:
            new_usd_ind.append(new_usd.index[i-1])
            new_usd_data.append(new_usd.value_usd[i-1])
    
    
    new_usd = pd.DataFrame(data = new_usd_data, index = new_usd_ind, columns = ["value_usd"])
    
    # Cleaning, processing and transforming the silver data
    silver = quandl.get("LBMA/SILVER")
    silver.drop(["GBP", "EURO"], axis=1, inplace=True)
    silver.columns = ["value_silver"]
    
    silver_ind = []
    silver_data = []
    for i in range(1, len(silver.index)):
        if(silver.index[i-1].day != silver.index[i].day-1):
            for k in range(silver.index[i].day - silver.index[i-1].day):
                try:
                    silver_ind.append(datetime(silver.index[i-1].year, silver.index[i-1].month, silver.index[i-1].day+k))
                    silver_data.append(silver.value_silver[i-1])
                except:
                    pass
        else:
            silver_ind.append(silver.index[i-1])
            silver_data.append(silver.value_silver[i-1])
    
    
    silver = pd.DataFrame(data = silver_data, index = silver_ind, columns = ["value_silver"])

    # Cleaning, processing and transforming the gas data
    
    gas = quandl.get("FRED/DGASUSGULF")
    gas.columns = ["value_gas"]
    
    gas_ind = []
    gas_data = []
    for i in range(1, len(gas.index)):
        if(gas.index[i-1].day != gas.index[i].day-1):
            for k in range(gas.index[i].day - gas.index[i-1].day):
                try:
                    gas_ind.append(datetime(gas.index[i-1].year, gas.index[i-1].month, gas.index[i-1].day+k))
                    gas_data.append(gas.value_gas[i-1])
                except:
                    pass
        else:
            gas_ind.append(gas.index[i-1])
            gas_data.append(gas.value_gas[i-1])
    
    
    gas = pd.DataFrame(data = gas_data, index = gas_ind, columns = ["value_gas"])
    # Cleaning, processing and transforming the Volatility data
    v_data = si.get_data("^VIX")
    v_data.drop(["ticker", "volume", 'open', 'high', 'low', 'close'], axis=1, inplace=True)
    v_data.columns = ["adjclose_v"]
    
    v_data_ind = []
    v_data_data = []
    for i in range(1, len(v_data.index)):
        if(v_data.index[i-1].day != v_data.index[i].day-1):
            for k in range(v_data.index[i].day - v_data.index[i-1].day):
                try:
                    v_data_ind.append(datetime(v_data.index[i-1].year, v_data.index[i-1].month, v_data.index[i-1].day+k))
                    v_data_data.append(v_data.adjclose_v[i-1])
                except:
                    pass
        else:
            v_data_ind.append(v_data.index[i-1])
            v_data_data.append(v_data.adjclose_v[i-1])
    
    
    v_data = pd.DataFrame(data = v_data_data, index = v_data_ind, columns = ["adjclose_v"])
    
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    df_ind = []
    df_data = {"open":[], "high":[], "low":[], "close":[], "adjclose":[], "volume":[], "ticker":[]}
    for i in range(1, len(df.index)):
        if(df.index[i-1].day != df.index[i].day-1):
            for k in range(df.index[i].day - df.index[i-1].day):
                try:
                    df_ind.append(datetime(df.index[i-1].year, df.index[i-1].month, df.index[i-1].day+k))
                    df_data["open"].append(df.open[i-1])
                    df_data["high"].append(df.high[i-1])
                    df_data["low"].append(df.low[i-1])
                    df_data["close"].append(df.close[i-1])
                    df_data["adjclose"].append(df.adjclose[i-1])
                    df_data["volume"].append(df.volume[i-1])
                    df_data["ticker"].append(df.ticker[i-1])
                except:
                    pass
        else:
            df_ind.append(df.index[i-1])
            df_data["open"].append(df.open[i-1])
            df_data["high"].append(df.high[i-1])
            df_data["low"].append(df.low[i-1])
            df_data["close"].append(df.close[i-1])
            df_data["adjclose"].append(df.adjclose[i-1])
            df_data["volume"].append(df.volume[i-1])
            df_data["ticker"].append(df.ticker[i-1])
    df = pd.DataFrame(data = df_data, index = df_ind)    
    final_df = pd.concat([df, v_data, gas, silver, gold, new_usd, pmi], axis=1, join="inner")
    print(final_df.columns)
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['final_df'] = final_df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in final_df.columns, f"'{col}' does not exist in the dataframe."

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            final_df[column] = scaler.fit_transform(np.expand_dims(final_df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    final_df['future'] = final_df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(final_df[feature_columns].tail(lookup_step))
    
    # drop NaNs
    final_df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(final_df[feature_columns].values, final_df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # return the result
    return result


def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model