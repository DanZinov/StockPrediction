import os
import time
from tensorflow.keras.layers import LSTM


# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next day
LOOKUP_STEP = 50

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'adjclose', 'volume',
       'adjclose_v', 'value_gas', 'value_silver', 'value_gold', 'value_usd',
       'PMI', 'Production', 'New Orders', 'Backlog of Orders',
       'Supplier Deliveries', 'Inventories', 'Customers Inventories',
       'Employment', 'Prices', 'New Export Orders', 'Imports']
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 350
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = True

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 100        
EPOCHS = 1000

# Apple stock market
ticker = "MSFT"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"