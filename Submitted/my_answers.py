import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    # reshape each
    
    # set the default data type as int64, the final data type will follow the input data
    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    
    length = len(series)
    for i in range(0, length - window_size):
        # get eash slice
        x = series[i:i+window_size]
        # concatenate to the earlier X
        X = np.concatenate((X,x))

        y1 = series[i+window_size:i+window_size+1]
        y = np.concatenate((y,y1))

    # set the shape of X and y
    X.shape = (int(len(X)/window_size), window_size)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, LSTM
    from keras.optimizers import RMSprop
    from keras.utils.data_utils import get_file
    import keras
    import random

    # TODO build the required RNN model: a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
    np.random.seed(0)
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))
    model.summary()

    # initialize optimizer
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile model --> make sure initialized optimizer and callbacks - as defined above - are used
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    pass


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # remove as many non-english characters and character sequences as you can 
    text = text.replace('/', ' ')
    text = text.replace('è', ' ')
    text = text.replace('?', ' ')
    text = text.replace('&', ' ')
    text = text.replace('%', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('à', ' ')
    text = text.replace('â', ' ')
    text = text.replace('é', ' ')
    text = text.replace('*', ' ')
    text = text.replace('@', ' ')
    text = text.replace('$', ' ')
    text = text.replace('-', ' ')
    text = text.replace('0', ' ')
    text = text.replace('1', ' ')
    text = text.replace('2', ' ')
    text = text.replace('3', ' ')
    text = text.replace('4', ' ')
    text = text.replace('5', ' ')
    text = text.replace('6', ' ')
    text = text.replace('7', ' ')
    text = text.replace('8', ' ')
    text = text.replace('9', ' ')
        
    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    length = len(text)
    for i in range(0, int((length-window_size)/step_size)):
        curStartLoc = i * step_size
        curInputSlice = text[curStartLoc:curStartLoc+window_size]
        inputs.append(curInputSlice)
        outputs.append(text[curStartLoc+window_size:curStartLoc+window_size+1])
    
    return inputs,outputs
