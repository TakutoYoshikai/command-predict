
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import numpy as np
import json
model = keras.models.load_model("model_lstm.h5", compile=False)
model.load_weights("model_lstm_weight.h5", by_name=False)

f = open("input_commands.json", "r")
input_s_c = json.load(f)
input_i_c = dict((int(s), c) for s, c in input_s_c.items())
input_c_i = dict((c, int(s)) for s, c in input_s_c.items())


f2 = open("next_commands.json", "r")
next_s_c = json.load(f2)
next_i_c = dict((int(s), c) for s, c in next_s_c.items())
next_c_i = dict((c, int(s)) for s, c in next_s_c.items())

def is_predictable(cmds):
    for cmd in cmds:
        if cmd not in input_c_i:
            return False
    return True

def predict(cmds):
    if !is_predictable(cmds):
        return None
    x = list(map(lambda a: makevec(input_c_i[a], len(input_i_c)),cmds))
    x = np.array(x)
    x = x.reshape(1, x.shape[0], x.shape[1])
    predict = model.predict(x)
    return next_i_c[get_index_from_vec(predict)[0]]

    
def makevec(i, n):
    res = []
    for j in range(n):
        res.append(0)
    res[i] = 1
    return res

def get_index_from_vec(vec):
    cdi = next_c_i["cd"]
    lsi = next_c_i["ls"]
    y = vec[0]
    index = -1
    maxval = 0
    for i in range(len(y)):
        if y[i] > maxval and (cdi != i and lsi != i):
            maxval = y[i]
            index = i
    return (index, maxval)

