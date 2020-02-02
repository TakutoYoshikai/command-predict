
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import numpy as np
import json
model = keras.models.load_model("model_lstm.h5", compile=False)
model.load_weights("model_lstm_weight.h5", by_name=False)

f = open("commands.json", "r")
s_c = json.load(f)
i_c = dict((int(s), c) for s, c in s_c.items())
c_i = dict((c, int(s)) for s, c in s_c.items())

x = [
"git status",
"git add titan.go",
]

def predict(cmds):
    x = list(map(lambda a: a.split(" ")[0], x))
    x = list(map(lambda a: makevec(c_i[a], 88),x))
    x = np.array(x)
    x = x.reshape(1, x.shape[0], x.shape[1])
    predict = model.predict(x)
    i_c[softmax(predict)[0]]
def makevec(i, n):
    res = []
    for j in range(n):
        res.append(0)
    res[i] = 1
    return res
def vectoi(vec):
    for i in range(len(vec)):
        if i == 1:
            return i
    return None

def get_index_from_vec(vec):
    cdi = c_i["cd"]
    lsi = c_i["ls"]
    y = vec[0]
    index = -1
    maxval = 0
    for i in range(len(y)):
        if y[i] > maxval and (cdi != i and lsi != i):
            maxval = y[i]
            index = i
    return (index, maxval)

