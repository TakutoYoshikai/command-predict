
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import numpy as np
import json
model = keras.models.load_model("model_lstm.h5", compile=False)
model.load_weights("model_lstm_weight.h5", by_name=False)

param_model = keras.models.load_model("model_lstm_param.h5", compile=False)
param_model.load_weights("model_lstm_param_weight.h5", by_name=False)


f = open("input_commands.json", "r")
input_s_c = json.load(f)
input_i_c = dict((int(s), c) for s, c in input_s_c.items())
input_c_i = dict((c, int(s)) for s, c in input_s_c.items())


f2 = open("next_commands.json", "r")
next_s_c = json.load(f2)
next_i_c = dict((int(s), c) for s, c in next_s_c.items())
next_c_i = dict((c, int(s)) for s, c in next_s_c.items())

f3 = open("params.json", "r")
param_s_p = json.load(f3)
param_i_p = dict((int(s), c) for s, c in param_s_p.items())
param_p_i = dict((c, int(s)) for s, c in param_s_p.items())

f4 = open("params_input.json", "r")
param_input_s_c = json.load(f4)
param_input_i_c = dict((int(s), c) for s, c in param_input_s_c.items())
param_input_c_i = dict((c, int(s)) for s, c in param_input_s_c.items())
def is_predictable(cmds):
    for cmd in cmds:
        if cmd not in input_c_i:
            return False
    return True

def predict(cmds):
    if not is_predictable(cmds):
        return None
    x = list(map(lambda a: makevec(input_c_i[a], len(input_i_c)),cmds))
    x = np.array(x)
    x = x.reshape(1, x.shape[0], x.shape[1])
    x2 = list(map(lambda a: makevec(param_input_c_i[a], len(param_input_i_c)),cmds))
    x2 = np.array(x2)
    x2 = x2.reshape(1, x2.shape[0], x2.shape[1])
    predict = model.predict(x)
    predict2 = param_model.predict(x2)
    return (next_i_c[get_index_from_vec(predict)[0]], param_i_p[get_index_from_vec(predict2)[0]])

    
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

import http.server as s

class Handler(s.BaseHTTPRequestHandler):
    def do_POST(self):
        content_len  = int(self.headers.get("content-length"))
        body = json.loads(self.rfile.read(content_len).decode('utf-8'))
        res = predict([body[0], body[1]])
        res = res[0] + " " + res[1]
        self.send_response(200)
        self.send_header('Content-type', 'application/json;charset=utf-8')
        self.end_headers()
        self.wfile.write(res.encode("utf-8"))

host = "0.0.0.0"
port = 3333
httpd = s.HTTPServer((host, port), Handler)
httpd.serve_forever()
