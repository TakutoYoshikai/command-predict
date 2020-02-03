
import json
import os
from functools import reduce
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

def flatten(arr):
    return reduce(lambda a, b: a + b, arr)

def get_bash_history():
    f = open(os.environ["HOME"] + "/.bash_history")
    lines = f.readlines()
    f.close()
    return lines

def get_commands(lines):
    commands = []
    for line in lines:
        command = line[:-1]
        if command is not "" and command[-1] == " ":
            command = command[:-1]
        commands.append(command)
    return commands

def make_dataset(commands, maxlen, step):
    input_commands = []
    next_commands = []
    for i in range(0, len(commands) - maxlen, step):
        input_commands.append(commands[i: i + maxlen])
        next_commands.append(commands[i + maxlen].split(" ")[0])
    return (input_commands, next_commands)

def make_param_dataset(commands, maxlen, step):
    input_commands = []
    params = []
    for i in range(0, len(commands) - maxlen, step):
        cmd_arr = commands[i + maxlen].split(" ")
        if len(cmd_arr) > 1:
            input_commands.append(commands[i: i + maxlen])
            params.append(cmd_arr[1])
    return (input_commands, params)

def make_next_command_list(next_commands):
    next_command_list = sorted(list(set(next_commands)))
    return next_command_list

def make_input_command_list(input_commands):
    input_command_list = sorted(list(set(flatten(input_commands))))
    return input_command_list

def make_param_list(params):
    param_list = sorted(list(set(params)))
    return param_list

def make_param_dicts(param_list):
    p_i = dict((param, i) for i, param in enumerate(command_list))
    i_p = dict((i, pararm) for i, param in enumerate(command_list))
    return (p_i, i_p)


def make_command_dicts(command_list):
    c_i = dict((command, i) for i, command in enumerate(command_list))
    i_c = dict((i, command) for i, command in enumerate(command_list))
    return (c_i, i_c)
lines = get_bash_history()
commands = get_commands(lines)

def learn_commands():
    maxlen = 2
    step = 1
    input_commands, next_commands = make_dataset(commands, maxlen, step)
    input_command_list = make_input_command_list(input_commands)
    next_command_list = make_next_command_list(next_commands)
    input_cmd_to_i, input_i_to_cmd = make_command_dicts(input_command_list)
    next_cmd_to_i, next_i_to_cmd = make_command_dicts(next_command_list)
    X = np.zeros((len(input_commands), maxlen, len(input_command_list)), dtype=np.bool)
    y = np.zeros((len(next_commands), len(next_command_list)), dtype=np.bool)
    for i, cmds in enumerate(input_commands):
        for t in range(maxlen):
            X[i, t, input_cmd_to_i[cmds[t]]] = 1 
        y[i, next_cmd_to_i[next_commands[i]]] = 1
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(input_command_list))))
    model.add(Dense(len(next_command_list)))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=30, verbose=1)
    model.save("model_lstm.h5")
    model.save_weights("model_lstm_weight.h5")
    with open("input_commands.json", "w") as f:
        json.dump(input_i_to_cmd, f, ensure_ascii=False)
    with open("next_commands.json", "w") as f:
        json.dump(next_i_to_cmd, f, ensure_ascii=False)
def learn_params():
    maxlen = 2
    step = 1
    input_commands, params = make_param_dataset(commands, maxlen, step)
    input_command_list = make_input_command_list(input_commands)
    param_list = make_param_list(params)
    input_cmd_to_i, input_i_to_cmd = make_command_dicts(input_command_list)
    param_to_i, i_to_param = make_command_dicts(param_list)
    X = np.zeros((len(input_commands), maxlen, len(input_command_list)), dtype=np.bool)
    y = np.zeros((len(params), len(param_list)), dtype=np.bool)
    for i, cmds in enumerate(input_commands):
        for t in range(maxlen):
            X[i, t, input_cmd_to_i[cmds[t]]] = 1 
        y[i, param_to_i[params[i]]] = 1
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(input_command_list))))
    model.add(Dense(len(param_list)))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=30, verbose=1)
    model.save("model_lstm_param.h5")
    model.save_weights("model_lstm_param_weight.h5")
    with open("params_input.json", "w") as f:
        json.dump(input_i_to_cmd, f, ensure_ascii=False)
    with open("params.json", "w") as f:
        json.dump(i_to_param, f, ensure_ascii=False)

learn_commands()
learn_params()
