
import json
import os
from functools import reduce

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

def prepare_dataset(commands, maxlen):
    input_commands = []
    next_commands = []
    for i in range(0, len(commands) - maxlen, step):
        input_commands.append(commands[i: i + maxlen])
        next_commands.append(commands[i + maxlen].split(" ")[0])
    return (input_commands, next_commands)

def make_next_command_list(next_commands):
    next_command_list = sorted(list(set(next_commands)))
    return next_command_list

def make_input_command_list(input_commands):
    input_command_list = sorted(list(set(flatten(input_commands))))
    return input_command_list


def make_command_dicts(command_list):
    c_i = dict((command, i) for i, command in enumerate(command_list))
    i_c = dict((i, command) for i, command in enumerate(command_list))
    return (c_i, i_c)

maxlen = 2
step = 1
lines = get_bash_history()
commands = get_commands(lines)
input_commands, next_commands = prepare_dataset(commands, maxlen)
input_command_list = make_input_command_list(input_commands)
next_command_list = make_next_command_list(next_commands)

input_c_i, input_i_c = make_command_dicts(input_command_list)
next_c_i, next_i_c = make_command_dicts(next_command_list)



import numpy as np
X = np.zeros((len(input_commands), maxlen, len(input_command_list)), dtype=np.bool)
y = np.zeros((len(next_commands), len(next_command_list)), dtype=np.bool)

for i, cmds in enumerate(input_commands):
    for t in range(maxlen):
        X[i, t, input_c_i[cmds[t]]] = 1 
    y[i, next_c_i[next_commands[i]]] = 1

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

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
    json.dump(input_i_c, f, ensure_ascii=False)
with open("next_commands.json", "w") as f:
    json.dump(next_i_c, f, ensure_ascii=False)
