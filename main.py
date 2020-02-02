
import json
import os

def get_bash_history():
    f = open(os.environ["HOME"] + "/.bash_history")
    lines = f.readlines()
    f.close()
    return lines

def get_commands(lines):
    commands = []
    for line in lines:
        command = line[:-1]
        if command[-1] == " ":
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

def make_command_list(input_commands):
    command_list = sorted(list(set(next_commands)))
    return command_list

def make_command_dicts(command_list):
    c_i = dict((command, i) for i, command in enumerate(command_list))
    i_c = dict((i, command) for i, command in enumerate(command_list))
    return (c_i, i_c)

maxlen = 2
step = 1
lines = get_bash_history()
commands = get_commands(lines)
input_commands, next_commands = prepare_dataset(commands, maxlen)
command_list = make_command_list(input_commands)
c_i, i_c = make_command_dicts(command_list)


import numpy as np
X = np.zeros((len(input_commands), maxlen, len(command_list)), dtype=np.bool)
y = np.zeros((len(input_commands), len(command_list)), dtype=np.bool)

for i, cmds in enumerate(input_commands):
    for t in range(maxlen):
        X[i, t, c_i[cmds[t].split(" ")[0]]] = 1 
    y[i, c_i[next_commands[i]]] = 1

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(command_list))))
model.add(Dense(len(command_list)))
model.add(Activation("softmax"))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=30, verbose=1)

model.save("model_lstm.h5")
model.save_weights("model_lstm_weight.h5")

print(c_i)
with open("commands.json", "w") as f:
    json.dump(i_c, f, ensure_ascii=False)
