import random

import numpy as np
from cubert import cubert
from NeuralNetwork import NeuralNetwork
import pickle




structure = (9*6, 32, 64, 128, 12)
structure = (9*6*6, 32, 64, 128, 12)
num_of_tests = 15000000
Net = NeuralNetwork(structure)

#Moves = [ w, w', o, o', g, g', r, r', b, b', y, y' ]


def runminibatch():
    mappings = {0: "w",
                1: "w'",
                2: "o",
                3: "o'",
                4: "g",
                5: "g'",
                6: "r",
                7: "r'",
                8: "b",
                9: "b'",
                10: "y",
                11: "y'"
                }
    colors = {
         'w': [0, 0, 0,0,0,1],
        'o': [0, 0, 0,0,1,0],
        'g': [0, 0, 0,1,0,0],
        'r': [0, 0, 1,0,0,0],
        'b': [0, 1, 0,0,0,0],
        'y': [1, 0, 0,0,0,0]
    }
    minibatch = []
    for i in range(5,6):
        z = cubert()
        for j in range(i):
            operation = random.randint(0, 11)
            z.run_moves(mappings[operation])
        x = []
        for i in range(len(z.cube)):
            x += colors[z.cube[i]]
        x = np.array(x)
        # x = np.array([colors[i] for i in z.cube])
        # inp = []
        x.resize((54*6,1))
        target = [0]*12
        target[operation] = 1
        target = np.array(target)
        target.resize((12,1))
        minibatch.append((x, target))
    loss = Net.gradient_descent(minibatch, 0.001)
    return loss

avg = 0


for i in range(num_of_tests):
    loss = runminibatch()
    # print(loss)
    if i % 5000 == 0:
        with open('data.txt', 'a') as outfile:
            outfile.write(str(avg) + "\n")
        print(avg)
        avg = 0
        print("{:,} / {:,}".format(i, num_of_tests))
    avg += loss/5000


with open('NeuralNetworkStorage.pickle', 'wb') as inputfile:
    pickle.dump(Net, inputfile)

