"""Converts EAC data to numpy array."""
from __future__ import absolute_import
from __future__ import division

import argparse
import os.path
import sys
import time
import random

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

FLAGS = None

tv_size = 16

nb_epoch = 100
batch_size = 25

layouts = [ [1], [16,1], [8,4,2,1], [8,8,8,1], [16,8,4,2,1] ]
layout = 0

model = Sequential()

def evaluate(data, labels):
    # train the model, iterating on the data in batches of 32 samples
    score = model.evaluate(data, labels, batch_size=10000)
    print("Score = " + str(score))

def train(data, labels):
    l = layouts[layout]
    model.add(Dense(l[0], input_dim=tv_size*8, activation='sigmoid'))
    for i in range(1, len(l)):
        model.add(Dense(l[i], activation='sigmoid'))

    model.compile(optimizer='SGD',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, nb_epoch=nb_epoch, batch_size=batch_size)

def convert_to(in_af, in_bf, size):
    tv = []
    out_arr = np.empty((size, tv_size * 8), dtype=float)
    labels = np.random.randint(2, size=(size, 1))

    for y in range(size):
        for x in range(tv_size):
            if labels[y] == 0:
                tv = bytearray(in_af.read(1))[0]
            else:
                tv = bytearray(in_bf.read(1))[0]

            for i in range(8):
                out_arr[y][x*8 + i] = 0.5 if (tv & 2**i) else -0.5

    return out_arr, labels

def process():
    with open(FLAGS.in_a, "rb") as in_af, open(FLAGS.in_b, "rb") as in_bf:
        print("Preparing test vectors")
        t_d, t_l = convert_to(in_af, in_bf, size=batch_size*nb_epoch)
        print("Training started")
        train(t_d, t_l)
        print("Trained")
        print("Preparing final test vectors")
        e_d, e_l = convert_to(in_af, in_bf, size=10000)
        evaluate(e_d, e_l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_a',
        type=str,
        default='./in_a.bin',
        help='Stream with first source of data'
    )
    parser.add_argument(
        '--in_b',
        type=str,
        default='./in_b.bin',
        help='Stream with second source of data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=25,
        help="""\
        Batch size.\
        """
    )
    parser.add_argument(
        '--layout',
        type=int,
        default=0,
        help="""\
        Type of layout (0-4).\
        """
    )
    parser.add_argument(
        '--tv_size',
        type=int,
        default=16,
        help="""\
        Number of examples to separate from the training data for the validation
        set.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tv_size = FLAGS.tv_size
    batch_size = FLAGS.batch_size
    layout = FLAGS.layout
    process()

