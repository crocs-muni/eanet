#!/usr/bin/env python3

import os
import sys
import signal
import shlex, subprocess
import json
import logging
import argparse


def main(script_path):
    batch_sizes = [10,25,50,100,250,500,1000]
    layouts = 5
    res_matrix = [[0 for x in range(len(batch_sizes))] for y in range(layouts)]
    tv_size = 16

    for i in range(len(batch_sizes)):
        for j in range(layouts):
            logging.info('Testing batch_size: ' + str(batch_sizes[i]) + ' layout ' + str(j))

            f = subprocess.Popen('python ' + script_path + ' --in_a AES_r10_b16.bin --in_b Salsa20_r2_b16.bin --tv_size 16 --batch_size ' + str(batch_sizes[i]) + ' --layout ' + str(j), shell=True, stdout=subprocess.PIPE).stdout

            progress_res = []
            res = None
            epoch = 0
            for l in f:
                line = l.decode('ascii')
                if 'loss' in line:
                    # todo parse loss and acc
                    res = line.split('[==============================]')[-1]
                    loss = float(res.split('loss: ')[1].split(' ')[0])
                    acc = float(res.split('acc: ')[1].split(' ')[0])
                    progress_res.append((loss, acc))
                if 'Score' in line:
                    res = line.split('[')[-1]
                    loss = float(res.split(',')[0])
                    acc = float(res.split(',')[1].split(']')[0])
                    res = (loss, acc)

            res_matrix[j][i] = (progress_res, res)

    with open(script_path + '_result.json', 'w') as fp:
        json.dump(res_matrix, fp)

    with open(script_path + '_result.txt', 'w') as fp:
        fp.write(str(res_matrix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--script',
        type=str,
        default='./eanet.py',
        help='Path to python script for run'
    )
    FLAGS, unparsed = parser.parse_known_args()
    logging.basicConfig(filename='ann_runner.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('Started script: ' + FLAGS.script)
    main(FLAGS.script)
    logging.info('Ended script: ' + FLAGS.script)
    
