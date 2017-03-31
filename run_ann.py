#!/usr/bin/env python3

import os
import sys
import signal
import shlex, subprocess
import json
import logging
import argparse


def main(script_path):
    res_list = {}
    tv_size = 16

    for filename in os.listdir("./"):
        if os.path.isfile(filename) and filename.endswith('.bin'):
            logging.info('Testing file: ' + filename)
            try:
                tv_size = filename.split('_')[2].split('.')[0].split('b')[1]
            except (ValueError, IndexError):
                continue

            f = subprocess.Popen('python ' + script_path + ' --in_b ' + filename + ' --tv_size ' + tv_size, shell=True, stdout=subprocess.PIPE).stdout

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

            res_list[filename] = (progress_res, res)

    with open('result.json', 'w') as fp:
        json.dump(res_list, fp)


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
    main(FLAGS.script)
