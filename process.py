#!/usr/bin/env python3


import json

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

def process(nn, path_dist):
    for key in sorted(nn):
        name = key.split('_')[0]
        
        final_loss, final_acc = nn[key][-1]
        x, y = list(map(list, zip(*nn[key][0])))
        a = [i for i in range(len(x))]

        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)
        plt.title('Learning process on: ' + key)

        par1 = host.twinx()
        par2 = host.twinx()

        host.set_xlim(0, len(x))
        host.set_ylim(0, 1)
        par1.set_ylim(0, 1)
        par2.set_ylim(0, 1)

        host.set_xlabel("epoch")
        host.set_ylabel("acc")

        p2, = par1.plot(a, x, label="loss")
        p1, = host.plot(a, y, label="acc")
        p3, = par2.plot(a, [0.5 for _ in range(len(x))], label="rnd", ls=':', lw=2, color=p1.get_color())

        host.legend(loc=3)

        #host.axis["left"].label.set_color(p1.get_color())
        plt.grid(axis='y', linestyle=':', lw=0.5)
        plt.yticks([final_loss, final_acc])

        plt.draw()
        plt.savefig('./' + path_dist + '/' + key + '.png')
        plt.clf() #  clean thefigure

        #line_x = plt.plot(a, x, 'acc')
        #line_y = plt.plot(a, y, 'loss')
        #plt.xlabel('epoch')
        #plt.title('Learning process on: ' + key)
        #plt.setp(line_x, color='r', linewidth=1.0)
        #plt.setp(line_y, color='r', linewidth=1.0)
        #plt.gca().yaxis.grid(True)
        
        # safe plot to png

def main():
    ann, cnn = None, None
    with open('result_ann.json', 'r') as fp:
        ann = json.load(fp)
        process(ann, 'ann')

    with open('result_cnn.json', 'r') as fp:
        cnn = json.load(fp)
        process(cnn, 'cnn')


if __name__ == "__main__":
    main()
