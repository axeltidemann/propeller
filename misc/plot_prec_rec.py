import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def plot():
    H = {"0":defaultdict(list),
         "1":defaultdict(list),
         "2":defaultdict(list),
         "3":defaultdict(list)
    }

    with open(args.infile) as f:
        lines = f.readlines()
        for l in lines:
            words = l.split()
            H[words[1]]["bits"].append(int(words[0]))
            H[words[1]]["precision"].append((float(words[2])))
            H[words[1]]["recall"].append((float(words[3])))
            H[words[1]]["hits"].append((float(words[4])))
        
        
    plt.grid()
    plt.plot(H["0"]["bits"], H["0"]["precision"], 'r--', label='H0-precision', linewidth=2)
    plt.plot(H["0"]["bits"], H["0"]["recall"], 'r', label='H0-recall', linewidth=2)
    plt.plot(H["1"]["bits"], H["1"]["precision"], 'b--', label='H1-precision', linewidth=2)
    plt.plot(H["1"]["bits"], H["1"]["recall"], 'b', label='H1-recall', linewidth=2)
    plt.plot(H["2"]["bits"], H["2"]["precision"], 'g--', label='H2-precision', linewidth=2)
    plt.plot(H["2"]["bits"], H["2"]["recall"], 'g', label='H2-recall', linewidth=2)
    plt.plot(H["3"]["bits"], H["3"]["precision"], 'y--', label='H3-precision', linewidth=2)
    plt.plot(H["3"]["bits"], H["3"]["recall"], 'y', label='H3-recall', linewidth=2)
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(H["0"]["bits"], H["0"]["hits"], 'r', label='H0', linewidth=2)
    plt.plot(H["1"]["bits"], H["1"]["hits"], 'b', label='H1', linewidth=2)
    plt.plot(H["2"]["bits"], H["2"]["hits"], 'g', label='H2', linewidth=2)
    plt.plot(H["3"]["bits"], H["3"]["hits"], 'y', label='H3', linewidth=2)
    plt.yscale('log')
    plt.legend(loc='upper right')

    plt.show()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--infile',
    help="the directory containg the h5 files",
    default='precision_recall.txt'
)

args = parser.parse_args()
plot()
