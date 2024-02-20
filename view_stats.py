import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show histogram')
    parser.add_argument('--bins', '-b', type=str, help='bins of numpy historgram')
    parser.add_argument('--hist', '-i', type=str, help='hist of numpy historgram')
    parser.add_argument('--func', '-f', type=str, help='e.g. probability density function or cummulative density function txt file [from|to|freq]')
    parser.add_argument('--delim', '-d', type=str, default='\t', help='delimiter for --func')

    args = parser.parse_args()

    hist_and_bins_prov = (args.hist is not None and args.bins is not None)
    func_prov = (args.func is not None)

    if hist_and_bins_prov + func_prov != 1:
        print("Please provide --func xor both --bins and --hist", file=sys.stderr)
        exit(1)

    if hist_and_bins_prov:
        bins = np.load(args.bins)
        hist = np.load(args.hist)
    elif func_prov:
        bins = []
        hist = []
        with open(args.func) as func_input:
            for l in func_input:
                l = l.strip()
                bin_from, bin_to, freq = l.split(args.delim)
                if len(bins) == 0:
                    bins.append(bin_from)
                bins.append(bin_to)
                hist.append(freq)

        bins = np.array(bins, dtype=float)
        hist = np.array(hist, dtype=float)

    plt.stairs(hist, bins, fill=True)
    plt.show()
