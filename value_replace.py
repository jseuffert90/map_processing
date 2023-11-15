#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import *

import argparse
import logging
import math
import os
import sys
from multiprocessing import Process

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def run(id_counter, proc_id, args):
    with tqdm(total=len(args.files), disable=(proc_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(args.files):
            f = args.files[cur_id]
            pref, ext = os.path.splitext(f)
            ext = ext.lower()

            cur_map = read_data(f)
            cur_map = cur_map.squeeze()
            if len(cur_map.shape) != 2:
                logger.error("only maps with a single channel and a single page are supported")
                exit(1)

            for n in args.needles:
                if n != n:
                    # it's NaN
                    cur_map[cur_map != cur_map] = args.replace
                else:
                    cur_map[cur_map == n] = args.replace

            new_path = pref + "_postproc" + ext
            write_data(new_path, cur_map)
            cur_id = id_counter.getAndInc()
            pbar.n = min(len(args.files), cur_id)
            pbar.refresh()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replaces certain values in a map by another')
    # positional arguments
    parser.add_argument('files', metavar='file', type=str, nargs="+", help="input files")

    # keyword arguments
    parser.add_argument('--dontrename', '-d', action="store_true", help="do not rename output files to input files")
    parser.add_argument('--loglevel', '-l', \
            choices=['critical', 'error', 'warning', 'info', 'debug'], \
            default="info", type=str, help="set the log level")
    parser.add_argument('--needles', '-n', nargs="+", metavar='needle', type=float, help="numeric values to be replaced ('inf', ' -inf' and 'nan' also supported)", required=True)
    parser.add_argument('--num_procs', default=8, type=int, help="number of worker processes")
    parser.add_argument('--replace', '-r', type=float, help="numeric value that replaces the needle(s) ('inf', ' -inf'  and 'nan' also supported)", required=True)

    args = parser.parse_args()
    
    log_level = getattr(logging, args.loglevel.upper())
    logger = logging.getLogger('mapviewer')
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)

    logger.info("replace values...")

    procs = []
    file_id_counter = AtomicIntegerProc(0)

    for proc_id in range(args.num_procs):
        my_proc = Process(target=run, args=(file_id_counter, proc_id, args))
        procs.append(my_proc)
        my_proc.start()
    
    for proc_id in range(args.num_procs):
        procs[proc_id].join()

    logger.info("replace values... finished")
    if not args.dontrename:
        logger.info("rename files...")
        
        with tqdm(total=len(args.files)) as pbar:
            for f in args.files:
                pref, ext = os.path.splitext(f)
                if "_postproc" in pref:
                    continue
                ext = ext.lower()

                new_path = pref + "_postproc" + ext
                os.rename(new_path, f)
                pbar.update()

        logger.info("rename files... finished")


