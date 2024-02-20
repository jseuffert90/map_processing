#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import *

import argparse
import logging
import multiprocessing
from multiprocessing import Process
import os
import sys

import numpy as np
import tqdm

def run(id_counter, source_files, x_map, procs_graph_sum_errors, procs_graph_count_errors, bin_edges, logger, proc_id):
    
    procs_graph_sum_errors[proc_id] = np.zeros(len(bin_edges)-1, dtype=np.float64)
    procs_graph_count_errors[proc_id] = np.zeros(len(bin_edges)-1, dtype=np.float64)

    with tqdm.tqdm(total=len(source_files), disable=(proc_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(source_files):
            input_path = source_files[cur_id]
            y_map = read_data(input_path).squeeze()
            if y_map.shape != x_map.shape:
                raise Exception("input map must have same shape as x map")

            valid_map = y_map == y_map
            valid_map *= y_map > -np.inf
            valid_map *= y_map < +np.inf

            xs = x_map[valid_map]
            ys = y_map[valid_map]

            cur_graph_sum_errors = np.zeros(len(bin_edges)-1, dtype=np.float64)
            cur_graph_count_errors = np.zeros(len(bin_edges)-1, dtype=np.float64)

            for i in range(len(bin_edges) - 1):
                lower = bin_edges[i]
                upper = bin_edges[i+1]
                if i == len(bin_edges) - 1:
                    interesting_xs = (xs >= lower) * (xs <= upper)
                else:
                    interesting_xs = (xs >= lower) * (xs < upper)
                cur_graph_sum_errors[i] += np.sum(ys[interesting_xs])
                cur_graph_count_errors[i] += np.sum(interesting_xs)

            procs_graph_sum_errors[proc_id] += cur_graph_sum_errors
            procs_graph_count_errors[proc_id] += cur_graph_count_errors

            # simulate overflow:
            if np.any(procs_graph_sum_errors[proc_id] > 0.8 * sys.float_info.max):
                logger.warning("Upcoming numberic overflow")
            if np.any(procs_graph_count_errors[proc_id] > 0.8 * sys.float_info.max):
                logger.warning("Upcoming numberic overflow")


            cur_id = id_counter.getAndInc()
            pbar.n = min(len(source_files), cur_id+1)
            pbar.refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates a graph mapping x-map's values to the mean values of all y-maps")
    
    parser.add_argument('--loglevel', '-l', \
            choices=['critical', 'error', 'warning', 'info', 'debug'], \
            default="warning", type=str, help="set the log level (default: warning)")
    parser.add_argument('--max_x', type=float, help="max x value; determined if not provided")
    parser.add_argument('--max_x_deg', type=float, help="max x value [degrees]; replaces --max_x; max_x = max_x_deg * pi / 180")
    parser.add_argument('--min_x', type=float, help="min x value; determined if not provided")
    parser.add_argument('--min_x_deg', type=float, help="min x value [degrees]; replaces --min_x; min_x = min_x_deg * pi / 180")
    parser.add_argument('--num_bins', default=20, type=int, help="number of bins between mimimum and maxumum x value")
    parser.add_argument('--num_procs', default=multiprocessing.cpu_count(), type=int, \
            help="number of worker processes (def: #cpus)")
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    parser.add_argument('--output_file_bin_edges', type=str, default='bin_edges.npy', help='filename bin edges of graph')
    parser.add_argument('--output_file_graph', type=str, default='graph.npy', help='filename graph values')
    parser.add_argument('--output_file_graph_txt', type=str, default='graph.txt', \
            help='filename graph txt file [from | to | mean error]')
    parser.add_argument('--stop_after', type=int, help="stop after n samples")
    parser.add_argument('--x_map', '-x', type=str, help="x map", required=True)

    parser.add_argument('files', metavar='FILE', type=str, nargs="+", help="input files with y values")

    args = parser.parse_args()

    log_level = getattr(logging, args.loglevel.upper())
    logger = logging.getLogger('xy_mapper')
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)
    
    source_files = args.files
    if args.stop_after is not None and args.stop_after > 0:
        source_files = source_files[:min(args.stop_after, len(source_files))]

    if args.min_x_deg is not None:
        if args.min_x is not None:
            logger.error("Cannot process both --min_x and --min_x_deg")
            exit(1)
        args.min_x = args.min_x_deg * np.pi / 180.0
    
    if args.max_x_deg is not None:
        if args.max_x is not None:
            logger.error("Cannot process both --max_x and --max_x_deg")
            exit(1)
        args.max_x = args.max_x_deg * np.pi / 180.0

    if len(source_files) == 0:
        logger.error("There should be at least one input file!")
        exit(1)

    id_counter = AtomicIntegerProc(0)
    manager = multiprocessing.Manager()
    procs_graph_sum_errors = manager.dict()
    procs_graph_count_errors = manager.dict()

    x_map = read_data(args.x_map)
    x_map = x_map.squeeze()

    x_min = args.min_x if args.min_x is not None else np.min(x_map)
    x_max = args.max_x if args.max_x is not None else np.max(x_map)

    logger.info(f"minimum x is {x_min}")
    logger.info(f"maximum x is {x_min}")

    if x_min in [np.inf, -np.inf] or x_min != x_min:
        logger.error("mimimum x value must be a real number")
        exit(1)
    if x_max in [np.inf, -np.inf] or x_max != x_max:
        logger.error("maximum x value must be a real number")
        exit(1)
    
    bin_edges = np.linspace(x_min, x_max, args.num_bins+1)

    procs = []
    for proc_id in range(args.num_procs):
        my_proc = Process(target=run, args=(id_counter, source_files, x_map, procs_graph_sum_errors, \
                procs_graph_count_errors, bin_edges, logger, proc_id))
        procs.append(my_proc)
        my_proc.start()

    for proc_id in range(args.num_procs):
        procs[proc_id].join()
   
    graph_sum_errors = np.zeros(len(bin_edges)-1, dtype=np.float64)
    graph_count_errors = np.zeros(len(bin_edges)-1, dtype=np.float64)

    for proc_id in range(args.num_procs):
        graph_sum_errors += procs_graph_sum_errors[proc_id]
        if np.any(graph_sum_errors > 0.8 * sys.float_info.max):
            logger.warning("Upcoming numberic overflow")
        graph_count_errors += procs_graph_count_errors[proc_id]
        if np.any(graph_count_errors > 0.8 * sys.float_info.max):
            logger.warning("Upcoming numberic overflow")

    graph_mean_error = np.zeros_like(graph_sum_errors)
    graph_mean_error[graph_count_errors > 0] = \
            graph_sum_errors[graph_count_errors > 0] / graph_count_errors[graph_count_errors > 0]
    
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, args.output_file_bin_edges), bin_edges)
    np.save(os.path.join(args.output_dir, args.output_file_graph), graph_mean_error)

    with open(os.path.join(args.output_dir, args.output_file_graph_txt), 'w') as graph_out:
        for i in range(0, args.num_bins):
            bin_from = bin_edges[i]
            bin_to = bin_edges[i+1]
            graph_out.write(f"{bin_from:0.04f}\t{bin_to:0.04f}\t{graph_mean_error[i]}\n")
