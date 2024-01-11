#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import *
from map_proc.pointcloud import *
from map_proc.proj_models import *

import argparse
import logging
from multiprocessing import Process, Manager
import os

import glob
import numpy as np
from PIL import Image
import tifffile
import tqdm

LOGGER_NAME = 'map stats'
OUT_FILE_HEADER = ['FILE', 'NUM_NAN', 'NUM_REAL', 'NUM_ZERO', 'NUM_+-INF', 'MIN', 'MAX', 'MEAN']
DELIM = ';'
NEWLINE = '\n'


def stats(input_map: np.ndarray):
    h, w = input_map.shape[:2]
    
    NaN_map = (input_map != input_map)
    num_NaNs = NaN_map.sum()
    
    zero_map = (input_map == 0)
    num_zeros = zero_map.sum()

    pm_inf_map = (np.abs(input_map) == np.inf)
    num_pm_inf = pm_inf_map.sum()

    real_map = np.abs(input_map) < np.inf
    num_real = real_map.sum()

    assert (num_real + num_pm_inf + num_NaNs) == input_map.size, f'{num_real=}, {num_pm_inf=}, {num_NaNs=}, {input_map.size=}'
    min_val = np.min(input_map[real_map])
    max_val = np.max(input_map[real_map])
    mean_val = np.mean(input_map[real_map])

    stats_out = [num_NaNs, num_real, num_zeros, num_pm_inf, min_val, max_val, mean_val]
    assert len(stats_out) == len(OUT_FILE_HEADER) - 1
   
    return stats_out

def run(id_counter, source_files, output_file_dict, proc_id, args):
    with tqdm.tqdm(total=len(source_files), disable=(proc_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(source_files):
            input_path = source_files[cur_id]
            
            data = read_data(input_path)
            #logger.debug(f"write file {output_path}")
            cur_stats = stats(data)
            cur_stats_ext = [input_path] + cur_stats
            output_file_dict[cur_id] = cur_stats_ext
            
            cur_id = id_counter.getAndInc()
            pbar.n = min(len(source_files), cur_id+1)
            pbar.refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map statistics')
    parser.add_argument('--input', '-i', metavar="file-or-dir", type=str, nargs="+", help="input maps", required=True)
    parser.add_argument('--log_level', '-l', type=str, default="warning", \
            choices=['critical', 'error', 'warning', 'info', 'debug'], help="log level of converter")
    parser.add_argument('--num_procs', '-n', default=2, type=int, help="number of worker processes")
    parser.add_argument('--output', '-o', metavar="file", type=str, help="statistics output file")
    parser.add_argument('--stop_after', '-s', type=int, metavar="N", help="stop after processing N files")
    parser.add_argument('--target_ext', '-t', type=str, help="target file extension, e.g. '.tiff'; " \
            + "if not provided: derived from input file extension")

    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)

    procs = []
    file_id_counter = AtomicIntegerProc(0)

    all_source_files = []

    for in_entry in args.input:
        if not os.path.exists(in_entry):
            logger.error(f"The input file '{in_entry}' does not exist.")
            exit(1)
        if os.path.isfile(in_entry):
            all_source_files.append(in_entry)
        elif os.path.isdir(in_entry):
            if args.target_ext is None:
                source_files = [f for f in sorted(glob.glob(f"{in_entry}/*")) if is_supported_data_file(f)]
            else:
                source_files = [f for f in sorted(glob.glob(f"{in_entry}/*")) if is_supported_data_file(f) and os.path.splitext(f)[1].lower() == args.target_ext.lower()]

            all_source_files += source_files

    all_source_files = sorted(all_source_files)

    if args.stop_after is not None:
        all_source_files = all_source_files[:args.stop_after]

    
    output_file_dict = Manager().dict()

    with open(args.output, 'w') as out_fd:
        for proc_id in range(args.num_procs):
            my_proc = Process(target=run, args=(file_id_counter, all_source_files, output_file_dict, proc_id, args))
            procs.append(my_proc)
            my_proc.start()
        
        for proc_id in range(args.num_procs):
            procs[proc_id].join()

        for j in range(len(OUT_FILE_HEADER)):
            if j != 0:
                out_fd.write(DELIM)
            out_fd.write(OUT_FILE_HEADER[j])
        out_fd.write(NEWLINE)

        
        for i in range(len(all_source_files)):
            cur_stats_ext = output_file_dict[i]
            assert len(cur_stats_ext) == len(OUT_FILE_HEADER)
            for j in range(len(cur_stats_ext)):
                if j != 0:
                    out_fd.write(DELIM)
                out_fd.write(str(cur_stats_ext[j]))
            out_fd.write(NEWLINE)
