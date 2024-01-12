#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import *
from map_proc.pointcloud import *
from map_proc.proj_models import *


import argparse
import logging
from multiprocessing import Process
import os
import re

import glob
import numpy as np
from PIL import Image
import tifffile
import tqdm

LOGGER_NAME = 'ERROR MAP GENERATOR'

def run(id_counter, input_maps, gt_maps, abs_target_files, rel_target_files, process_id, args):
    logger = logging.getLogger(LOGGER_NAME)

    with tqdm.tqdm(total=len(input_maps), disable=(process_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(input_maps):
            input_path = input_maps[cur_id]
            gt_path = gt_maps[cur_id]
            abs_error_path = abs_target_files[cur_id]
            rel_error_path = rel_target_files[cur_id]

            input_map = read_data(input_path)
            gt_map = read_data(gt_path)

            if input_map.shape != gt_map.shape:
                raise ValueError("input files must have the same dimensions as gt files.")

            valid_mask  = input_map == input_map
            valid_mask *= np.abs(input_map) < np.inf
            valid_mask *= gt_map == gt_map
            valid_mask *= np.abs(gt_map) < np.inf

            abs_error_map = np.zeros_like(input_map) * float('nan')
            rel_error_map = np.zeros_like(input_map) * float('nan')

            abs_error_map[valid_mask] = np.abs(input_map[valid_mask] - gt_map[valid_mask])
            rel_error_map[valid_mask] = abs_error_map[valid_mask] / gt_map[valid_mask]

            if args.dryrun:
                logger.debug(f"write file {abs_error_path} ... SKIPPED (dryrun)")
                logger.debug(f"write file {rel_error_path} ... SKIPPED (dryrun)")
            else:
                logger.debug(f"write file {abs_error_path}")
                write_data(abs_error_path, abs_error_map)
                logger.debug(f"write file {rel_error_path}")
                write_data(rel_error_path, rel_error_map)
            
            cur_id = id_counter.getAndInc()
            pbar.n = min(len(input_maps), cur_id+1)
            pbar.refresh()

def get_fid(fname: str, logger=None):
    all_numbers_as_str = re.findall(r'\d+', fname)
    fid = int(sorted(all_numbers_as_str, key=len)[-1])
    print(f"{fid=}")
    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=LOGGER_NAME)
    parser.add_argument('--abs', action='store_true', help="produce absolute error maps")
    parser.add_argument('--abs_error_file_pattern', '-a', metavar='f-string pattern', type=str, help="output file pattern for absolute error maps," + \
            "e.g. abs_euc_dist_error_map_fid_{fid:07d}.tiff", required=True)
    parser.add_argument('--dryrun', '-d', action="store_true", help="does not save files nor create directories")
    parser.add_argument('--gt_dir', '-g', metavar="dir", type=str, help="input ground truth maps", required=True)
    parser.add_argument('--gt_file_pattern', '-p', metavar='f-string pattern', type=str, help="gt file pattern, e.g. {fid}.exr", required=True)
    parser.add_argument('--input_dir', '-i', metavar="dir", type=str, help="input maps", required=True)
    parser.add_argument('--input_ext', '-x', type=str, help="file extension for maps in input map directories to consider, e.g. '.tiff'; " \
            + "if not provided: derived from output file extension")
    parser.add_argument('--log_level', '-l', type=str, default="warning", \
            choices=['critical', 'error', 'warning', 'info', 'debug'], help="log level of converter")
    parser.add_argument('--num_processes', '-n', default=8, type=int, help="number of worker processes")
    parser.add_argument('--output_dir', '-o', metavar="dir", type=str, help="output maps directory", required=True)
    parser.add_argument('--rel', action='store_true', help="produce relative error maps")
    parser.add_argument('--rel_error_file_pattern', '-r', metavar='f-string pattern', type=str, help="output file pattern for relative error maps," + \
            "e.g. abs_euc_dist_error_map_fid_{fid:07d}.tiff", required=True)
    parser.add_argument('--stop_after', '-s', type=int, metavar="N", help="stop after processing N files")

    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)

    if args.abs is False and args.rel is False:
        logger.error("Please provide either --abs or --rel.")
        exit(1)
    
    # is abs_error_file_pattern secure?
    test_p = args.abs_error_file_pattern.replace('{fid}','')
    test_p = test_p.replace('{fid:','')
    if test_p.find('{') >= 0:
        logger.error("The file pattern for --abs_error_file_pattern is not supported.")
        exit(1)
    
    # is rel_error_file_pattern secure?
    test_p = args.rel_error_file_pattern.replace('{fid}','')
    test_p = test_p.replace('{fid:','')
    if test_p.find('{') >= 0:
        logger.error("The file pattern for --rel_error_file_pattern is not supported.")
        exit(1)
    
    # is gt_file_pattern secure?
    test_p = args.gt_file_pattern.replace('{fid}','')
    test_p = test_p.replace('{fid:','')
    if test_p.find('{') >= 0:
        logger.error("The file pattern for --gt_file_pattern is not supported.")
        exit(1)

    if args.input_ext is None:
        _, args.input_ext = os.path.splitext(args.abs_error_file_pattern)
        logger.info(f"Set --input_ext to '{args.input_ext}'")

    processes = []
    file_id_counter = AtomicIntegerProc(0)

    all_input_maps = []
    all_gt_maps = []
    all_abs_target_files = []
    all_rel_target_files = []

    for root, dirs, files in os.walk(args.input_dir):
        logger.debug(f"{root=}")
        #logger.debug(f"{files=}")
        logger.debug(f"{dirs=}")
        for f in files:
            logger.debug(f"found file {f}")
            if os.path.splitext(f)[1] != args.input_ext:
                logger.debug(f"filter out {f}")
                continue
            cur_in_fname = f
            cur_in_path = os.path.join(root, cur_in_fname)
            logger.debug(f"{cur_in_path=}")
            all_input_maps.append(cur_in_path)
            
            fid = get_fid(cur_in_fname)
            cur_gt_fname = eval("f'{}'".format(args.gt_file_pattern))
            cur_gt_path = os.path.join(args.gt_dir, cur_gt_fname)
            if not os.path.isfile(cur_gt_path):
                logger.error(f"GT file {cur_gt_path} does not exist!")
                exit(1)
            all_gt_maps.append(cur_gt_path)

            abs_error_fname = eval("f'{}'".format(args.abs_error_file_pattern))
            abs_error_path = os.path.join(args.output_dir, abs_error_fname)
            all_abs_target_files.append(abs_error_path)
            
            rel_error_fname = eval("f'{}'".format(args.rel_error_file_pattern))
            rel_error_path = os.path.join(args.output_dir, rel_error_fname)
            all_rel_target_files.append(rel_error_path)
        break

    if args.stop_after is not None:
        all_input_maps = all_input_maps[:args.stop_after]
        all_gt_maps = all_gt_maps[:args.stop_after]
        all_abs_target_files = all_abs_target_files[:args.stop_after]
        all_rel_target_files = all_rel_target_files[:args.stop_after]

    logger.debug("f{all_input_maps=}")
    logger.debug("f{all_gt_maps=}")
    logger.debug("f{all_abs_target_files=}")
    logger.debug("f{all_rel_target_files=}")

    for process_id in range(args.num_processes):
        my_process = Process(target=run, args=(file_id_counter, all_input_maps, all_gt_maps, all_abs_target_files, all_rel_target_files, process_id, args))
        processes.append(my_process)
        my_process.start()
    
    for process_id in range(args.num_processes):
        processes[process_id].join()
