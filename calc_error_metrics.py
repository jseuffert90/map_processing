#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import AtomicIntegerProc

import argparse
import glob
import logging
import multiprocessing
from multiprocessing import Process, Array
import os
import re
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
import tqdm

LOGGER_NAME = 'calc error metrics'

def get_sample_id(filename):
    numbers = re.findall(r'\d+', filename)
    logger = logging.getLogger(LOGGER_NAME)
    if len(numbers) == 0:
        logger.error(f"Could not determine sample id of file {filename}")
        exit(1)
    num_digits = [len(n) for n in numbers]
    num_digits = np.array(num_digits)
    id_longest_number = np.argmax(num_digits)
    return int(numbers[id_longest_number])

def get_valid(error_map):
    valid = (error_map == error_map)
    with np.errstate(invalid='ignore'):
        # nan values in error map cause unimportant warnings --> ignore
        valid *= (error_map >= 0)
        valid *= (error_map < np.inf)
    return valid

def get_joint_valid_mask(error_maps):
    valid = None
    masks = {}
    for name, error_map in error_maps.items():
        if valid is None:
            valid = get_valid(error_map)
        else:
            valid *= get_valid(error_map)
        masks[name] = valid
    assert valid is not None
    return valid, masks

def run(samples, sample_ids, sample_id_counter, mae_values, rmse_values, args, proc_id, logger_lock):
    logger = logging.getLogger(LOGGER_NAME)
    with tqdm.tqdm(total=len(sample_ids), disable=(proc_id > 0)) as pbar:
        sample_id_pos = sample_id_counter.getAndInc()
        while sample_id_pos < len(sample_ids):
            if args.stop_after > 0 and sample_id_pos >= args.stop_after:
                break
            sample_id = sample_ids[sample_id_pos]
            files_cur_sample = samples[sample_id]
            maps_cur_sample = {}
            height, width = -1, -1

            for name in files_cur_sample:
                cur_map = read_data(files_cur_sample[name]).squeeze()
                assert len(cur_map.shape) == 2
                if height == -1:
                    height, width = cur_map.shape
                if cur_map.shape[0] != height:
                    with logger_lock:
                        logger.error("File {files_cur_sample[name]} expected to have a height of {height} not {cur_map.shape[0]}")
                if cur_map.shape[1] != width:
                    with logger_lock:
                        logger.error("File {files_cur_sample[name]} expected to have a width of {width} not {cur_map.shape[1]}")
                maps_cur_sample[name] = cur_map
            
            joint_valid_mask, masks = get_joint_valid_mask(maps_cur_sample)

            if args.show_samples:
                fig, axs = plt.subplots(2, num_sample_dirs)

            log_str_mae = f"MAE {sample_id} "
            log_str_rmse = f"RMSE {sample_id} "
            for pos, name in enumerate(args.names):
                error_map = maps_cur_sample[name]
                cur_error_map_joint_valid = np.copy(error_map)
                cur_error_map_joint_valid[np.bitwise_not(joint_valid_mask)] = float('nan')
                
                mae_v = np.mean(error_map[joint_valid_mask])
                mae_values[pos] += mae_v
                rmse_v = np.sqrt(np.mean(np.square(error_map[joint_valid_mask])))
                rmse_values[pos] += rmse_v
                
                mae_own_mask = np.mean(error_map[masks[name]])

                max_error_own_mask = np.max(error_map[masks[name]])
                # print(f"{max_error_own_mask=}")
                rmse_own_mask = np.sqrt(np.mean(np.square(error_map[masks[name]])))
                
                log_str_mae += f"{name} {mae_v:.05f}, "
                log_str_rmse += f"{name} {rmse_v:.05f}, "
                if args.show_samples:
                    handle_own_mask = axs[0, pos].imshow(error_map)
                    axs[0, pos].set_title(f'{name} - own mask')
                    axs[0, pos].text(10, 60, f'MAE = {mae_own_mask:.05f}')
                    axs[0, pos].text(10, 110, f'RMSE = {rmse_own_mask:.05f}')
                    handle_joint_mask = axs[1, pos].imshow(cur_error_map_joint_valid)
                    axs[1, pos].set_title(f'{name} - joint mask')
                    axs[1, pos].text(10, 60, f'MAE = {mae_v:.05f}')
                    axs[1, pos].text(10, 110, f'RMSE = {rmse_v:05f}')
            if args.show_samples:
                plt.show()

            log_str_mae = log_str_mae[:-2]
            log_str_rmse = log_str_rmse[:-2]
            with logger_lock:
                logger.debug(log_str_mae)
                logger.debug(log_str_rmse)

            pbar.n = min(len(sample_ids), sample_id_pos+1)
            pbar.refresh()
            sample_id_pos = sample_id_counter.getAndInc()

def main():
    parser = argparse.ArgumentParser(description='Joint error calculation')
    parser.add_argument('--debug', '-d', action="store_true", help='activate debug mode')
    parser.add_argument('--inputdirs', '-i', required=True, nargs="+", help='input map directories')
    parser.add_argument('--inputfileregex', '-r', required=True, nargs="+", help='input map filename regex')
    parser.add_argument('--logfile', '-l', type=str, help='optional: output log file')
    parser.add_argument('--names', '-n', required=True, nargs="+", help='names of the experiments')
    parser.add_argument('--num_procs', default=multiprocessing.cpu_count(), type=int, help="number of worker procs")
    parser.add_argument('--output', '-o', required=True, default="results.xml", help='errors output file')
    parser.add_argument('--stop_after', '-s', default=-1, type=int, help='stop after n samples')
    parser.add_argument('--show_samples', action="store_true", help='show masked maps of the samples')
    args = parser.parse_args()
    
    logger = logging.getLogger(LOGGER_NAME)
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logger.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if args.logfile is not None:
        file_handler = logging.FileHandler(args.logfile)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    logger.debug("test")

    if args.output[-4:].lower() != ".xml":
        logger.error("Output file must be an xml file")
        exit(1)

    if os.path.exists(args.output):
        logger.error(f"{args.output} alredy exists. Cannot continue... Please remove or rename this file first.")
        exit(1)

    num_sample_dirs = len(args.inputdirs)
    if num_sample_dirs != len(args.inputfileregex):
        logger.error("The number of input directories and regex must be equal")
        exit(1)
    
    if num_sample_dirs != len(args.names):
        logger.error("For each input directory there must be a descriptive experiment name")
        exit(1)

    num_samples = -1
    samples = {}
    for i in range(num_sample_dirs):
        cur_name = args.names[i]
        cur_input_dir = args.inputdirs[i]
        cur_input_regex = args.inputfileregex[i]

        cur_input_files = glob.glob(os.path.join(cur_input_dir, cur_input_regex))
        if num_samples == -1:
            num_samples = len(cur_input_files)

        if len(cur_input_files) == 0:
            logger.error(f"No input files found in directory {cur_input_dir} with pattern '{cur_input_regex}'")
            exit(1)
        if len(cur_input_files) != num_samples:
            logger.error(f"Input directory {cur_input_dir} contains more samples than expected (expected: {num_samples}; found: {len(cur_input_files)})")
            exit(1)

        cur_input_file_ids = [get_sample_id(os.path.basename(path)) for path in cur_input_files]
        for s in range(num_samples):
            sid = cur_input_file_ids[s]
            cur_input_file = cur_input_files[s]
            if sid not in samples:
                samples[sid] = {}
            if cur_name in samples[sid]:
                logger.error(f"found duplicate for {cur_name}:")
                logger.error(f"{samples[sid][cur_name]}")
                logger.error(f"{cur_input_file}")
                exit(1)
            samples[sid][cur_name] = cur_input_file

    sample_ids =  list(samples.keys())
    
    for sample_id in sample_ids:
        assert len(samples[sample_id]) == num_sample_dirs, f"{sample_id=}; {num_sample_dirs=}"

    if args.debug:
        debug_sample = sample_ids[0]
        logger.debug(f"debug sample {debug_sample} tuple:")
        logger.debug(samples[debug_sample])

    #mae_values = {name : Value0.0 for name in args.names}
    #rmse_values = {name : 0.0 for name in args.names}

    mae_values = Array('d', len(args.names))
    rmse_values = Array('d', len(args.names))
 

    sample_id_counter = AtomicIntegerProc(0)
    procs = []
    logger_lock = multiprocessing.Lock()

    if args.num_procs == 1:
        run(samples, sample_ids, sample_id_counter, mae_values, rmse_values, args, 0, logger_lock)
    else:
        for proc_id in range(args.num_procs):
            my_proc = Process(target=run, args=(samples, sample_ids, sample_id_counter, mae_values, rmse_values, args, proc_id, logger_lock))
            procs.append(my_proc)
            my_proc.start()

        for proc_id in range(args.num_procs):
            procs[proc_id].join()

    for pos in range(len(args.names)):
        mae_values[pos] /= num_samples
        rmse_values[pos] /= num_samples

    results = ET.Element('results')
    mae_elem = ET.SubElement(results, 'error_metric')
    mae_elem.attrib['name'] = 'mae'
    rmse_elem = ET.SubElement(results, 'error_metric')
    rmse_elem.attrib['name'] = 'rmse'
    
    for pos, name in enumerate(args.names):
        cur_mae = ET.SubElement(mae_elem, "error_value")
        cur_mae.attrib["src"] = name
        cur_mae.text = str(mae_values[pos])
    
        cur_rmse = ET.SubElement(rmse_elem, "error_value")
        cur_rmse.attrib["src"] = name
        cur_rmse.text = str(rmse_values[pos])
        
        logger.info(f"Average MAE {name} {mae_values[pos]}")
        logger.info(f"Average RMSE {name} {rmse_values[pos]}")

    pretty_xml_str = minidom.parseString(ET.tostring(results)).toprettyxml(indent = " " * 4)
    with open(args.output, "w") as xml_file:
        xml_file.write(pretty_xml_str)

    logger.debug("Done")

if __name__ == "__main__":
    main()
