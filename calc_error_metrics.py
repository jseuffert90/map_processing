#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import re
import tifffile
import xml.etree.ElementTree as ET
from xml.dom import minidom

import matplotlib.pyplot as plt
import numpy as np
import tqdm

def get_sample_id(filename):
    numbers = re.findall(r'\d+', filename)
    if len(numbers) == 0:
        logging.error(f"Could not determine sample id of file {filename}")
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


def main():
    parser = argparse.ArgumentParser(description='Joint error calculation')
    parser.add_argument('--debug', '-d', action="store_true", help='activate debug mode')
    parser.add_argument('--inputdirs', '-i', required=True, nargs="+", help='input map directories')
    parser.add_argument('--inputfileregex', '-r', required=True, nargs="+", help='input map filename regex')
    parser.add_argument('--names', '-n', required=True, nargs="+", help='names of the experiments')
    parser.add_argument('--output', '-o', required=True, default="results.xml", help='errors output file')
    parser.add_argument('--stop_after', '-s', default=-1, type=int, help='stop after n samples')
    parser.add_argument('--show_samples', action="store_true", help='show masked maps of the samples')
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()
    
    if args.output[-4:].lower() != ".xml":
        logging.error("Output file must be an xml file")
        exit(1)

    if os.path.exists(args.output):
        logging.error(f"{args.output} alredy exists. Cannot continue... Please remove or rename this file first.")
        exit(1)

    num_sample_dirs = len(args.inputdirs)
    if num_sample_dirs != len(args.inputfileregex):
        logging.error("The number of input directories and regex must be equal")
        exit(1)
    
    if num_sample_dirs != len(args.names):
        logging.error("For each input directory there must be a descriptive experiment name")
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
            logging.error(f"No input files found in directory {cur_input_dir} with pattern '{cur_input_regex}'")
            exit(1)
        if len(cur_input_files) != num_samples:
            logging.error(f"Input directory {cur_input_dir} contains more samples than expected (expected: {num_samples}; found: {len(cur_input_files)})")
            exit(1)

        cur_input_file_ids = [get_sample_id(os.path.basename(path)) for path in cur_input_files]
        for s in range(num_samples):
            sid = cur_input_file_ids[s]
            cur_input_file = cur_input_files[s]
            if sid not in samples:
                samples[sid] = {}
            if cur_name in samples[sid]:
                logging.error(f"found duplicate for {name}:")
                logging.error(f"{samples[sid][cur_name]}")
                logging.error(f"{cur_input_file}")
                exit(1)
            samples[sid][cur_name] = cur_input_file

    sample_ids =  list(samples.keys())
    
    for sample_id in sample_ids:
        assert len(samples[sample_id]) == num_sample_dirs, f"{sample_id=}; {num_sample_dirs=}"

    if args.debug:
        debug_sample = sample_ids[0]
        logging.debug(f"debug sample {debug_sample} tuple:")
        logging.debug(samples[debug_sample])

    mae_values = {name : 0.0 for name in args.names}
    rmse_values = {name : 0.0 for name in args.names}
   
    for sample_count, sample_id in enumerate(tqdm.tqdm(sample_ids)):
        if args.stop_after > 0 and sample_count >= args.stop_after:
            break

        files_cur_sample = samples[sample_id]
        maps_cur_sample = {}
        height, width = -1, -1

        for name in files_cur_sample:
            cur_map = tifffile.imread(files_cur_sample[name]).squeeze()
            assert len(cur_map.shape) == 2
            if height == -1:
                height, width = cur_map.shape
            if cur_map.shape[0] != height:
                logging.error("File {files_cur_sample[name]} expected to have a height of {height} not {cur_map.shape[0]}")
            if cur_map.shape[1] != width:
                logging.error("File {files_cur_sample[name]} expected to have a width of {width} not {cur_map.shape[1]}")
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
            mae_values[name] += mae_v
            rmse_v = np.sqrt(np.mean(np.square(error_map[joint_valid_mask])))
            rmse_values[name] += rmse_v
            
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
        logging.debug(log_str_mae)
        logging.debug(log_str_rmse)

    for name in args.names:
        mae_values[name] /= num_samples
        rmse_values[name] /= num_samples

    results = ET.Element('results')
    mae_elem = ET.SubElement(results, 'error_metric')
    mae_elem.attrib['name'] = 'mae'
    rmse_elem = ET.SubElement(results, 'error_metric')
    rmse_elem.attrib['name'] = 'rmse'
    
    for name in args.names:
        cur_mae = ET.SubElement(mae_elem, "error_value")
        cur_mae.attrib["src"] = name
        cur_mae.text = str(mae_values[name])
    
    for name in args.names:
        cur_rmse = ET.SubElement(rmse_elem, "error_value")
        cur_rmse.attrib["src"] = name
        cur_rmse.text = str(rmse_values[name])
        
    logging.info(f"Average MAE {name} {mae_values[name]}")
    logging.info(f"Average RMSE {name} {rmse_values[name]}")

    pretty_xml_str = minidom.parseString(ET.tostring(results)).toprettyxml(indent = " " * 4)
    with open(args.output, "w") as xml_file:
        xml_file.write(pretty_xml_str)

    logging.debug("Done")

if __name__ == "__main__":
    main()
