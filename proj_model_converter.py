#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import *
from map_proc.proj_models import *

import argparse
import json
import logging
from multiprocessing import Process
import os

#import cv2
import tqdm
import torch
import glob

PROJ_CONV_LOGGER_NAME = "Proj Model Converter"

def get_in_and_out_files(source_dir: str, target_dir: str):
    basenames = [os.path.basename(f) for f in sorted(glob.glob(f"{source_dir}/*")) if is_supported_input_file(f)]
    source_files = [os.path.join(source_dir, bname) for bname in basenames]
    target_files = [os.path.join(target_dir, bname) for bname in basenames]
    return source_files, target_files

def run(id_counter, source_files, target_files, proc_id, args):
    fov_rad_in_x = args.fov_in_x / 180.0 * np.pi
    fov_rad_in_y = args.fov_in_y / 180.0 * np.pi
    fov_rad_out_x = args.fov_out_x / 180.0 * np.pi
    fov_rad_out_y = args.fov_out_y / 180.0 * np.pi
    
    logger = logging.getLogger(PROJ_CONV_LOGGER_NAME)

    source_height, source_width, mapping = None, None, None
    target_height, target_width, rays = None, None, None

    mapping = None
    mapping = None
    
    with tqdm.tqdm(total=len(source_files), disable=(proc_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(source_files):
            input_path = source_files[cur_id]
            output_path = target_files[cur_id]

            data = read_data(input_path)
            # if data.dtype == np.float64:
            #    data = data.astype(np.float32)
            
            # new input shape: N, C , H, W
            h, w = data.shape[:2]
            data = torch.from_numpy(data)
            data_dtype = data.dtype
            data = data.to(torch.float64)
            data = data.reshape(h, w, -1)
            data = torch.permute(data, (2, 0, 1)) # H, W, C -> C, H, W
            data = data[None] # N, C, H, W
            
            if args.input_scale != 1.0:
                data *= args.input_scale

            cur_source_height, cur_source_width = data.shape[2:4]
            if args.output_width is None or args.output_height is None:
                cur_target_height, cur_target_width = cur_source_height, cur_source_width
            else:
                cur_target_height, cur_target_width = args.output_height, args.output_width

            if rays is None or cur_target_height != target_height or cur_target_width != target_width \
                    or mapping is None or cur_source_height != source_height or cur_source_width != source_width:
                # rays or mapping do/does not belong to the current image -> recalculate

                target_height, target_width = cur_target_height, cur_target_width
                target_shape = (target_height, target_width)
                source_height, source_width = cur_source_height, cur_source_width
                source_shape = (source_height, source_width)
                
                if args.output_model.name.lower() == "m9":
                    rays = eval(f"get_rays_m9, target_shape, fov_rad_out_x, fov_rad_out_y, args.output_calib_json)")
                else:
                    rays = eval(f"get_rays_{args.output_model.name.lower()}(target_shape, fov_rad_out_x, fov_rad_out_y)")

                
                if args.input_model.name.lower() == "m9":
                    mapping = eval(f"rays_to_m9(rays, source_shape, fov_rad_in_x, fov_rad_in_y, args.input_calib_json)")
                else:
                    mapping = eval(f"rays_to_{args.input_model.name.lower()}(rays, source_shape, fov_rad_in_x, fov_rad_in_y)")
                # mapping = rays_to_equidist(rays, source_shape, fov_rad_in_x, fov_rad_in_y)
                #mapping = mapping.astype(np.float32)
                mapping = torch.from_numpy(mapping)
                mapping = mapping[None] # H, W, 2 -> N, H, W, 2
                # grid_sample's parameter align_corner=True is expensive
                # new coordinate system for the usage with align_corner=False:
                # for x dim: coordinate -1 indicates left  BORDER of the very left  pixels
                #            coordinate +1 indicates right BORDER of the very right pixels
                # for y dim: coordinate -1 indicates upper BORDER of the top    pixels
                #            coordinate +1 indicates lower BORDER of the bottom pixel
    
                # before:
                # 
                # LUT in x dir:   | 0 | 1 | 2 |
                                 
                #                 | |   |   | |
                #                 V |   V   | V
                #                -1 |   0   | +1
                #                   V       V
                #                 -2/3     2/3
    
                # after:
                # LUT in x dir:  |-2/3| 0 |+2/3|
    
                # formular to convert to new coord system
                # x_new = -1 + 1/w + x_old * 2/w
                # y_new = -1 + 1/h + y_old * 2/h
    
                mapping[:, :, :, 0] = -1 + (1 / w) + mapping[:, :, :, 0] * (2 / w)
                mapping[:, :, :, 1] = -1 + (1 / h) + mapping[:, :, :, 1] * (2 / h)

                if args.mask_out is not None:
                    mapping_x = mapping[:, :, :, 0]
                    mapping_y = mapping[:, :, :, 1]
                    valid_mapping  = (mapping_x >= -1) * (mapping_x <= +1)
                    valid_mapping *= (mapping_y >= -1) * (mapping_y <= +1)

                # print(f"{rays[3, 1010, 0]=}")
                # print(f"{rays[3, 1010, 1]=}")
                # print(f"{rays[3, 1010, 2]=}")
                # print(f"{mapping[0, 3, 1010, 0]=}")
                # print(f"{mapping[0, 3, 1010, 1]=}")

                if args.debug_plot:
                    plot_data(rays[:, :, 0], "light ray's x coordinate")
                    plot_data(rays[:, :, 1], "light ray's y coordinate")
                    plot_data(rays[:, :, 2], "light ray's z coordinate")
                #theta = np.arccos(rays[:, :, 2])
                #theta_deg = theta * 180 / np.pi
                #theta_deg[theta >= 90] = 0
                #plot_data(theta_deg, "theta in degrees", cmap="gist_ncar")
    
            nan = float('nan')
            borderValue = (0, 0, 0) if data.dtype in [np.uint8, np.uint16] else (nan, nan, nan)

            # OpenCVs remap relies on float32 luts but the precision us insufficient. Torch can deal with 64 bit luts.
            #out = cv2.remap(data, mapping_x, mapping_y, interpolation=cv2.INTER_LINEAR, borderValue=borderValue)

            out = torch.nn.functional.grid_sample(data, mapping, align_corners=False, padding_mode='zeros')
            out = out[0] # N, C, H, W -> C, H, W

            if args.mask_out is not None:
                out[~valid_mapping] = args.mask_out

            out = torch.permute(out, (1, 2, 0)) # C, H, W -> H, W, C

            if data_dtype == torch.uint8:
                out = torch.round(out)
                out = out.to(data_dtype)
            out = out.cpu().detach().numpy()

            if args.dryrun:
                logger.debug(f"write file {output_path} ... SKIPPED (dryrun)")
            else:
                logger.debug(f"write file {output_path}")
                write_data(output_path, out)
            
            cur_id = id_counter.getAndInc()
            pbar.n = min(len(source_files), cur_id+1)
            pbar.refresh()
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Projection Model Converter')
    parser.add_argument('--dryrun', '-d', action="store_true", help="does not save files nor create directories")
    parser.add_argument('--fov_in', '-f', type=float, default=None, help="field of view (x and y direction) in degrees of input map or image [def: 180]", required=False)
    parser.add_argument('--fov_in_x', type=float, default=None, help="field of view (x direction) in degrees of input map or image [def: 180]", required=False)
    parser.add_argument('--fov_in_y', type=float, default=None, help="field of view (y direction) in degrees of input map or image [def: 180]", required=False)
    parser.add_argument('--fov_out', type=float, default=None, help="field of view (x and y direction) in degrees of output map or image [def: 180]", required=False)
    parser.add_argument('--fov_out_x', type=float, default=None, help="field of view (x direction) in degrees of output map or image [def: 180]", required=False)
    parser.add_argument('--fov_out_y', type=float, default=None, help="field of view (y direction) in degrees of output map or image [def: 180]", required=False)
    parser.add_argument('--input', '-i', metavar="file-or-dir", type=str, nargs="+", help="input maps", required=True)
    parser.add_argument('--input_scale', default=1.0, type=float, help="scales input map values")
    parser.add_argument('--input_calib', type=str, help="calibration file of input camera [json] (only for M9 model)")
    parser.add_argument('--input_model', '-a', type=str, choices=[x.name for x in ProjModel], \
            help="projection model of input file", required=True)
    parser.add_argument('--num_procs', '-n', default=8, type=int, help="number of worker processes")
    parser.add_argument('--log_level', '-l', type=str, default="warning", \
            choices=['critical', 'error', 'warning', 'info', 'debug'], help="log level of converter")
    parser.add_argument('--output', '-o', metavar="file-or-dir", type=str, nargs="+", help="output maps", required=True)
    parser.add_argument('--output_calib', type=str, help="calibration file of output camera [json] (only for M9 model)")
    parser.add_argument('--output_model', '-b', type=str, choices=[x.name for x in ProjModel], \
            help="projection model of output file", required=True)
    parser.add_argument('--output_width', type=int, help="output width (default: same as input width)")
    parser.add_argument('--output_height', type=int, help="output width (default: same as input width)")
    parser.add_argument('--mask_out', type=float, default=None, help="replace pixels in output map by MASK_OUT if there's no compiant in input map [def: None]")
    parser.add_argument('--debug_plot', action='store_true', help="plot some debug maps")
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = logging.getLogger(PROJ_CONV_LOGGER_NAME)
    logger.setLevel(log_level)

    args.input_model = getattr(ProjModel, args.input_model)
    args.output_model = getattr(ProjModel, args.output_model)

    if args.input_model == args.output_model:
        logger.error(f"Input model is output model")
        exit(1)

    if args.output_width is not None and args.output_width <= 0:
        logger.error("output width must be positive or None")
        exit(1)
    
    if args.output_height is not None and args.output_height <= 0:
        logger.error("output width must be positive or None")
        exit(1)

    # postprocess input FOV:
    if args.fov_in is not None:
        if args.fov_in_x is not None:
            logger.error("Please provide --fov_in xor --fov_in_x or none of them but not both.")
            exit(1)
        else:
            args.fov_in_x = args.fov_in
        if args.fov_in_y is not None:
            logger.error("Please provide --fov_in xor --fov_in_y or none of them but not both.")
            exit(1)
        else:
            args.fov_in_y = args.fov_in
    else:
        # args.fov_in is None
        if args.fov_in_x is None:
            args.fov_in_x = 180.0
        if args.fov_in_y is None:
            args.fov_in_y = 180.0

    assert args.fov_in_x is not None
    assert args.fov_in_y is not None

    # postprocess output FOV:
    if args.fov_out is not None:
        if args.fov_out_x is not None:
            logger.error("Please provide --fov_out xor --fov_out_x or none of them but not both.")
            exit(1)
        else:
            args.fov_out_x = args.fov_out
        if args.fov_out_y is not None:
            logger.error("Please provide --fov_out xor --fov_out_y or none of them but not both")
            exit(1)
        else:
            args.fov_out_y = args.fov_out
    else:
        # args.fov_out is None
        if args.fov_out_x is None:
            args.fov_out_x = 180.0
        if args.fov_out_y is None:
            args.fov_out_y = 180.0

    assert args.fov_out_x is not None
    assert args.fov_out_y is not None

    if args.input_calib is not None:
        with open(args.input_calib) as calib:
            args.input_calib_json = json.loads(calib.read())
    else:
        args.input_calib_json = None
    
    if args.output_calib is not None:
        with open(args.output_calib) as calib:
            args.output_calib_json = json.loads(calib.read())
    else:
        args.output_calib_json = None
    
    procs = []
    file_id_counter = AtomicIntegerProc(0)

    all_source_files = []
    all_target_files = []

    for in_entry, out_entry in zip(args.input, args.output):
        if not os.path.exists(in_entry):
            logger.error(f"The input file '{in_entry}' does not exist.")
            exit(1)
            
        if os.path.isfile(in_entry):
            if os.path.isdir(out_entry):
                logger.error(f"The output path '{out_entry}' for input file '{in_entry}' is a directory. \
                        It should be a file path.")
                exit(1)

            all_source_files.append(in_entry)
            all_target_files.append(out_entry)
            
            parent = os.path.dirname(out_entry)
            if len(parent) > 0 and not os.path.isdir(parent):
                if args.dryrun:
                    logger.debug(f"create directory {parent} ... SKIPPED (dryrun)")
                else:
                    logger.debug(f"create directory {parent}")
                    os.makedirs(parent, exist_ok=True)

        elif os.path.isdir(in_entry):
            if os.path.isfile(out_entry):
                logger.error(f"The output path '{out_entry}' for input directory '{in_entry}' is a file. \
                        It should be a directory path.")
                exit(1)
   
            source_files, target_files = get_in_and_out_files(in_entry, out_entry)
            
            all_source_files += source_files
            all_target_files += target_files

            if not os.path.isdir(out_entry):
                if args.dryrun:
                    logger.debug(f"create directory {out_entry} ... SKIPPED (dryrun)")
                else:
                    logger.debug(f"create directory {out_entry}")
                    os.makedirs(out_entry, exist_ok=True)
   
    if args.num_procs == 1:
        run(file_id_counter, all_source_files, all_target_files, 0, args)
    else:
        for proc_id in range(args.num_procs):
            my_proc = Process(target=run, args=(file_id_counter, all_source_files, all_target_files, proc_id, args))
            procs.append(my_proc)
            my_proc.start()
        
        for proc_id in range(args.num_procs):
            procs[proc_id].join()
