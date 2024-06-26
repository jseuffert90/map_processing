#!/usr/bin/env python3

from map_proc.image_helper import *
from map_proc.numeric_helper import *
from map_proc.pointcloud import *
from map_proc.proj_models import *

import argparse
import logging
from multiprocessing import Process
import os

import glob
import json
import numpy as np
from PIL import Image
import tifffile
import tqdm

LOGGER_NAME = 'omni map converter'

def from_depth(depth_map: np.ndarray, rays, method_args):
    fov = method_args['fov']
    height, width = depth_map.shape[0:2]
    scale = depth_map.reshape((height, width, 1)) / rays[:, :, 2:3]
    return rays * scale

def from_dist(dist_map: np.ndarray, rays, method_args):
    fov = method_args['fov']
    height, width = dist_map.shape[0:2]
    return rays * dist_map.reshape((height, width, 1))

def from_disp(disp_map: np.ndarray, rays, method_args):
    logger = logging.getLogger(LOGGER_NAME)

    fov = method_args['fov']
    baseline = method_args['baseline']

    height, width = disp_map.shape[0:2]
    
    # angle between ray and -x axis
    # rays assumed to be normalized

    # cos(beta_l) = <r^T,-e_x> = -r_x

    logger.debug(f"{rays.shape=}")
    cos_angle = -rays[:, :, 0]
    logger.debug(f"{cos_angle.shape=}")
    
    # handling numerical imprecision
    cos_angle[cos_angle > 1] = 1
    cos_angle[cos_angle < -1] = -1
    beta_l = np.arccos(cos_angle)

    #      r 
    #        \
    #         \
    #   beta_l \ delta
    #  ---------+ - - - - >
    # -e_x             e_x

    logger.debug(f"{beta_l.shape=}")

    #                /\
    #               /  \
    #              / d  \
    #             /      \
    #            /        \
    #           /          \
    #      bl  /         br \
    #  ......./______________\
    #         C_l            C_r   (left and right camera)
    #
    # disparity = beta_l - beta_r
    # beta_r = beta_l - disparity

    logger.debug(f"{disp_map.shape=}")

    beta_r = beta_l - disp_map
    
    # distance / sin(beta_r) = baseline / sin(disp) (sine rule)
    
    dist_map = np.sin(beta_r) * baseline / np.sin(disp_map)
    return rays * dist_map[:, :, None]

def from_invdist(dist_map: np.ndarray, rays, method_args):
    fov = method_args['fov']
    height, width = dist_map.shape[0:2]
    return rays * (1 / dist_map.reshape((height, width, 1)))

def from_pdisp(pdisp_map: np.ndarray, rays, method_args):
    # this method returns the perspective disparity
    baseline = method_args['baseline']
    focal = method_args['focal']
    assert len(pdisp_map.shape) == 2

    z = np.zeros_like(pdisp_map) * float('nan')
    valid_pdisp = pdisp == pdisp
    valid_pdisp *= pdisp != 0
    valid_pdisp *= np.abs(pdisp) < np.inf

    z[valid_pdisp] = focal * baseline / pdisp_map[valid_pdisp]
    return rays * z / rays[:, :, 0]

def to_depth(pc: np.ndarray, method_args):
    assert pc.shape[2] == 3, "The shape of the ordered point cloud must be [height, width, 3]"
    return pc[:, :, 2]

def to_dist(pc: np.ndarray, method_args): 
    assert pc.shape[2] == 3, "The shape of the ordered point cloud must be [height, width, 3]"
    return np.linalg.norm(pc, ord=2, axis=2)

def to_disp(pc: np.ndarray, method_args):
    baseline = method_args['baseline']
    assert pc.shape[2] == 3, "The shape of the ordered point cloud must be [height, width, 3]"
    
    C_r = np.array([baseline, 0., 0.])
    C_r = C_r[None, None, :]

    rays_obj_to_Cl = -pc  # C_l - pc
    rays_obj_to_Cr = C_r - pc

    norm_l = np.linalg.norm(rays_obj_to_Cl, ord=2, axis=2)
    norm_r = np.linalg.norm(rays_obj_to_Cr, ord=2, axis=2)
    norm_lr = norm_l * norm_r

    scalar_prod = rays_obj_to_Cl * rays_obj_to_Cr
    scalar_prod = np.sum(scalar_prod, axis=2)
    cos_disp = scalar_prod / norm_lr
    
    # handling numerical imprecision
    cos_disp[cos_disp > 1] = 1
    cos_disp[cos_disp < -1] = -1
    disp_map = np.arccos(cos_disp)
    return disp_map

def to_invdist(pc: np.ndarray, method_args): 
    assert pc.shape[2] == 3, "The shape of the ordered point cloud must be [height, width, 3]"
    return 1 / to_dist(pc, method_args)

def to_pdisp(pc: np.ndarray, method_args):
    # this method returns the perspective disparity
    baseline = method_args['baseline']
    focal = method_args['focal']
    assert pc.shape[2] == 3, "The shape of the ordered point cloud must be [height, width, 3]"

    z = pc[:, :, 2]
    pdisp_map = np.zeros_like(z) * float('nan')
    valid_z = z == z
    valid_z *= z != 0
    valid_z *= np.abs(z) < np.inf
    pdisp_map[valid_z] = focal * baseline / z[valid_z]
    return pdisp_map

# dummy mehtod
def to_pc(pc: np.array, method_args):
    return pc

def read_colors(path: str):
    logger = logging.getLogger(LOGGER_NAME)
    cmap = np.asarray(Image.open(path))
    if cmap.dtype not in [np.uint8, np.uint16]:
        raise ValueError(f"only 8 bit or 16 bit unsigned int formats are supported")
    return cmap

def run(id_counter, source_files, colormaps, target_files, proc_id, args):
    fov_rad = args.fov / 180.0 * np.pi

    from_to_args = {
            'fov': fov_rad,
            'baseline': args.baseline,
            'focal': args.focal
    }
    
    logger = logging.getLogger(LOGGER_NAME)

    height, width, rays = None, None, None
    
    with tqdm.tqdm(total=len(source_files), disable=(proc_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(source_files):
            input_path = source_files[cur_id]
            output_path = target_files[cur_id]
            cmap_path = None
            colormap = None
            if colormaps is not None:
                cmap_path = colormaps[cur_id]
                colormap = read_colors(cmap_path)

            data = read_data(input_path)
            
            if args.repl_value is not None:
                rep, by = args.repl_value.split(";")
                rep = float(rep)
                by = float(by)
                data[data == rep] = by

            if args.input_scale != 1.0:
                data *= args.input_scale

            cur_height, cur_width = data.shape[:2]
            if rays is None or cur_height != height or cur_width != width:
                # rays do not match the current image -> recalculate
                height, width = cur_height, cur_width

                if args.proj_model.name.lower() == "m9":
                    rays = get_rays_m9((height, width), fov_rad, fov_rad, args.calib_json)
                else:
                    rays = eval(f"get_rays_{args.proj_model.name.lower()}((height, width), fov_rad, fov_rad)")

            if cur_width != width or cur_height != height:
                raise ValueError("All input files must have the same dimensions.")

            if from_to_args['focal'] is None:
                from_to_args['focal'] = min(height, width) / fov_rad

            pc = eval(f"from_{args.input_type}(data, rays, from_to_args)")
            out = eval(f"to_{args.output_type}(pc, from_to_args)")
            if args.output_scale != 1.0:
                out *= args.output_scale
            
            ext = os.path.splitext(input_path)[-1].lower()
            if args.input_type == "pc":
                if ext != ".ply":
                    raise ValueError(f"Point clouds must be saved in .ply format. Got {ext}.")
            else:
                if ext == ".ply":
                    raise ValueError(f"Cannot load {args.input_type} from {ext} file")

            ext = os.path.splitext(output_path)[-1].lower()
            if args.output_type == "pc":
                if ext != ".ply":
                    raise ValueError(f"Point clouds must be saved in .ply format. Got {ext}.")
            else:
                if ext == ".ply":
                    raise ValueError(f"Cannot save {args.output_type} into {ext} file")

            if args.dryrun:
                logger.debug(f"write file {output_path} ... SKIPPED (dryrun)")
            else:
                logger.debug(f"write file {output_path}")
                write_data(output_path, out, colormap=colormap)
            
            cur_id = id_counter.getAndInc()
            pbar.n = min(len(source_files), cur_id+1)
            pbar.refresh()

def get_stem(path: str):
    return os.path.splitext(path)[0]

def get_in_and_out_files(source_dir: str, target_dir: str, colormap_dir=None, target_ext=None, repl_str=None):
    basenames = [os.path.basename(f) for f in sorted(glob.glob(f"{source_dir}/*")) if is_supported_data_file(f)]
    logger.debug(f"{basenames=}")
    source_files = [os.path.join(source_dir, bname) for bname in basenames]

    if repl_str is not None:
        rep, by = repl_str.split(";")
        basenames_target = [b.replace(rep, by) for b in basenames]
    else:
        basenames_target = basenames
    
    if target_ext is None:
        target_files = [os.path.join(target_dir, bname) for bname in basenames_target]
    else:
        target_files = [os.path.join(target_dir, get_stem(bname) + target_ext) for bname in basenames_target]

    colormap_files = None

    if colormap_dir is not None:
        colormap_files = [f for f in sorted(glob.glob(f"{colormap_dir}/*")) if is_supported_colormap_file(f)]
        if len(source_files) != len(colormap_files):
            raise ValueError("There must be exaclty one colormap for each input map or no color map.")

    return source_files, target_files, colormap_files



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Omni Map Converter')
    parser.add_argument('--baseline', type=float, default=None, \
            help="baseline of stereo camera pair (same unit as depth)")
    parser.add_argument('--dryrun', '-d', action="store_true", help="does not save files nor create directories")
    parser.add_argument('--calib', type=str, help="calibration file of camera [json] (only for M9 model)")
    parser.add_argument('--colormap', '-c', metavar="file-or-dir", type=str, \
            nargs="+", \
            help="provide a color map for point cloud generation " \
            + "(ignored if output is not 'pc'. If provided, must have " \
            + "same num of args as --input)")
    parser.add_argument('--focal', type=float, help="focal length of the camera " \
            + "(if not provided: determined by --fov assuming equidistant camera)")
    parser.add_argument('--fov', '-f', type=float, default="180.0", help="field of view in degrees", required=True)
    parser.add_argument('--input', '-i', metavar="file-or-dir", type=str, nargs="+", help="input maps")
    parser.add_argument('--input_scale', default=1.0, type=float, help="scales input map values")
    parser.add_argument('--input_type', '-a', metavar="in_type", type=str, required=True, choices=["depth", "disp", "dist"], \
            help="type of each input map")
    parser.add_argument('--log_level', '-l', type=str, default="warning", \
            choices=['critical', 'error', 'warning', 'info', 'debug'], help="log level of converter")
    parser.add_argument('--num_procs', '-n', default=multiprocessing.cpu_count(), type=int, help="number of worker procs")
    parser.add_argument('--output', '-o', metavar="file-or-dir", type=str, nargs="+", help="output maps")
    parser.add_argument('--output_scale', default=1.0, type=float, help="scales output map values")
    parser.add_argument('--output_type', '-b', metavar="out_type", type=str, required=True, \
            choices=["depth", "disp", "dist", "pc", "pdisp", "invdist"], help="type of each output map")
    parser.add_argument('--proj_model', '-p', type=str, choices=[x.name for x in ProjModel], \
            default="EQUIDIST", help="projection model")
    parser.add_argument('--repl_str', '-r', type=str, help="replace str in input file name by another str " \
            + "in output file name (e.g. '_depth;_pdisp' replaces '_depth' by '_pdisp')")
    parser.add_argument('--repl_value', '-v', type=str, help="replace certain value a by value b in input map before processing (default: None / do not replace anything)" \
            + ", e.g. '1.0;nan' replaces 1.0 by NaN")
    
    parser.add_argument('--stop_after', '-s', type=int, metavar="N", help="stop after processing N files")
    parser.add_argument('--target_ext', '-t', type=str, help="target file extension, e.g. '.tiff'; " \
            + "if not provided: derived from input file extension")

    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    # stream = logging.StreamHandler()
    # stream.setLevel(log_level)
    # logger.addHandler(stream)

    args.proj_model = getattr(ProjModel, args.proj_model)

    if args.input_type == "disp" or args.output_type == "disp":
        if args.baseline is None:
            logger.error("Baseline needed if calculations are based on disparity.")
            exit(1)
    
    if args.calib is not None:
        with open(args.calib) as calib:
            args.calib_json = json.loads(calib.read())
    else:
        args.calib_json = None

    procs = []
    file_id_counter = AtomicIntegerProc(0)

    all_source_files = []
    all_colormaps = []
    all_target_files = []

    colormap_paths = args.colormap
    if args.colormap is None:
        colormap_paths = [None] * len(args.input)

    for in_entry, c_entry, out_entry in zip(args.input, colormap_paths, args.output):
        if not os.path.exists(in_entry):
            logger.error(f"The input file '{in_entry}' does not exist.")
            exit(1)
            
        if os.path.isfile(in_entry):
            if os.path.isdir(out_entry):
                logger.error(f"The output path '{out_entry}' for input file '{in_entry}' is a directory. \
                        It should be a file path.")
                exit(1)
            if c_entry is not None:
                if os.path.isdir(c_entry):
                    logger.error(f"The color map path '{c_entry}' for input file '{in_entry}' is a directory. \
                        It should be a file path.")
                    exit(1)
                if not os.path.isfile(c_entry):
                    logger.error(f"The color map path '{c_entry}' does not exist. \
                        It should be a file path.")
                    exit(1)

            all_source_files.append(in_entry)
            if args.colormap is not None:
                all_colormaps.append(c_entry)
            if args.output_type == 'pc':
                stem, _ = os.path.splitext(out_entry)
                all_target_files.append(stem + ".ply")
            else:
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
            if c_entry is not None:
                if os.path.isfile(c_entry):
                   logger.error(f"The color map path '{c_entry}' for input directory '{in_entry}' is a file. \
                        It should be a directory path.")
                   exit(1)
                if not os.path.isdir(c_entry):
                    logger.error(f"The color map path '{c_entry}' does not exist. \
                        It should be a directory path.")
                    exit(1)
   
            target_ext = args.target_ext
            if args.output_type == "pc":
                target_ext = ".ply"
            source_files, target_files, colormap_files = \
                    get_in_and_out_files(in_entry, out_entry, c_entry, target_ext=target_ext, repl_str=args.repl_str)
            
            all_source_files += source_files
            all_target_files += target_files
            if colormap_files is not None:
                all_colormaps += colormap_files

            if not os.path.isdir(out_entry):
                if args.dryrun:
                    logger.debug(f"create directory {out_entry} ... SKIPPED (dryrun)")
                else:
                    logger.debug(f"create directory {out_entry}")
                    os.makedirs(out_entry, exist_ok=True)

    if args.stop_after is not None:
        all_colormaps = all_colormaps[:args.stop_after]
        all_source_files = all_source_files[:args.stop_after]
        all_target_files = all_target_files[:args.stop_after]

    if len(all_colormaps) == 0:
        all_colormaps = None

    for proc_id in range(args.num_procs):
        my_proc = Process(target=run, args=(file_id_counter, all_source_files, all_colormaps, all_target_files, proc_id, args))
        procs.append(my_proc)
        my_proc.start()
    
    for proc_id in range(args.num_procs):
        procs[proc_id].join()
