#!/usr/bin/env python3

from map_proc.image_helper import *

import argparse
import logging
import math
import os
import sys

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import tifffile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Displays multiple map files')
    # positional arguments
    parser.add_argument('file', type=str, nargs="+", help="input files")

    # keyword arguments
    parser.add_argument('--cmap', '-c', type=str, default='gnuplot2', help="matplotlib colormap")
    parser.add_argument('--dpi', '-d', type=int, help="output file dpi")
    parser.add_argument('--figwidth', '-w', type=float, help="figure width in cm")
    parser.add_argument('--figheight', type=float, help="figure height in cm")
    parser.add_argument('--fontsize', '-f', type=int, default=12, help="font size")
    parser.add_argument('--loglevel', '-l', \
            choices=['critical', 'error', 'warning', 'info', 'debug'], \
            default="warning", type=str, help="set the log level")
    parser.add_argument('--names', '-n', type=str, nargs="+", help="title or name for each subfigure (default: file name)")
    parser.add_argument('--outfile', '-o', type=str, help="output file (if set, no plots are shown on the display)")
    parser.add_argument('--vmax', type=float, \
            help="maximum value to be in color range (default: max. observable value of all input files)")
    parser.add_argument('--vmin', type=float, \
            help="minimum value to be in color range (default: min. observable value of all input files)")

    args = parser.parse_args()
    
    log_level = getattr(logging, args.loglevel.upper())
    logger = logging.getLogger('mapviewer')
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)

    # cm -> inch
    figure_width = args.figwidth / 2.54 if args.figwidth is not None else None
    figure_height = args.figheight / 2.54 if args.figheight is not None else None

    if figure_width is None and figure_height is not None:
        figure_width = figure_height
    elif figure_width is not None and figure_height is None:
        figure_height = figure_width
   
    font = {'family' : 'serif',
            'serif'  : ['Computer Modern'],
            'size'   : args.fontsize}
    rc('font', **font)
    rc('text', usetex=True)

    # if args.fontsize is not None:
    #     plt.rcParams.update({'font.size': args.fontsize})

    images = []
    for f in args.file:
        _, ext = os.path.splitext(f)
        ext = ext.lower()

        if ext == ".tif" or ext == ".tiff":
            image = tifffile.imread(f)
        elif ext == ".exr":
            image = import_exr_grayscale(f)
        else:
            image = read_image(f)

        image = image.squeeze()

        if len(image.shape) != 2:
            logger.error("only maps with a single channel and a single page are supported")
            exit(1)
        images.append(image)

    num_images = len(args.file)
    if num_images < 3:
        n_rows = 1
        n_cols = num_images
    else:
        n_rows = round(math.sqrt(num_images))
        n_cols = math.ceil(num_images / n_rows)

    
    debug_tensor = np.vstack(images)
    valid_mask = (debug_tensor == debug_tensor)
    valid_mask = np.logical_and(valid_mask, np.logical_not(np.isinf(debug_tensor)))

    debug_tensor_valid = debug_tensor[valid_mask]
    min_val = args.vmin if args.vmin is not None else np.min(debug_tensor_valid)
    max_val = args.vmax if args.vmax is not None else np.max(debug_tensor_valid)

    logger.debug(f"{min_val=}")
    logger.debug(f"{max_val=}")
    logger.debug(f"{n_rows=}")
    logger.debug(f"{n_cols=}")

    if figure_width is not None and figure_height is not None:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figure_width, figure_height))
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    n_fields = n_rows * n_cols

    if n_fields > 1:
        axes_ravel = axes.ravel().tolist()
    else:
        axes_ravel = [axes]

    for i, axis in enumerate(axes_ravel):
        if i < num_images:
            handle = axis.imshow(images[i], vmin=min_val, vmax=max_val, cmap=args.cmap)
            if args.names is not None and len(args.names) > i:
                t = args.names[i]
            else:
                t = os.path.basename(args.file[i])
            axis.set_title(t)
        else:
            fig.delaxes(axis)
   
    fig.colorbar(handle, ax=axes_ravel)

    if args.outfile is None:
        plt.show()
    else:
        plt.savefig(args.outfile, dpi=(args.dpi if args.dpi is not None else 'figure'))

