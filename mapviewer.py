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

def replace_invalid(map_or_img, rep_val):
    tmp_img = np.copy(map_or_img)
    tmp_img[tmp_img != tmp_img] = sys.float_info.min
    tmp_img[tmp_img == np.inf] = sys.float_info.min
    tmp_img[tmp_img == -np.inf] = sys.float_info.min
    return tmp_img

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
    parser.add_argument('--title', '-t', type=str, help="title or figure")
    parser.add_argument('--vmax', type=float, \
            help="maximum value to be in color range (default: max. observable value of all input files)")
    parser.add_argument('--vmin', type=float, \
            help="minimum value to be in color range (default: min. observable value of all input files)")
    parser.add_argument('--grid', type=str, \
            help="define matplotlibs grid layout in the format <#rows>x<#cols>, e.g. 2x2 or 2x or x2 for auto completion of cols or rows, respectively")
    parser.add_argument('--abs', action='store_true', \
            help='show absolute values |v| instead of real values v')
    parser.add_argument('--no_colorbar', default=False, action='store_true', \
            help='do not plot color bar')


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
        image = read_data(f)

        if args.abs:
            image = np.abs(image)
        image = image.squeeze()

        if len(image.shape) != 2:
            logger.error("only maps with a single channel and a single page are supported")
            exit(1)
        logger.debug(f"{image.shape=}")
        images.append(image)

    num_images = len(args.file)
    if args.grid:
        try:
            n_rows, n_cols = args.grid.split('x')
            if len(n_rows) == 0 and len(n_cols) == 0:
                raise Exception
            n_rows = int(n_rows) if len(n_rows) > 0 else None
            n_cols = int(n_cols) if len(n_cols) > 0 else None
            if n_rows is None:
                n_rows = math.ceil(num_images / n_cols)
            elif n_cols is None:
                n_cols = math.ceil(num_images / n_rows)
            if n_rows <= 0 or n_cols <= 0:
                raise Exception
        except:
            logger.error(f"Malformed grid argument {args.grid} which is not in format RxC (rows x cols)")
            exit(1)
    else:
        if num_images < 3:
            n_rows = 1
            n_cols = num_images
        else:
            n_rows = round(math.sqrt(num_images))
            n_cols = math.ceil(num_images / n_rows)

    min_val = args.vmin
    if min_val is None:
        logger.debug(f"{images[0].shape=}")
        min_val = np.min(images[0])
        for i in range(1, len(images)):
            min_val = min(np.min(images[i]), min_val)
    
    min_val = args.vmin
    if min_val is None:
        tmp = replace_invalid(images[0], sys.float_info.max)
        min_val = np.min(tmp)
        for i in range(1, len(images)):
            tmp = replace_invalid(images[i], sys.float_info.max)
            min_val = min(np.min(tmp), min_val)
    
    max_val = args.vmax
    if max_val is None:
        tmp = replace_invalid(images[0], sys.float_info.min)
        max_val = np.max(tmp)
        for i in range(1, len(images)):
            tmp = replace_invalid(images[i], sys.float_info.min)
            max_val = min(np.max(tmp), max_val)

    logger.debug(f"{min_val=}")
    logger.debug(f"{max_val=}")
    logger.debug(f"{n_rows=}")
    logger.debug(f"{n_cols=}")

    if figure_width is not None and figure_height is not None:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figure_width, figure_height))
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    n_fields = n_rows * n_cols

    if args.title is not None:
        fig.suptitle(args.title)

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
  
    if args.no_colorbar == False:
        fig.colorbar(handle, ax=axes_ravel)

    if args.outfile is None:
        plt.show()
    else:
        plt.savefig(args.outfile, dpi=(args.dpi if args.dpi is not None else 'figure'))

