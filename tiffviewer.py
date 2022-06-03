#!/usr/bin/env python3

import argparse
import logging
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Displays multiple float-channel tiff files')
    parser.add_argument('file', type=str, nargs="+", help="input files")
    parser.add_argument('--loglevel', '-l', \
            choices=['critical', 'error', 'warning', 'info', 'debug'], \
            default="warning", type=str, help="set the log level")

    args = parser.parse_args()
    
    log_level = getattr(logging, args.loglevel.upper())
    logger = logging.getLogger('tiffviewer')
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)
    
    images = []
    for f in args.file:
        image = tifffile.imread(f)
        image = image.squeeze()
        if len(image.shape) != 2:
            logger.error("only tiffs with a single channels and a signle page are supported")
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
    min_val = np.min(debug_tensor_valid)
    max_val = np.max(debug_tensor_valid)

    logger.debug(f"{min_val=}")
    logger.debug(f"{max_val=}")
    logger.debug(f"{n_rows=}")
    logger.debug(f"{n_cols=}")

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    n_fields = n_rows * n_cols

    if n_fields > 1:
        axes_ravel = axes.ravel().tolist()
    else:
        axes_ravel = [axes]

    for i, axis in enumerate(axes_ravel):
        if i < num_images:
            handle = axis.imshow(images[i], vmin=min_val, vmax=max_val)
            t = os.path.basename(args.file[i])
            axis.set_title(t)
        else:
            fig.delaxes(axis)
   
    fig.colorbar(handle, ax=axes_ravel)

    plt.show()

