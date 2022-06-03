#!/usr/bin/env python3

import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm

def rot_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]])

def rot_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]])

def rotate_front(front, angle):
    hom = rot_z(angle)
    front_np = np.array(front)[:, None]
    new_front_np = hom @ front_np
    return new_front_np.ravel().tolist()


def animation(vis, angles, max_idx, cur_idx, save_as_video, video_file_descriptor, output_dir, pbar):
    logger = logging.getLogger("Point Cloud Viz")
    idx = cur_idx[0] # idx mus be saved in a list in order to be able to change it
    logger.debug(f"Current frame ID is {idx}")
    if idx >= max_idx:
        vis.close()
        vis.register_animation_callback(None)
        if save_as_video:
            video_file_descriptor.release()
        exit(0)
        return False
    ctr = vis.get_view_control()
    cur_front = rotate_front(start_front, angles[idx])
    ctr.set_front(cur_front)

    image = vis.capture_screen_float_buffer(True)
    image = np.asarray(image)
    if save_as_video:
        logger.debug(f"{image.shape=}")
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_file_descriptor.write(image)
    else:
        plt.imsave(os.path.join(output_dir, f"{idx:05d}.png"), image, dpi = 1)
    cur_idx[0] += 1
    if pbar is not None:
        pbar.n = cur_idx[0]
        pbar.refresh()
    return False

def callback_factory(angles, max_idx, cur_idx: list, save_as_video: bool, video_file_descriptor=None, output_dir=None, pbar=None):
    return lambda x : animation(x, angles, max_idx, cur_idx, save_as_video, video_file_descriptor, output_dir, pbar)

def calc_start_front(elevation):
    start_front = np.array([0, 1, 0])[:, None]
    hom = rot_x(elevation)
    start_front = hom @ start_front
    return start_front.ravel().tolist()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Point Cloud Viz")
    parser.add_argument("--cam_elevation", "-e", type=float, default=30.0, help="camera's elevation in degrees")
    parser.add_argument("--height", type=int, default=720, \
            help="output image or video height")
    parser.add_argument("--input", "-i", metavar="pc file", required=True, type=str, help="path to point cloud")
    parser.add_argument("--log_level", "-l", \
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING', help="logging level")
    parser.add_argument("--num_frames", "-n", type=int, default=140, \
            help="number of frames to be captured from the point cloud")
    parser.add_argument("--output", "-o", metavar="dir or file", required=True, type=str, \
            help="if output directory is given, " + \
            "all rendered images are stored into this directory." + \
            "Otherwise the output parameter denotes the path to an mp4 output video")
    parser.add_argument("--up", "-u", default="z", choices=['x', 'y', 'z', 'nx', 'ny', 'nz'], type=str, \
            help="camera's up vector (n* meanse negative direction)")
    parser.add_argument("--video_fr", default=10.0, type=float, help="output video frame rate")
    parser.add_argument("--width", "-w", type=int, \
            help="output image or video width (default: derived from height with aspect ratio 16:9)")
    parser.add_argument("--with_axes", "-a", action="store_true", help="plots x-, y- and z-axis")
    parser.add_argument("--zoom", "-z", default=0.3, type=float, help="zoom factor of camera (lower is nearer)")
    args = parser.parse_args()

    if args.up == "x":
        up_vector = [1, 0, 0]
    elif args.up == "y":
        up_vector = [0, 1, 0]
    elif args.up == "z":
        up_vector = [0, 0, 1]
    elif args.up == "nx":
        up_vector = [-1, 0, 0]
    elif args.up == "ny":
        up_vector = [0, -1, 0]
    elif args.up == "nz":
        up_vector = [0, 0, -1]
    
    logger = logging.getLogger("Point Cloud Viz")
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    if args.width is None:
        args.width = args.height * 16 // 9

    logger.debug(f"{args.width=}")
    logger.debug(f"{args.height=}")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=args.width, height=args.height, visible=False)
    if not os.path.isfile(args.input):
        logger.error(f"The file {args.input} does not exist")
        exit(1)
    pcd = o3d.io.read_point_cloud(args.input)
    vis.add_geometry(pcd)
    if args.with_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5)
        vis.add_geometry(axes)
    
    ctrl = vis.get_view_control()

    elevation = args.cam_elevation * np.pi / 180
    start_front = calc_start_front(elevation)
    ctrl.set_front(start_front)
    ctrl.set_lookat([0, 0, 0])
    ctrl.set_up(up_vector)
    ctrl.set_zoom(args.zoom)

    idx = 0
    cur_idx = [idx]
    max_idx = 140
    angles = np.linspace(0, 2 * np.pi, max_idx + 1)[:-1]
    angles = angles.tolist()

    if os.path.isdir(args.output) or args.output[-1] == "/" or "." not in args.output:
        logger.debug("image mode")
        save_as_video = False
        video_file_descriptor = None
        os.makedirs(args.output, exist_ok=True)
    else:
        logger.debug("video mode")
        save_as_video = True
        if os.path.splitext(args.output)[-1].lower() != ".mp4":
            logger.error("Only MP4 is supported as an output video format.")
            exit(1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_file_descriptor = cv2.VideoWriter(args.output, fourcc, args.video_fr, (args.width, args.height))    

    output_dir = None if save_as_video else args.output

    with tqdm(total=max_idx) as pbar:
        rotate = callback_factory(angles, max_idx, cur_idx, save_as_video, video_file_descriptor, output_dir, pbar)
        vis.register_animation_callback(rotate)
        logger.debug("run visualizer")
        vis.run()
        logger.debug("visualizer finished")
        vis.destroy_window()
