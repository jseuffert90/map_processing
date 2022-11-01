#!/usr/bin/env python3

import argparse
import os
import time
import sys

import cv2
import numpy as np


class Refine():
    def __init__(self, img, pos_x, pos_y, name):
        self.scale = 9 # should be odd to avoid loosing info while casting pos after scaling
        self.pos_x = int(self.scale * (pos_x + 0.5) - 0.5)
        self.pos_y = int(self.scale * (pos_y + 0.5) - 0.5)
        self.img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        self.zoomed_snippet = None
        self.name = name
        self.new_pos_x = self.pos_x
        self.new_pos_y = self.pos_y
        
        self.snippet_w = int(40 * self.scale + 1)
        self.snippet_h = self.snippet_w

        self.cut_snippet()
        cv2.createTrackbar('horiz_refine', name, int((self.snippet_w - 1)/2), self.snippet_w - 1, self.on_horiz_refine_change)
        cv2.createTrackbar('vert_refine',  name, int((self.snippet_h - 1)/2), self.snippet_h - 1, self.on_vert_refine_change)

    
    def on_horiz_refine_change(self, value):
        assert self.snippet_w % 2 == 1
        offset = value - int((self.snippet_w - 1)/2)
        self.new_pos_x = self.pos_x + offset 
        self.cut_snippet()
    
    def on_vert_refine_change(self, value):
        assert self.snippet_w % 2 == 1
        offset = value - int((self.snippet_h - 1)/2)
        self.new_pos_y = self.pos_y + offset
        self.cut_snippet()

    def get_refined_coords(self):
        refined_x = ((self.new_pos_x + 0.5) / self.scale) - 0.5
        refined_y = ((self.new_pos_y + 0.5) / self.scale) - 0.5
        return refined_x, refined_y

    def cut_snippet(self):
        assert self.snippet_w % 2 == 1
        self.snippet_l = self.new_pos_x - int((self.snippet_w - 1) / 2)
        self.snippet_r = self.new_pos_x + int((self.snippet_w - 1) / 2)
        self.snippet_t = self.new_pos_y - int((self.snippet_h - 1) / 2)
        self.snippet_b = self.new_pos_y + int((self.snippet_h - 1) / 2)

        self.zoomed_snippet = self.img[self.snippet_t:self.snippet_b+1, self.snippet_l:self.snippet_r+1]
        assert self.zoomed_snippet.shape[:2] == (self.snippet_h, self.snippet_w)
        self.plot_snippet()
    
    def plot_snippet(self):
        cpy = np.copy(self.zoomed_snippet)
        cross_x = round((self.snippet_w - 1) / 2)
        cross_y = round((self.snippet_h - 1) / 2)
        cpy[:, cross_x, :] = 0
        cpy[cross_y, :, :] = 0
        cv2.imshow(self.name, cpy)


## Converts a pixel location to a light ray and determins angle between ray and -x axis
#
#  Camera model: equiangular
#
#  @param y pixel's y coordinate
#  @param x pixel's x coordinate
#  @param focal focal length
#  @param c_y distortion center's y coordinate
#  @param c_x distortion center's x coordinate
#  @return light ray and the angle between ray and -x axis
def pixel_to_beta(y, x, focal, c_y, c_x):
    y_aligned = y - c_y
    x_aligned = x - c_x

    rho = np.sqrt(x_aligned ** 2 + y_aligned ** 2)
    theta = rho / focal # elevation
    phi = np.arctan2(y_aligned, x_aligned) % (2 * np.pi) # azimuth

    sin_phi   = np.sin(phi)
    cos_phi   = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    ray = np.array([
        [cos_phi * sin_theta],
        [sin_phi * sin_theta],
        [cos_theta]])

    cos_beta = -cos_phi * sin_theta
    if cos_beta > 1:
        cos_beta = 1
    if cos_beta < -1:
        cos_beta = -1
    beta = np.arccos(cos_beta)
    return ray, beta

## Determines Euc. distance of left camera and world point
#
#  Camera model: equiangular
#  X axis of both cameras assumed to be colinear
#
#  @param beta_l angle between -x axis and light ray falling into left camera
#  @param beta_r angle between -x axis and light ray falling into right camera
#  @param baseline distance between the cameras
#  @return Euc. distance between left camera and world point
def get_euc_dist(beta_l, beta_r, baseline):
    return baseline * np.sin(beta_r) / np.sin(beta_l - beta_r)

def process_stereo_pair():
    global focal_l
    global focal_r
    global c_x_l
    global c_y_l
    global c_x_r
    global c_y_r
    global baseline
    global x_l, y_l
    global x_r, y_r

    global img_l
    global img_l

    if x_l is None:
        return
    if y_l is None:
        return
    if x_r is None:
        return
    if y_r is None:
        return

    print("Press any key on keyboard to take refinement.")
    left_refine  = Refine(img_l, x_l, y_l, "refine_left")
    right_refine = Refine(img_r, x_r, y_r, "refine_right")
    cv2.waitKey(0)
    cv2.destroyWindow("refine_left")
    cv2.destroyWindow("refine_right")
    
    refined_x_l, refined_y_l = left_refine.get_refined_coords()
    refined_x_r, refined_y_r = right_refine.get_refined_coords()

    print(f"refined_x_l: {refined_x_l}; xl: {x_l}")
    print(f"refined_y_l: {refined_y_l}; yl: {y_l}")
    print(f"refined_x_r: {refined_x_r}; xl: {x_r}")
    print(f"refined_y_r: {refined_y_r}; yl: {y_r}")

    x_l = refined_x_l
    y_l = refined_y_l
    x_r = refined_x_r
    y_r = refined_y_r

    ray_l, beta_l = pixel_to_beta(y_l, x_l, focal_l, c_y_l, c_x_l)
    ray_r, beta_r = pixel_to_beta(y_r, x_r, focal_r, c_y_r, c_x_r)

    disparity = beta_l - beta_r
    euc_dist = get_euc_dist(beta_l, beta_r, baseline)
    world_point = ray_l * euc_dist
    depth = world_point[2, 0]
    x_cam, y_cam, z_cam = world_point.ravel().tolist()

    print("##################################################")
    print(f"left image [x, y]: [{x_l:0.1f}, {y_l:0.1f}]")
    print(f"right image [x, y]: [{x_r:0.1f}, {y_r:0.1f}]")
    print(f"disparity: {disparity}")
    print(f"Euclidean distance: {euc_dist}")
    print(f"z depth: {depth}")
    print(f"point in left cam coordinate system: {x_cam, y_cam, z_cam}")
    

def click_img_left(event, x, y, flags, params):
    click_img(True, event, x, y)

def click_img_right(event, x, y, flags, params):
    click_img(False, event, x, y)

def click_img(is_left, event, x, y):
    global x_l
    global y_l
    global x_r
    global y_r
    global img_l
    global img_r
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if x_l is not None and x_r is not None:
            x_l, y_l, x_r, y_r = None, None, None, None
            cv2.imshow("left image", img_l)
            cv2.imshow("right image", img_r)
            # print("click", name)
        if is_left:
            x_l = x
            y_l = y
            img_cpy = img_l.copy()
            cv2.circle(img_cpy, (x, y), 10, (255, 0, 0), 2)
            cv2.imshow("left image", img_cpy)
        else:
            x_r = x
            y_r = y
            img_cpy = img_r.copy()
            cv2.circle(img_cpy, (x, y), 10, (255, 0, 0), 2)
            cv2.imshow("right image", img_cpy)
        process_stereo_pair()

if __name__ == "__main__":

    global focal_l
    global focal_r
    global c_x_l
    global c_y_l
    global c_x_r
    global c_y_r
    global baseline
    global x_l, y_l
    global x_r, y_r
    global img_l
    global img_r
    

    parser = argparse.ArgumentParser(description='Determine depth from point correspondence')
    parser.add_argument('--baseline', type=float, required=True, \
            help="baseline of stereo camera pair (same unit as depth)")
    parser.add_argument('--fov', '-f', type=float, help="field of view in degrees", default=180.0)
    parser.add_argument('--left', '-l', metavar="image", type=str, help="left camera's image", required=True)
    parser.add_argument('--right', '-r', metavar="image", type=str, help="right camera's image", required=True)

    args = parser.parse_args()

    pid = os.getpid()
    print(f"Process ID: {pid}")

    x_l = None
    y_l = None
    x_r = None
    y_r = None

    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)

    if img_l is None:
        print(f"ERROR: Could not read {args.left}", file=sys.stderr)
        exit(1)
    if img_r is None:
        print(f"ERROR: Could not read {args.right}", file=sys.stderr)
        exit(1)
    
    height_l, width_l = img_l.shape[:2]
    height_r, width_r = img_r.shape[:2]

    focal_l = height_l / (args.fov / 180.0 * np.pi)
    focal_r = height_r / (args.fov / 180.0 * np.pi)
    c_x_l = (width_l - 1) / 2.0
    c_y_l = (height_l - 1) / 2.0
    c_x_r = (width_r - 1) / 2.0
    c_y_r = (height_r - 1) / 2.0
    baseline = args.baseline

    cv2.namedWindow("left image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("right image", cv2.WINDOW_NORMAL)
    cv2.imshow("left image", img_l)
    cv2.setMouseCallback('left image', click_img_left)
    
    cv2.imshow("right image", img_r)
    cv2.setMouseCallback('right image', click_img_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
