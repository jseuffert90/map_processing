# from .image_helper import *

from enum import Enum
import sys

import numpy as np
from .image_helper import write_data

class ProjModel(Enum):
    PERSPECTIVE = 0
    EQUIDIST = 1
    EPIPOL_EQUIDIST = 2
    M9 = 3
    ERP = 4

def fov_rays_through_pixel_center(fov, side_length):
    '''Converts the FOV into solid angle of rays traversing pixel centers only
      
    In contrast to this solid angle, the FOV comprises also light rays that
    do not exactly traverse the center of a pixel.
    
    Parameters
    ----------
    fov : float
        field of view
    side_length : int
        the side length to which the field of view refers

    Returns
    -------
    float
        field of view from light rays traversing pixel centers only
    '''
    return fov / side_length * (side_length - 1)

def get_rays_equidist(shape, fov_x, fov_y):
    height, width = shape
    if height != width:
        raise NotImplementedError("aspect ratio must be 1:1 currently")
    if fov_x != fov_y:
        raise NotImplementedError("fov must be equal for both x and y dimension")

    fov_center_pixel = fov_rays_through_pixel_center(fov_x, width)
    max_theta_center_pixel = fov_center_pixel / 2

    # coordinates normalized image plane (focal length = 1)
    x_lin_space = np.linspace(-max_theta_center_pixel, max_theta_center_pixel, width)
    y_lin_space = np.linspace(-max_theta_center_pixel, max_theta_center_pixel, height)
    xx, yy = np.meshgrid(x_lin_space, y_lin_space)
    phi = np.arctan2(yy, xx) % (2 * np.pi) # azimuth
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = rho # focal length is 1; elevation

    x_unit_sphere = np.cos(phi) * np.sin(theta)
    y_unit_sphere = np.sin(phi) * np.sin(theta)
    z_unit_sphere = np.cos(theta)

    rays = stack_coords(x_unit_sphere, y_unit_sphere, z_unit_sphere)
    
    return rays

def get_rays_epipol_equidist(shape, fov_x, fov_y):
    '''Light ray calculation for an image following the epipolar equiangular projection model of (Abraham & Förstner, 2005)

    Abraham, S., & Förstner, W. (2005). Fish-eye-stereo calibration and epipolar rectification. ISPRS Journal of Photogrammetry
    and Remote Sensing, 59(5), 278–288. https://doi.org/10.1016/j.isprsjprs.2005.03.001
    
    Parameters
    ----------
    shape : tuple of int
        shape of the image that should be rectified (height, width)
    fov_x : float
        field of view x direction
    fov_y : float
        field of view y direction

    Returns
    -------
    rays : np.ndarray of float
        light rays for each pixel
    '''

    height, width = shape
    fov_center_pixel_x = fov_rays_through_pixel_center(fov_x, width)
    fov_center_pixel_y = fov_rays_through_pixel_center(fov_y, height)
    
    # (Abraham & Förstner, 2005) used angle psi for longitude and angle beta for laditude direction
    # Here: beta is negative at the image's top

    max_abr_psi  = fov_center_pixel_x / 2
    max_abr_beta = fov_center_pixel_y / 2

    x_lin_space = np.linspace(-max_abr_psi, max_abr_psi, width)
    y_lin_space = np.linspace(-max_abr_beta, max_abr_beta, height)
    abr_psi, abr_beta = np.meshgrid(x_lin_space, y_lin_space)

    x_unit_sphere = np.sin(abr_psi)
    y_unit_sphere = np.sin(abr_beta) * np.cos(abr_psi)
    z_unit_sphere = np.cos(abr_beta) * np.cos(abr_psi)

    rays = stack_coords(x_unit_sphere, y_unit_sphere, z_unit_sphere)
    
    return rays

def get_rays_erp(shape, fov_x, fov_y):
    '''Light ray calculation for an image following the epirectangular (ERP) projection model

    Parameters
    ----------
    shape : tuple of int
        shape of the image that should be rectified (height, width)
    fov_x : float
        field of view x direction
    fov_y : float
        field of view y direction

    Returns
    -------
    rays : np.ndarray of float
        light rays for each pixel
    '''

    height, width = shape
    fov_center_pixel_x = fov_rays_through_pixel_center(fov_x, width)
    fov_center_pixel_y = fov_rays_through_pixel_center(fov_y, height)
    
    max_phi   = fov_center_pixel_x / 2
    max_theta = fov_center_pixel_y / 2

    x_lin_space = np.linspace(-max_phi, max_phi, width)
    y_lin_space = np.linspace(-max_theta, max_theta, height)
    phi, theta = np.meshgrid(x_lin_space, y_lin_space)

    x_unit_sphere = np.cos(theta) * np.sin(phi)
    y_unit_sphere = np.sin(theta)
    z_unit_sphere = np.cos(theta) * np.cos(phi)

    rays = stack_coords(x_unit_sphere, y_unit_sphere, z_unit_sphere)
    
    return rays

def get_rays_m9(shape, fov_x, fov_y, calib_json):
    height, width = shape

    fx = calib_json["fx"]
    fy = calib_json["fy"]
    cx = calib_json["cx"]
    cy = calib_json["cy"]
    k0 = calib_json["k0"]
    k1 = calib_json["k1"]
    k2 = calib_json["k2"]
    k3 = calib_json["k3"]
    k4 = calib_json["k4"]

    x_img_space = np.linspace(0, width - 1, width)
    y_img_space = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x_img_space, y_img_space, indexing='xy')
    xx_norm = (xx - cx) / fx
    yy_norm = (yy - cy) / fy

    phi = np.arctan2(yy_norm, xx_norm) % (2 * np.pi) # azimuth
    rho = np.sqrt(xx_norm**2 + yy_norm**2)
    assert np.all(rho < np.inf)
    
    theta_0 = rho
    best_theta = theta_0
    epsilon = 1e-6
    num_it = 1000
    for i in range(num_it + 1):
        theta = best_theta
        theta_2 = theta*theta
        theta_3 = theta_2*theta
        theta_4 = theta_3*theta
        theta_5 = theta_4*theta
        theta_6 = theta_5*theta
        theta_7 = theta_6*theta
        theta_8 = theta_7*theta
        theta_9 = theta_8*theta

        assert np.all(theta_4 < np.inf)
        assert np.all(theta_9 < np.inf)

        assert np.all(theta_9 == theta_9)
        assert np.all(rho == rho)
        func_val = k0 * theta + k1 * theta_3 + k2 * theta_5 + k3 * theta_7 + k4 * theta_9 - rho
        reproj_error_last_it = func_val
        if np.all(reproj_error_last_it < epsilon):
            break
        if i == num_it:
            print("FATAL: Did not find the inverse of the M9 projection function.", file=sys.stderr)
            exit(1)
        
        func_1st_dev = k0 + k1 * (3* theta_2) + k2 * (5* theta_4) + k3 * (7 * theta_6) + k4 * (9 * theta_8)
        best_theta = best_theta - func_val / func_1st_dev
    
    x_unit_sphere = np.cos(phi) * np.sin(best_theta)
    y_unit_sphere = np.sin(phi) * np.sin(best_theta)
    z_unit_sphere = np.cos(best_theta)

    rays = stack_coords(x_unit_sphere, y_unit_sphere, z_unit_sphere)
    
    return rays

def get_rays_perspective(shape, fov_x, fov_y):
    '''Light ray calculation for an image following the epirectangular (ERP) projection model

    Parameters
    ----------
    shape : tuple of int
        shape of the image that should be rectified (height, width)
    fov_x : float
        field of view x direction
    fov_y : float
        field of view y direction

    Returns
    -------
    rays : np.ndarray of float
        light rays for each pixel
    '''

    height, width = shape

    focal_x = (width / 2) / np.tan(fov_x / 2)
    focal_y = (height / 2) / np.tan(fov_y / 2)

    c_x = (width - 1) / 2
    c_y = (height - 1) / 2

    x_lin_space = np.linspace(0, width - 1, width)
    y_lin_space = np.linspace(0, height - 1, height)
    x_image, y_image = np.meshgrid(x_lin_space, y_lin_space, indexing='xy')
    
    x_norm = (x_image - c_x) / focal_x
    y_norm = (y_image - c_y) / focal_y

    r = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan(r)
    phi = np.arctan2(y_norm, x_norm)

    x_unit_sphere = np.cos(phi) * np.sin(theta)
    y_unit_sphere = np.sin(phi) * np.sin(theta)
    z_unit_sphere = np.cos(theta)

    rays = stack_coords(x_unit_sphere, y_unit_sphere, z_unit_sphere)
    
    return rays

def rays_to_epipol_equidist(rays, map_shape, fov_x, fov_y):
    raise NotImplementedError("function has not been implemented yet")

def rays_to_erp(rays, map_shape, fov_x, fov_y):
    raise NotImplementedError("function has not been implemented yet")

def rays_to_equidist(rays, map_shape, fov_x, fov_y):
    if fov_x != fov_y:
        raise NotImplementedError("fov must be equal for both x and y dimension")
    map_height, map_width, c_x, c_y, f = get_intr_equidist(map_shape, fov_x)
    if map_height != map_width:
        raise NotImplementedError("aspect ratio must be 1:1 currently")

    assert len(rays.shape) == 3
    assert rays.shape[2] == 3 # x, y, z
    rays_height, rays_width, _ = rays.shape

    fov_center_pixel = fov_rays_through_pixel_center(fov_x, map_width)

    #x = rays[:, :, 0]
    #y = rays[:, :, 1]
    z = rays[:, :, 2]

    # plot_data(z, "z-map", vmin=-0.1, vmax=0.1)

    # elevation
    theta = np.arccos(z / np.linalg.norm(rays, axis=2))
    theta[z != z] = float('nan')
    # plot_data(theta, "Theta map of reays_to_equidist")
    # print(f"{theta.dtype=}")

    rho = theta * f
    # print(f"{rho[3, 1010]=}")
    # print(f"{theta[3, 1010]=}")

    u_r = rays[:, :, :2]
    u_r /= np.linalg.norm(u_r, axis=2, keepdims=True)

    img_points = u_r * rho[:, :, None]
    img_points += np.array([[[c_x, c_y]]]) # decenter

    return img_points

def rays_to_m9(rays, map_shape, fov_x, fov_y, calib_json):
    if fov_x != fov_y:
        raise NotImplementedError("fov must be equal for both x and y dimension")
    map_height, map_width = map_shape

    assert len(rays.shape) == 3
    assert rays.shape[2] == 3 # x, y, z
    rays_height, rays_width, _ = rays.shape

    #x = rays[:, :, 0]
    #y = rays[:, :, 1]
    z = rays[:, :, 2]

    # elevation
    theta = np.arccos(z / np.linalg.norm(rays, axis=2))
    theta[z != z] = float('nan')
    # plot_data(theta, "Theta map of reays_to_equidist")
    # print(f"{theta.dtype=}")

    fx = calib_json["fx"]
    fy = calib_json["fy"]
    cx = calib_json["cx"]
    cy = calib_json["cy"]
    k0 = calib_json["k0"]
    k1 = calib_json["k1"]
    k2 = calib_json["k2"]
    k3 = calib_json["k3"]
    k4 = calib_json["k4"]

    theta_2 = theta * theta
    theta_3 = theta * theta_2
    theta_5 = theta_3 * theta_2
    theta_7 = theta_5 * theta_2
    theta_9 = theta_7 * theta_2
    
    rho_norm = k0 * theta + k1 * theta_3 + k2 * theta_5 + k3 * theta_7 + k4 * theta_9
    write_data("/tmp/rho_normed.tiff", rho_norm.squeeze())
    write_data("/tmp/theta_normed.tiff", theta.squeeze())

    u_r = rays[:, :, :2]
    u_r /= np.linalg.norm(u_r, axis=2, keepdims=True)

    img_points_norm= u_r * rho_norm[:, :, None]
    img_points_x = fx * img_points_norm[:, :, 0:1] + cx # decenter
    img_points_y = fy * img_points_norm[:, :, 1:2] + cy # decenter
    img_points = np.concatenate((img_points_x, img_points_y), axis=2)

    return img_points

def get_intr_equidist(map_shape: tuple, fov: float):
    height, width = map_shape
    c_x = (width - 1) / 2
    c_y = (height - 1) / 2
    d = min(height, width)
    f = d / fov
    return height, width, c_x, c_y, f

def stack_coords(xx, yy, zz):
    return np.concatenate([xx[:, :, None], yy[:, :, None], zz[:, :, None]], axis=2)

