from image_helper import *

from enum import Enum

import numpy as np

class ProjModel(Enum):
    PERSPECTIVE = 0
    EQUIDIST = 1
    EPIPOL_EQUIDIST = 2

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

def rays_to_epipol_equidist(rays, map_shape, fov_x, fov_y):
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

    x = rays[:, :, 0]
    y = rays[:, :, 1]
    z = rays[:, :, 2]

    # plot_data(z, "z-map", vmin=-0.1, vmax=0.1)

    # elevation
    theta = np.ones((rays_height, rays_width), dtype=float) * np.pi
    theta[z != 0] = np.arctan(np.sqrt(x[z != 0]**2 + y[z != 0]**2) / z[z != 0])
    theta[z != z] = float('nan')
    # plot_data(theta, "Theta map of reays_to_equidist")
    print(f"{theta.dtype=}")

    rho = theta * f
    print(f"{rho[3, 1010]=}")
    print(f"{theta[3, 1010]=}")

    u_r = rays[:, :, :2]
    u_r /= np.linalg.norm(u_r, axis=2, keepdims=True)

    img_points = u_r * rho[:, :, None]
    img_points += np.array([[[c_x, c_y]]]) # decenter

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

