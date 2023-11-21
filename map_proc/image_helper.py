from .pointcloud import PointCloud

import logging
import os

import Imath
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
from PIL import Image
import tifffile

IMAGE_HELPER_LOGGER_NAME = "Image Helper"

def import_exr_grayscale(file_path: str):
    # inspired by: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)
    
    exr_file = OpenEXR.InputFile(file_path)
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    for ch_key in ['R', 'Y', 'Z', 'H']:
        channel = None
        try:
            channel = exr_file.channel(ch_key, PIXEL_TYPE)
            break
        except TypeError:
            pass
    
    if channel == None:
        channel_keys = exr_file.header()['channels'].keys()
        if len(channel_keys) != 1:
            logging.getLogger(IMAGE_HELPER_LOGGER_NAME).error("Unsupported EXR format")
            raise Error("Unsupported EXR format")
        channel = exr_file.channel(list(channel_keys)[0], PIXEL_TYPE)

    map_np = np.frombuffer(channel, dtype=np.float32)
    map_np.shape = (height, width)
    map_np = map_np.copy()
    return map_np

def export_exr_grayscale(map_np: np.ndarray, file_path: str):
    # inspired by: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)

    map_np = map_np.squeeze()  
    map_np = map_np.astype(np.float32)
    if len(map_np.shape) == 3:
        # take green channel as "gray channel"
        map_np = map_np[:, :, 1]

    pixels = map_np.tobytes()
    header = OpenEXR.Header(map_np.shape[1], map_np.shape[0])
    channel = Imath.Channel(PIXEL_TYPE)
    header['channels'] = {"Y": channel}
    exr = OpenEXR.OutputFile(str(file_path), header)
    exr.writePixels({'Y': pixels})
    exr.close()

def read_data(path: str):
    logger = logging.getLogger(IMAGE_HELPER_LOGGER_NAME)
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".tif", ".tiff"]:
        data = tifffile.imread(path)
        data = data.squeeze()
        if len(data.shape) == 3:
            logger = logging.getLogger(IMAGE_HELPER_LOGGER_NAME)
            logger.warning("Green channel taken from RGB map")
            data = data[1, :, :]
    elif ext == ".exr":
        data = import_exr_grayscale(path)
    else:
        data = read_image(path)
    return data

def write_data(path: str, data, colormap=None):
    if colormap is not None:
        assert colormap.shape[2] == 1 or colormap.shape[2] == 3

    ext = os.path.splitext(path)[-1].lower()
    if ext in [".tif", ".tiff"]:
        tifffile.imwrite(path, data, compression="ZSTD")
    elif ext == ".exr":
        export_exr_grayscale(data, path)
    elif ext == ".ply":
        data_Nx3= data.ravel().reshape(-1, 3)
        nan_locs_Nx3 = data_Nx3 != data_Nx3
        nan_locs_Nx3 = np.logical_or(np.logical_or(nan_locs_Nx3[:, 0], nan_locs_Nx3[:, 1]), nan_locs_Nx3[:, 2])
        nan_locs_Nx3 = nan_locs_Nx3[:, None]
        nan_locs_Nx3 = np.tile(nan_locs_Nx3, (1, 3))
        data_Nx3 = data_Nx3[np.logical_not(nan_locs_Nx3)].reshape(-1, 3)

        if colormap is None:
            cloud = PointCloud(data_Nx3)
        else:
            n_channels = colormap.shape[2] 
            colors_Nx3 = colormap.ravel().reshape(-1, n_channels)
            if n_channels == 1:
                colors_Nx3 = np.tile(colors_Nx3, (1, 3))
            assert colors_Nx3.shape[1] == 3
            colors_Nx3 = colors_Nx3[np.logical_not(nan_locs_Nx3)].reshape(-1, 3)
            colors_Nx3[colors_Nx3 != colors_Nx3] = 0
            cloud = PointCloud(data_Nx3, colors_Nx3)

        cloud.save(path)
    else:
        write_image(path, data)

def read_image(path: str):
    return np.array(Image.open(path)) 

def write_image(path: str, img_np):
    Image.fromarray(img_np).save(path)

def plot_data(data, name, vmin=None, vmax=None, cmap='jet'):
    plt.figure()
    if name is not None:
        plt.title(name)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.show()

def is_supported_data_file(path: str):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[-1].lower()
        if ext in [".tif", ".tiff", ".exr"]:
            return True
    return False

def is_supported_colormap_file(path: str):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[-1].lower()
        if ext in [".tif", ".tiff", ".png", ".jpeg", ".jpg", ".webp"]:
            return True
    return False

def is_supported_input_file(path: str):
    if is_supported_data_file(path):
        return True
    return is_supported_colormap_file(path)
