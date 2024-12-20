# MAP PROCESSING

This repository constitutes a set of Python scripts to process or visualize
depth, distance and disparity maps as well as point clouds.
The following camera projection models are supported:
equidistant, equipolar equidistant, (partially) perspective

Hereinafter, scripts providing a main function are introduced:
- `calc_error_metrics.py`:  
  This script calculates MAE and RMSE of multiple error maps for regions in which all maps are valid.
- `calc_l1_error_maps.py`: Calculates error maps by taking the absolute value of the subtraction of a map and it's ground truth.
- `det_depth_from_point_corresp.py`:  
  This script opens a GUI and allows to manually select point correspondences to manually retrieve the depth of a pixel.
- `map_converter.py`:
  The map converter allows to convert maps and point clouds. Provided data types are:
  - depth maps (z distance of a point to the camera)
  - distance maps (Euclidean distance of a point to the camera)
  - inverse distance maps (inv. Euclidean distance of a point to the camera)
  - disparity maps (maps containing disparity values defined in [1])
  - point clouds (ply format)
- `proj_model_converter.py`:
  This script converts maps to other maps underlying a different projection model (equidistant & equipolar equidistant).
- `scale_dataset.py`:
  This script down- up upscales the maps or images of a given dataset. `torchvision.transforms.Resize` is used for that purpose as this method strictly preserves the FOV of the images/maps.
- `mapviewer.py`:
  The mapviewer is reminiscent of the program `tifffile` of the pip package `tifffile`.
  However, the mapviewer can plot multiple tiffs or maps at once and ignores NaNs for adjusting the range of the axis.
- `map_mask.py`:
  This script can mask out values in maps according to a given set of masks.
  It is possible to provide mutliple masks for each sample, e.g., multiple masks for multiple persons
  in a sample.
- `map_stats.py`:
  This script examines a set of input maps and collect statistics on each input map, such as:  
  Number of NaN values, number of +- inf values, number of 0s, number of real values, minimum, maximum and mean values over all real values 
- `view_stats.py`:
  This script displays histograms or binned function values. It can be used to view the results of `xy_mapper.py`.
- `viz_pc.py`:
  This script makes a video from a point cloud.
- `xy_mapper.py`:
  This script finds in a x map x values and averages their corresponding y values (input maps / same (i,j)-position as for x map).
  Example:  
    x map could be the angle between an axis and an incomming light ray.  
    y maps could be some error maps.  
    result: (Binned) graph that maps the angle onto the mean error

## Dependencies

- libilmbase (v2.3.0+)
- PyTorch (v1.11+)
- PIL (v9.1+)
- NumPy (v1.22+)
- OpenCV (v4.5+)
- Matplotlib (v3.5+)
- OpenEXR (python version v1.3.8+)
- tifffile (v2022.5+)
- tqdm (v4.46+)
- Open3D (only for `viz_pc.py`; v0.15.1+)
- a matplotlib GUI backend like: pyqt5 (v5.15+)
- system packages: cm-super, dvipng, python3-venv, texlive-latex-extra

Please install the required system packages first:
```bash
sudo apt install cm-super dvipng python3-venv texlive-latex-extra
```

The script `create_env.sh` creates a virtual Python environment with all (python)
dependencies installed. However, Open3D is only installed if available for the current Python version.
If no such package is present, the user has to build his or her own Open3D wheel package.
Please confer http://www.open3d.org/docs/release/compilation.html.

## References

[1] S. Li, “Trinocular Spherical Stereo,” in *2006 IEEE/RSJ International Conference on Intelligent Robots and Systems*, Beijing, China, Oct. 2006, pp. 4786–4791. doi: [10.1109/IROS.2006.282350](https://doi.org/10.1109/IROS.2006.282350).
