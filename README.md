# MAP PROCESSING

This repository constitutes a set of Python scripts to process or visualize
depth, distance and disparity maps as well as point clouds.
The following camera projection models are supported:
equidistant, equipolar equidistant, (partially) perspective

Hereinafter, scripts providing a main function are introduced:
- `det_depth_from_point_corresp.py`:  
  This script opens a GUI and allows to manually select point correspondences to manually retrieve the depth of a pixel.
- `map_converter.py`:
  The map converter allows to convert maps and point clouds. Provided data types are:
  - depth maps (z distance of a point to the camera)
  - distance maps (Euclidean distance of a point to the camera)
  - disparity maps (maps containing disparity values defined in [1])
  - point clouds (ply format)
- `proj_model_converter.py`:
  This script converts maps to other maps underlying a different projection model (equidistant & equipolar equidistant).
- `scale_dataset.py`:
  This script down- up upscales the maps or images of a given dataset. `torchvision.transforms.Resize` is used for that purpose as this method strictly preserves the FOV of the images/maps.
- `tiffviewer.py`:
  The tiffviewer is reminiscent of the program `tifffile` of the pip package `tifffile`.
  However, this viewer can plot multiple tiffs at once and ignores NaNs for adjusting the range of the axis.
- `viz_pc.py`:
  This script makes a video from a point cloud.

## Dependencies

- PyTorch (v1.11+)
- PIL (v9.1+)
- NumPy (v1.22+)
- OpenCV (v4.5+)
- Matplotlib (v3.5+)
- OpenEXR (python version v1.3.8+)
- tifffile (v2022.5+)
- tqdm (v4.46+)
- Open3D (only for `viz_pc.py`; v0.15.1+)

The script `create_env.sh` creates a virtual Python environment with all dependencies installed. However, Open3D is only installed if available for the current Python version. If no such package is present, the user has to build his or her own Open3D wheel package. Please confer http://www.open3d.org/docs/release/compilation.html.

## References

[1] S. Li, “Trinocular Spherical Stereo,” in *2006 IEEE/RSJ International Conference on Intelligent Robots and Systems*, Beijing, China, Oct. 2006, pp. 4786–4791. doi: [10.1109/IROS.2006.282350](https://doi.org/10.1109/IROS.2006.282350).
