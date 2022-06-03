#!/usr/bin/env python3

from image_helper import *
from numeric_helper import *

from PIL import Image
import glob
from pathlib import Path
import threading
import tqdm
import argparse as argprs
from psutil import cpu_count
import torch
from torchvision import transforms
import numpy as np
import sys

SUPPORTED_EXT = ['.exr', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']
TO_TENSOR = transforms.ToTensor()
TO_PIL = transforms.ToPILImage()

        
    

def run(id_counter, source_files, target_files, thread_id, args, scaler):

    with tqdm.tqdm(total=len(source_files), disable=(thread_id > 0)) as pbar:
        cur_id = id_counter.getAndInc()
        while cur_id < len(source_files):
            # do stuff
            input_path = source_files[cur_id]
            output_path = target_files[cur_id]

            # INPUT
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']:
                input_img = Image.open(str(input_path))
                if input_img.n_frames > 1:
                    raise ValueError("Multi frame images are not supported.")

                input_tensor = TO_TENSOR(input_img)
            elif input_path.suffix.lower() == '.exr':
                input_tensor_np = import_exr_grayscale(str(input_path))
                input_tensor = torch.from_numpy(input_tensor_np)
            else:
                assert False # Forgot to implement a new file format?
        
            if input_tensor.dim() == 2:
                input_tensor = input_tensor[None] # [C, H, W]
            
            assert input_tensor.dim() == 3
    
            if args.map_val_to_nan is not None:
                if input_tensor.dtype == torch.float16 or \
                        input_tensor.dtype == torch.float32 or \
                        input_tensor.dtype == torch.float64:
                    input_tensor[input_tensor == args.map_val_to_nan] = float('nan')

            # RESIZING with torch as this is the default resizing operation in ANNs
            output_tensor = scaler(input_tensor)
            
            # OUTPUT
            if output_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']:
                output_tensor = output_tensor.squeeze()
                output_img = TO_PIL(output_tensor)
                output_img.save(str(output_path), lossless=(not args.lossy), quality=args.quality, method=args.method)
            elif output_path.suffix.lower() == '.exr':
                output_tensor_np = output_tensor.detach().numpy()
                export_exr_grayscale(output_tensor_np, output_path)
            else:
                assert False # Forgot to implement a new file format?

            cur_id = id_counter.getAndInc()
            pbar.n = min(len(source_files), cur_id+1)
            pbar.refresh()


if __name__ == "__main__":

    epilog = """HINTS:

* Please consider to use an appropriate BLAS library like MKL for Intel processors in order to boost up scaling speed.
  For switching a BLAS library, you may confer:
  https://conda-forge.org/docs/maintainer/knowledge_base.html?highlight=mesa#switching-blas-implementation

* Please choose an appropriate number of scaling workers and have a look at the number of interations.
  A small number of threads (1 or 2)  might be sufficient as BLAS also takes care of parallelization.

EXAMPLES:

* scale images of two directories to a new target size (1024, 1024):

  python scale_dataset.py --indirs "/path/to/dataset/images_A" "/path/to/dataset/images_B" \\ 
      --outdirs "/path/to/dataset/images_A_scale_1024" "/path/to/dataset/images_B_scale_1024" \\
      --height 1024 --width 1024

* scale depth maps of two directories to a new target size (1024, 1024):

  python scale_dataset.py -r NEAREST --indirs "/path/to/dataset/depth_maps" \\ 
      --outdirs "/path/to/dataset/depth_maps_scale_1024" \\
      --height 1024 --width 1024

* scale omnidirectional depth maps of a directory to a new target size (1024, 1024)
  mapping invalid pixels (zero depth) outside the FOV to NaN:

  python scale_dataset.py -r NEAREST --indirs "/path/to/dataset/depth_maps" \\ 
      --outdirs "/path/to/dataset/depth_maps_scale_1024" \\
      --height 1024 --width 1024 --map_val_to_nan 0"""


    parser = argprs.ArgumentParser(description='Scale the images or maps in certain directories and save them into other directories', epilog=epilog, formatter_class=argprs.RawDescriptionHelpFormatter)
    parser.add_argument('--indirs', '-i', type=Path, nargs='+', required=True, help="input directories <in1> \
            [<in2> [...]]")
    parser.add_argument('--outdirs', '-o', type=Path, nargs='+', required=True, help="output directories <out1> \
            [<out2> [...]]")
    parser.add_argument('--width', '-w', type=int, required=True, help="target width")
    parser.add_argument('--height', type=int, required=True, help="target height")
    parser.add_argument('--resample_method', '-r', default="BILINEAR", type=str, help="PyTorch resample method (BILINEAR, \
            BICUBIC, NEAREST), default is BILINEAR")
    parser.add_argument('--lossy', action="store_true", help="save lossy images (default: lossless)")
    parser.add_argument('--quality', '-q', type=int, default=0, choices=range(0,101), help="quality for lossy / compression \
            effort for lossless")
    parser.add_argument('--method', '-m', type=int, default=0, choices=range(0,7), help="quality / speed trade-off (0 for \
            best speed, 6 for best file size or quality)")
    parser.add_argument('--num_threads', '-n', type=int, default=2, help="number of worker threads (default: 2)")
    parser.add_argument('--file_ext', '-f', type=str, default=SUPPORTED_EXT, nargs='+', help="file extensions of files to be \
            considered")
    parser.add_argument('--out_ext', type=str, default=None, help="output file extention (None = same as input)")
    parser.add_argument('--num_files_per_folder', type=int, default=-1, help="number of files to be considered in each onput \
            folder (default: -1 = all)")
    parser.add_argument('--map_val_to_nan', type=float, help="map a value to NaN before scaling (helpful to avoid mixing up \
            invalid pixels with valid during interpolation; only works for float images)")
    args = parser.parse_args()
    
    if len(args.indirs) != len(args.outdirs):
        raise ValueError("The number of input directories must match the number of output directories.")
    
    if args.resample_method is not None:
        args.resample_method = eval(f"transforms.InterpolationMode.{args.resample_method}")
    
    for source_dir, target_dir in zip(args.indirs, args.outdirs):
        if not source_dir.is_dir():
            raise ValueError(f"The input directory {source_dir} does not exist.")
        if source_dir == target_dir:
            raise ValueError(f"Input and outout directory must be different to prevent overwriting. Got: {source_dir}")
        
    assert len(args.file_ext) > 0
    for e in args.file_ext:
        if e[0] != '.':
            raise ValueError("The input extension must start with a full stop (e.g. '.png' instead of 'png')")
        if e.lower() not in SUPPORTED_EXT:
            raise NotImplementedError(f"The input extension {e} is not supported yet.")
        
    if args.out_ext is not None:
        if args.out_ext[0] != '.':
            raise ValueError("The output extension must start with a full stop (e.g. '.png' instead of 'png')")
        if args.out_ext not in SUPPORTED_EXT:
            raise NotImplementedError(f"The output extension {e} is not supported yet.")
        
    all_source_files=[]
    all_target_files=[]    
    
    for source_dir, target_dir in zip(args.indirs, args.outdirs):       
        for ext in args.file_ext:
            regex = f"{source_dir}/*{ext}"
            source_files = [Path(f) for f in sorted(glob.glob(regex))]
            if len(source_files) == 0:
                continue # with another ext
            
            if args.num_files_per_folder > 0:
                source_files = source_files[:args.num_files_per_folder]
            
            if args.out_ext is None:
                target_files = [target_dir / f.name for f in source_files]
            else:
                target_files = [target_dir / (f.stem + args.out_ext)  for f in source_files]
            all_source_files += source_files
            all_target_files += target_files
            
            target_dir.mkdir(parents=True, exist_ok=True)
       
    threads = []
    file_id_counter = AtomicInteger(0)
    scaler = transforms.Resize((args.height, args.width), interpolation=args.resample_method)
    
    for thread_id in range(args.num_threads):
        my_thread = threading.Thread(target=run, args=(file_id_counter, all_source_files, all_target_files, thread_id, args, scaler))
        threads.append(my_thread)
        my_thread.start()
    
    for thread_id in range(args.num_threads):
        threads[thread_id].join()
