from map_proc.image_helper import *

import argparse
import logging
import glob
import re
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser('Mask maps')
    parser.add_argument('--maps', '-p', type=str, help='glob regex for maps to mask', required=True)
    parser.add_argument('--masks', '-s', type=str, help='glob regex for masks', required=True)
    parser.add_argument('--loglevel', '-l', \
            choices=['critical', 'error', 'warning', 'info', 'debug'], \
            default="warning", type=str, help="set the log level")
    parser.add_argument('--allow_diff_number_masks_maps', '-a', action='store_true', help="allow different numbers of masks and maps, mask and maps are matched by there ID")
    parser.add_argument('--output_dir', '-o', type=str, help="output directory", required=True)
    parser.add_argument('--map_id_at_index', type=int, default=0, help="if -a: the id is the (x+1)th number in filename of map")
    parser.add_argument('--mask_id_at_index', type=int, default=0, help="if -a: the id is the (x+1)th number in filename of mask")

    args = parser.parse_args()

    log_level = getattr(logging, args.loglevel.upper())
    logger = logging.getLogger('map_mask')
    logger.setLevel(log_level)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    logger.addHandler(stream)

    map_paths = glob.glob(args.maps)
    map_paths = sorted(map_paths)
    
    mask_paths = glob.glob(args.masks)
    mask_paths = sorted(mask_paths)

    if len(map_paths) == 0:
        logger.error("No maps found")
        exit(1)

    if not args.allow_diff_number_masks_maps:
        infiles = map_paths
    else:
        try:
            mask_path_ids = [get_id(path, logger, args.mask_id_at_index) for path in mask_paths]
            #logger.debug(f'{mask_paths=}')
            id_to_map_paths = {}
            for path in map_paths:
                logger.debug(f'{path=}')
                id = get_id(path, logger, args.map_id_at_index)
                logger.debug(f'{id=}')
                id_to_map_paths[id] = path
            infiles = [id_to_map_paths[x] for x in mask_path_ids]
        except Exception as e:
            logger.error(e)
            exit(1)
    if len(mask_paths) != len(infiles):
        logger.error(f"Number of maps and mask differ. Got {len(infiles)} maps but {len(mask_paths)} masks.")
        exit(1)

    for i in tqdm(range(len(infiles))):
        logger.debug(f'{infiles[i]}\t{mask_paths[i]}')
        cur_map_path = infiles[i]
        cur_mask_path = mask_paths[i]

        cur_map = read_data(cur_map_path)
        cur_mask = read_data(cur_mask_path)

        if cur_mask.dtype is not np.dtype(bool):
            tmp = np.zeros_like(cur_mask, dtype=bool)
            max_val = np.max(cur_mask)
            tmp[cur_mask == max_val] = True
            cur_mask = tmp

        if cur_map.dtype is np.dtype(float) or cur_map.dtype is np.dtype('float32'):
            cur_map[~cur_mask] = float('nan')
        else:
            cur_map[~cur_mask] = 0

        map_base, map_ext = os.path.splitext(os.path.basename(cur_map_path))
        mask_base, mask_ext = os.path.splitext(os.path.basename(cur_mask_path))
        out_fname = mask_base + map_ext
        out_path = os.path.join(args.output_dir, out_fname)
        write_data(out_path, cur_map)


        




def get_id(path, logger, id_at_index=0):
    fname = os.path.basename(path)
    logger.debug(f'{fname=}')
    numbers = re.findall("[0-9]+", fname)
    if len(numbers) == 0:
        raise Exception("Could not determine sample ID!")
    return numbers[id_at_index]


if __name__ == '__main__':
    main()
