# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# The all-in-one script to preprocess hic matrices. 
# --------------------------------------------------------

import os
import argparse
from dataset_informations import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', default='GM12878', dest='cell_line',
                        help='REQUIRED: Cell line for analysis[example:GM12878]',
                        required=False)
    parser.add_argument('-hr', '--high-res', help='High resolution specified[default:10kb]',
                           default='10kb', choices=res_map.keys())
    parser.add_argument('-dr', '--downsample-ratio', help='The ratio of downsampling[example:16]',
                          default=16, type=int)
    parser.add_argument('-ds', '--downsample-seed', help = 'The seed of downsampling[default:0]',
                        default = 0, type=int)
    args = parser.parse_args()
    
    cell_line = args.cell_line
    high_res = args.high_res
    ratio = args.downsample_ratio
    seed = args.downsample_seed

    # 1. Convert the raw data to npz files
    RAW2npz_command = f'python -m data_processing.RAWobserved2npz -c {cell_line} -hr {high_res}'
    os.system(RAW2npz_command)

    # 2. Downsampling data
    downsample_command = f'python -m data_processing.Downsample -c {cell_line} -hr {high_res} -lr {high_res}_d{ratio} -r {ratio} --seed {seed}'
    os.system(downsample_command)

    # 3. Data normalization
    hr_transform_command = f'python -m data_processing.Transform -c {cell_line} -r {high_res} --cutoff 255 -tn HiC'
    lr_transform_command = f'python -m data_processing.Transform -c {cell_line} -r {high_res}_d{ratio}_seed{seed} --cutoff 100 -tn HiC'
    os.system(hr_transform_command)
    os.system(lr_transform_command)

    # 4. Divide data into 40 * 40 submatrices
    for dataset in set_dict.keys():
        generate_command = f'python -m data_processing.Generate -c {cell_line} -s {dataset} -hr {high_res} -lr  {high_res}_d{ratio}_seed{seed}'
        os.system(generate_command)
