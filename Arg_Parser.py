# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------
import os
import argparse

# the Root directory for all raw and processed data
root_dir = 'HiC'  # Example of root directory name

res_map = {'5kb': 5_000, '10kb': 10_000, '25kb': 25_000, '50kb': 50_000, '100kb': 100_000, '250kb': 250_000,
           '500kb': 500_000, '1mb': 1_000_000}

# 'train' and 'valid' can be changed for different train/valid set splitting
set_dict = {'train': [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22],
            'valid': [2, 6, 10, 12],
            'test': (4, 14, 16, 20)}

help_opt = (('--help', '-h'), {
    'action': 'help',
    'help': "Print this help message and exit"})


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


# chr12_10kb.npz, predict_chr13_40kb.npz
def chr_num_str(x):
    start = x.find('chr')
    part = x[start + 3:]
    end = part.find('_')
    return part[:end]


def chr_digit(filename):
    chrn = chr_num_str(os.path.basename(filename))
    if chrn == 'X':
        n = 23
    else:
        n = int(chrn)
    return n


def data_read_parser():
    parser = argparse.ArgumentParser(description='Read raw data from Rao\'s Hi-C.', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:10kb]',
                           default='10kb', choices=res_map.keys())
    misc_args.add_argument('-q', dest='map_quality', help='Mapping quality of raw data[default:MAPQGE30]',
                           default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0'])
    misc_args.add_argument('-n', dest='norm_file', help='The normalization file for raw data[default:KRnorm]',
                           default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm'])
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_down_parser():
    parser = argparse.ArgumentParser(description='Downsample data from high resolution data', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
                          default='10kb', choices=res_map.keys(), required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
                          default='40kb', required=True)
    req_args.add_argument('-r', dest='ratio', help='REQUIRED: The ratio of downsampling[example:16]',
                          default=16, type=int, required=True)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_divider_parser():
    parser = argparse.ArgumentParser(description='Divide data for train and predict', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          required=True)
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
                          default='10kb', choices=res_map.keys(), required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
                          default='40kb', required=True)
    req_args.add_argument('-lrc', dest='lr_cutoff', help='REQUIRED: cutoff for low resolution maps[example:100]',
                          default=100, type=int, required=True)
    req_args.add_argument('-s', dest='dataset', help='REQUIRED: Dataset for train/valid/predict(all)',
                          default='train', choices=['K562_train', 'K562_valid', 'K562_test', 'GM12878_train', 'GM12878_valid', 'GM12878_test',
                                                    'train', 'valid','test'], )
    hicarn_args = parser.add_argument_group('HiCARN Arguments')
    hicarn_args.add_argument('-chunk', dest='chunk', help='REQUIRED: chunk size for dividing[example:40]',
                              default=40, type=int, required=True)
    hicarn_args.add_argument('-stride', dest='stride', help='REQUIRED: stride for dividing[example:40]',
                              default=40, type=int, required=True)
    hicarn_args.add_argument('-bound', dest='bound', help='REQUIRED: distance boundary interested[example:201]',
                              default=201, type=int, required=True)
    hicarn_args.add_argument('-scale', dest='scale', help='REQUIRED: Downpooling scale[example:1]',
                              default=1, type=int, required=True)
    hicarn_args.add_argument('-type', dest='pool_type', help='OPTIONAL: Downpooling type[default:max]',
                              default='max', choices=['max', 'avg'])
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_predict_parser():
    parser = argparse.ArgumentParser(description='Predict data using HiCARN model', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', default='GM12878', help='REQUIRED: Cell line for analysis[example: GM12878]',
                          required=False)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example: 40kb]',
                          default='40kb', required=False)
    req_args.add_argument('-f', dest='file_name', help='REQUIRED: Matrix file to be enhanced[example: ',
                                                       default='Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_GM12878_test.npz', required=False)
    req_args.add_argument('-m', default='HiCARN_1', dest='model', help='REQUIRED: Choose your model[example: HiCARN_1]', required=False)
    gan_args = parser.add_argument_group('GAN model Arguments')
    gan_args.add_argument('-ckpt', default='HiC/checkpoints/03_12_22_56_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_HiCARN_1_GM12878_total.pytorch', dest='checkpoint', help='REQUIRED: Checkpoint file of HiCARN model',
                          required=False)
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('--cuda', dest='cuda', help='Whether or not using CUDA[default:1]',
                           default=0, type=int)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser
