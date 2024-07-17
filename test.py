import sys
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
import torch
from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
from Utils.io import spreadM, together
from Models.hicmamba import HiCMamba

from scipy.stats import pearsonr, spearmanr

import argparse
import os


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


help_opt = (('--help', '-h'), {
    'action': 'help',
    'help': "Print this help message and exit"})


def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
    
def get_chr_nums(data):
	inds = torch.tensor(data['inds'], dtype=torch.long)
	chr_nums = sorted(list(np.unique(inds[:, 0])))
	return chr_nums


def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
    print(filename)
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale


def predictor(deepmodel, hicmamba_loader, ckpt_file, device, data_file):
    deepmodel.load_state_dict(torch.load(ckpt_file, map_location=device))
    print(f'Loading checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []
    target_data = []
    target_inds = []
    raw_data = []
    raw_inds = []
    chr_nums = get_chr_nums(data_file)
    
    results_dict = dict()
    test_metrics = dict()
    for chr in chr_nums:
        test_metrics[f'{chr}'] = {'mse': 0, 'mae': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'pcc': 0, 'srcc': 0,
                                  'nsamples': 0}
        results_dict[f'{chr}'] = [[], [], [], [], [], [], []]  # Make respective lists for ssim, psnr, mse, and repro
	
    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(hicmamba_loader, desc='HiCMamba Predicting: '):
            lr, hr, inds = batch
            batch_size = lr.size(0)
            ind = f'{(inds[0][0]).item()}'
            test_metrics[ind]['nsamples'] += batch_size
            lr = lr.to(device)[:,:1,:,:]
            hr = hr.to(device)[:,:1,:,:]
            out = deepmodel(lr)

            sr_flat = out.view(out.size(0), -1).cpu().numpy()
            hr_flat = hr.view(hr.size(0), -1).cpu().numpy()
            pccs = [pearsonr(sr_flat[i], hr_flat[i])[0] for i in range(sr_flat.shape[0])]
            srccs = [spearmanr(sr_flat[i], hr_flat[i])[0] for i in range(sr_flat.shape[0])]
            
            batch_mse = ((out - hr) ** 2).mean()
            batch_mae = (abs(out - hr)).mean()
            test_metrics[ind]['mse'] += batch_mse * batch_size
            test_metrics[ind]['mae'] += batch_mae * batch_size
            test_metrics[ind]['pcc'] += sum(pccs) / len(pccs)
            test_metrics[ind]['srcc'] += sum(srccs) / len(srccs)
            batch_ssim = ssim(out, hr)
            test_metrics[ind]['ssims'] += batch_ssim * batch_size
            test_metrics[ind]['psnr'] = 10 * log10(1 / (test_metrics[ind]['mse'] / test_metrics[ind]['nsamples']))
            test_metrics[ind]['ssim'] = test_metrics[ind]['ssims'] / test_metrics[ind]['nsamples']          
            ((results_dict[ind])[0]).append((test_metrics[ind]['ssim']).item())
            ((results_dict[ind])[1]).append(batch_mse.item())
            ((results_dict[ind])[2]).append(test_metrics[ind]['psnr'])
            ((results_dict[ind])[4]).append(batch_mae.item())
            ((results_dict[ind])[5]).append(sum(pccs) / len(pccs))
            ((results_dict[ind])[6]).append(sum(srccs) / len(srccs))
            
            for i, j in zip(hr, out):
                out1 = torch.squeeze(j, dim=0)
                hr1 = torch.squeeze(i, dim=0)
                out2 = out1.cpu().detach().numpy()
                hr2 = hr1.cpu().detach().numpy()
                genomeDISCO = compute_reproducibility(out2, hr2, transition=True)
                ((results_dict[ind])[3]).append(genomeDISCO)
  
            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
            target_data.append(hr.to('cpu').numpy())
            target_inds.append(inds.numpy())
            raw_data.append(lr.to('cpu').numpy())
            raw_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    target_data = np.concatenate(target_data, axis=0)
    target_inds = np.concatenate(target_inds, axis=0)
    raw_data = np.concatenate(raw_data, axis=0)
    raw_inds = np.concatenate(raw_inds, axis=0)

    mean_ssims = []
    mean_mses = []
    mean_psnrs = []
    mean_gds = []
    mean_maes = []
    mean_pcc = []
    mean_srcc = []
    result_list = [[], [], [], [], [], [], []]
    for key, value in results_dict.items():
        value[0] = round(sum(value[0])/len(value[0]), 10)
        value[1] = round(sum(value[1])/len(value[1]), 10)
        value[2] = round(sum(value[2])/len(value[2]), 10)
        value[3] = round(sum(value[3])/len(value[3]), 10)
        value[4] = round(sum(value[4]) / len(value[4]), 10)
        value[5] = round(sum(value[5]) / len(value[5]), 10)
        value[6] = round(sum(value[6]) / len(value[6]), 10)
        mean_ssims.append(value[0])
        mean_mses.append(value[1])
        mean_psnrs.append(value[2])
        mean_gds.append(value[3])
        mean_maes.append(value[4])
        mean_pcc.append(value[5])
        mean_srcc.append(value[6])
        
        print("\n")
        print("Chr", key, "SSIM: ", value[0])
        print("Chr", key, "MSE: ", value[1])
        print("Chr", key, "PSNR: ", value[2])
        print("Chr", key, "GenomeDISCO: ", value[3])
        print("Chr", key, "MAE: ", value[4])
        print("Chr", key, "PCC: ", value[5])
        print("Chr", key, "SRCC: ", value[6])
        result_list[0].append(value[0])
        result_list[1].append(value[1])
        result_list[2].append(value[2])
        result_list[3].append(value[3])
        result_list[4].append(value[4])
        result_list[5].append(value[5])
        result_list[6].append(value[6])
    print(result_list)
    print("\n")
    print("___________________________________________")
    print("Means across chromosomes")
    print("SSIM: ", round(np.mean(np.array(mean_ssims)), 4), "±", round(np.std(np.array(mean_ssims)), 4))
    print("PSNR: ", round(np.mean(np.array(mean_psnrs)), 4), "±", round(np.std(np.array(mean_psnrs)), 4))
    print("PCC: ", round(np.mean(np.array(mean_pcc)), 4), "±", round(np.std(np.array(mean_pcc)), 4))
    print("SRCC: ", round(np.mean(np.array(mean_srcc)), 4), "±", round(np.std(np.array(mean_srcc)), 4))
    print("MSE: ", round(np.mean(np.array(mean_mses)), 4), "±", round(np.std(np.array(mean_mses)), 8))
    print("MAE: ", round(np.mean(np.array(mean_maes)), 4), "±", round(np.std(np.array(mean_maes)), 8))

    print("GenomeDISCO: ", round(np.mean(np.array(mean_gds)), 4), "±", round(np.std(np.array(mean_gds)), 4))


    # print("SSIM: ", round(sum(mean_ssims) / len(mean_ssims), 10))
    # print("MSE: ", round(sum(mean_mses) / len(mean_mses), 10))
    # print("PSNR: ", round(sum(mean_psnrs) / len(mean_psnrs), 10))
    # print("GenomeDISCO: ", round(sum(mean_gds) / len(mean_gds), 10))
    # print("MAE: ", round(sum(mean_maes) / len(mean_maes), 10))
    # print("PCC: ", round(sum(mean_pcc) / len(mean_pcc), 10))
    # print("SRCC: ", round(sum(mean_srcc) / len(mean_srcc), 10))
    print("___________________________________________")
    print("\n")
    
    hicmamba_hics = together(result_data, result_inds, tag='Reconstructing: ')
    target_hics = together(target_data, target_inds, tag='Reconstructing: ')
    raw_hics = together(raw_data, raw_inds, tag='Reconstructing: ')
    return hicmamba_hics, target_hics, raw_hics
    

def save_data(carn, compact, size, file):
    hicmamba = spreadM(carn, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hicmamba, compact=compact)
    print('Saving file:', file)

def data_predict_parser():
    parser = argparse.ArgumentParser(description='Predict data using HiCMamba', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example: 40kb]',
                          default='40kb', required=False)


    req_args.add_argument('-c', dest='cell_line', default='K562', help='REQUIRED: Cell line for analysis[example: GM12878]',
                          required=False)
    req_args.add_argument('-f', dest='file_name', help='REQUIRED: Matrix file to be enhanced[example: ',
                                                       default='Multi_10kb10kb_d16_seed0_c40_s40_ds40_b200_K562_test.npz', required=False)
    req_args.add_argument('-m', default='HiCMamba', dest='model', help='REQUIRED: Choose your model[example: HiCMamba]', required=False)
    gan_args = parser.add_argument_group('GAN model Arguments')
    # GM12878-32-2-8: 04_09_16_46_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_MambaLeff_GM12878_32.pytorch
    # GM12878-32-1-8: 04_10_00_44_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_MambaLeff_GM12878_32.pytorch
    # K562-32-2-8: 04_11_15_40_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_MambaLeff_K562_32_2_8.pytorch
    # K562-32-2-4: 04_12_23_55_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_MambaLeff_K562_32_2_4.pytorch
    # HiCARN_1_K562: 04_15_18_30_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_HiCARN_1_K562_total.pytorch
    # HiCARN_1_GM12878: 04_15_15_16_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_HiCARN_1_GM12878_total.pytorch
    # HiCARN_2_GM12878: 05_27_12_25_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_HiCARN_2_GM12878.pytorch
    # HiCARN_2_K562: 05_27_12_26_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_HiCARN_2_K562.pytorch
    gan_args.add_argument('-ckpt',
                          default='HiC/checkpoints/07_16_20_47_bestg_10kb10kb_d16_seed0_c40_s40_b200_nonpool_HiCMamba_32_2_8.pytorch',
                          dest='checkpoint', help='REQUIRED: Checkpoint file of HiCMamba model',
                          required=False)


    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('--cuda', dest='cuda', help='Whether or not using CUDA[default:1]',
                           default=0, type=int)


    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    cuda = args.cuda
    model_name = args.model
    HiCMamba_file = args.file_name
    print('WARNING: Predict process requires large memory, thus ensure that your machine has ~150G memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()
    root_dir = 'HiC'
    in_dir = '../data_processing/data/hic_data_new_split/data_new'
    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    chunk, stride, bound, scale = filename_parser(HiCMamba_file)

    device = torch.device(
        f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data: {HiCMamba_file}')
    hicmamba_data = np.load(os.path.join(in_dir, HiCMamba_file), allow_pickle=True)
    hicmamba_loader = dataloader(hicmamba_data)

    model = HiCMamba(embed_dim=32, d_state=8, layer=2, mlp_ratio=4).to(device)

    hicmamba_hics, target_hics, raw_hics = predictor(model, hicmamba_loader, ckpt_file, device, hicmamba_data)

    indices, compacts, sizes = data_info(hicmamba_data)
    def save_data_n(key):
        file = os.path.join(out_dir, f'{model_name}_predict_chr{key}_{low_res}.npz')
        save_data(hicmamba_hics[key], compacts[key], sizes[key], file)
        file = os.path.join(out_dir, f'{model_name}_target_chr{key}_{low_res}.npz')
        save_data(target_hics[key], compacts[key], sizes[key], file)
        file = os.path.join(out_dir, f'{model_name}_raw_chr{key}_{low_res}.npz')
        save_data(raw_hics[key], compacts[key], sizes[key], file)

    for key in compacts.keys():
        save_data_n(key)
