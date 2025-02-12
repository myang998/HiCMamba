import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Utils.SSIM import ssim
from math import log10
from Arg_Parser import root_dir
from Models.hicmamba import HiCMamba
import argparse
import random
from scipy.stats import pearsonr, spearmanr


class MyDataset(Dataset):
    def __init__(self, data, target, index, transform=None):
        self.transform = transform
        self.data = data
        self.target = target
        self.index = index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        taget = self.target[idx]
        index = self.index[idx]

        if self.transform:
            data = self.transform(data)

        return data, taget, index


random.seed(0)
cs = np.column_stack

torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--name', type=str, default='HiCMamba')
parser.add_argument('--train_name', type=str, default='K562_train')
parser.add_argument('--valid_name', type=str, default='K562_valid')
parser.add_argument('--save_epoch', type=int, default=99999)
parser.add_argument('--model', type=str, default='HiCMamba')
args = parser.parse_args()
args = args.__dict__



# data_dir: directory storing processed data
data_dir = os.path.join(root_dir, 'data')

# out_dir: directory storing checkpoint files
out_dir = os.path.join(root_dir, 'checkpoints')
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
visdom_str = time.strftime('%m%d')

resos = '10kb10kb_d16_seed0'
chunk = 40
stride = 40
bound = 200
ds_train = 40
ds_valid = 40
pool = 'nonpool'
name = args['name']
train_name = args['train_name']
valid_name = args['valid_name']

param = {}
param['num_epochs'] = 100
param['batch_size'] = 64
param['embed_dim'] = 32
param['d_state'] = 8
param['layer'] = 2
param['learning_rate'] = 1e-4
param['mlp_ratio'] = 4
print(param)

device = f'cuda:{args["device"]}' if torch.cuda.is_available() else 'cpu'
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)
torch.cuda.set_device(device)
# prepare training dataset
train_file = os.path.join('../data_processing/data/hic_data_new_split/data_new', f'Multi_{resos}_c{chunk}_s{stride}_ds{ds_train}_b{bound}_{train_name}.npz')

train = np.load(train_file)

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

train_set = MyDataset(train_data, train_target, train_inds)

valid_file = os.path.join('../data_processing/data/hic_data_new_split/data_new', f'Multi_{resos}_c{chunk}_s{stride}_ds{ds_valid}_b{bound}_{valid_name}.npz')
valid = np.load(valid_file)

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = MyDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=param['batch_size'], shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=param['batch_size'], shuffle=False, drop_last=True)

# load network
netG = HiCMamba(embed_dim=param['embed_dim'], d_state=param['d_state'], layer=param['layer'], mlp_ratio=param['mlp_ratio']).to(device)

# loss function
criterion = nn.L1Loss()

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=param['learning_rate'])

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0
for epoch in range(1, param['num_epochs'] + 1):
    run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}

    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory

    torch.cuda.empty_cache()

    netG.train()
    train_bar = tqdm(train_loader)
    step = 0
    for data, target, _ in train_bar:
        data = data[:,:1,:,:]
        target = target[:,:1,:,:]
        step += 1
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size

        real_img = target.to(device)
        z = data.to(device)
        fake_img = netG(z)

        ######### Train HiCMamba #########
        netG.zero_grad()
        g_loss = criterion(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size
        train_bar.set_description(
            desc=f"[{epoch}/{param['num_epochs']}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
    train_gloss = run_result['g_loss'] / run_result['nsamples']

    valid_result = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()
    batch_pccs = []
    batch_srccs = []
    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            val_lr = val_lr[:,:1,:,:]
            val_hr = val_hr[:,:1,:,:]
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = sr
            hr_out = hr
            g_loss = criterion(sr, hr)

            sr_flat = sr_out.view(sr.size(0), -1).cpu().numpy()
            hr_flat = hr_out.view(hr.size(0), -1).cpu().numpy()
            pccs = [pearsonr(sr_flat[i], hr_flat[i])[0] for i in range(sr_flat.shape[0])]
            srccs = [spearmanr(sr_flat[i], hr_flat[i])[0] for i in range(sr_flat.shape[0])]

            valid_result['g_loss'] += g_loss.item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (abs(sr - hr)).mean()
            valid_result['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            valid_result['ssims'] += batch_ssim * batch_size
            valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            valid_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

            batch_pccs.append(sum(pccs) / len(pccs))
            batch_srccs.append(sum(srccs) / len(srccs))
            batch_ssims.append(valid_result['ssim'])
            batch_psnrs.append(valid_result['psnr'])
            batch_mses.append(batch_mse)
            batch_maes.append(batch_mae)
    ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
    psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
    mse_scores.append((sum(batch_mses) / len(batch_mses)))
    mae_scores.append((sum(batch_maes) / len(batch_maes)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_ssim = sum(batch_ssims) / len(batch_ssims)
    print(
        f"[Predicting in Valid set for epoch {epoch}] PSNR:, {sum(batch_psnrs) / len(batch_psnrs)}, dB SSIM: {sum(batch_ssims) / len(batch_ssims)}, PCC: {sum(batch_pccs) / len(batch_pccs)}, SRCC: {sum(batch_srccs) / len(batch_srccs)}, MSE: {sum(batch_mses) / len(batch_mses)}, MAE: {sum(batch_maes) / len(batch_maes)}")

    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        print(
            f"[Predicting in Valid set for epoch {epoch}] PSNR:, {sum(batch_psnrs) / len(batch_psnrs)}, dB SSIM: {sum(batch_ssims) / len(batch_ssims)}, PCC: {sum(batch_pccs) / len(batch_pccs)}, SRCC: {sum(batch_srccs) / len(batch_srccs)}, MSE: {sum(batch_mses) / len(batch_mses)}, MAE: {sum(batch_maes) / len(batch_maes)}")

        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}_{param["embed_dim"]}_{param["layer"]}_{param["d_state"]}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))

    if epoch % args['save_epoch'] == 0:
        tmp_ckpt_file = f'{datestr}_epoch{epoch}g_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}_{param["embed_dim"]}_{param["layer"]}_{param["d_state"]}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, tmp_ckpt_file))

final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}_{param["embed_dim"]}_{param["layer"]}_{param["d_state"]}.pytorch'
torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
