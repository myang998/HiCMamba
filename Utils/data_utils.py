# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the HiCARN implementation, utilities for HiC matrix processing with support of multichannel matrices.
# References:
# HiCARN: https://github.com/OluwadareLab/HiCARN
# --------------------------------------------------------

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tempfile
from scipy.sparse import coo_matrix
import numpy as np
import logging


def get_oe_matrix(matrix, bound=100000000, oe=True):
    max_offset = min(matrix.shape[0], bound)

    expected = [np.mean(np.diagonal(matrix, offset)) for offset in range(max_offset)]

    if oe:
        e_matrix = np.zeros_like(matrix, dtype=np.float32)
        oe_matrix = np.zeros_like(matrix, dtype=np.float32)
        for i in range(matrix.shape[0]):
            for j in range(max(i - bound + 1, 0), min(i + bound, matrix.shape[1])):
                e_matrix[i][j] = expected[abs(i - j)]
                oe_matrix[i][j] = matrix[i][j] / expected[abs(i - j)] if expected[abs(i - j)] != 0 else 0

        return oe_matrix, e_matrix
    else:
        return expected


def print_info(s: str):
    print(s)
    logging.info(s)


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

def readcoo2mat(cooFile, normFile, resolution):
    """
    Function used for reading a coordinated tag file to a square matrix.
    """
    norm = open(normFile, 'r').readlines()
    norm = np.array(list(map(float, norm)))
    compact_idx = list(np.where(np.isnan(norm) ^ True)[0])
    pd_mat = pd.read_csv(cooFile, sep='\t', header=None, dtype=int)
    row = pd_mat[0].values // resolution
    col = pd_mat[1].values // resolution
    val = pd_mat[2].values
    mat = coo_matrix((val, (row, col)), shape=(len(norm), len(norm))).toarray()
    mat = mat.astype(float)
    norm[np.isnan(norm)] = 1
    mat = mat / norm
    mat = mat.T / norm
    HiC = mat + np.tril(mat, -1).T
    return HiC, norm, compact_idx

def compactM(matrix, compact_idx, fn=None, verbose=False):
    """
    Compacts the matrix according to the index list, using a memory-mapped file to handle large arrays.
    Stores the memory-mapped file in the specified directory.
    """


    compact_size = len(compact_idx)
    new_shape = (matrix.shape[0], compact_size, compact_size)
    print(new_shape)

    if fn:
        memmap_dir = 'tmp'
        if not os.path.exists(memmap_dir):
            os.makedirs(memmap_dir)
        out_fn = os.path.join(memmap_dir, fn)
        print(out_fn)
        result = np.memmap(str(out_fn), dtype='float64', mode='w+', shape=new_shape)
    else:
        result = np.zeros(new_shape).astype(matrix.dtype)
    # 创建一个 memory-mapped 数组

    if verbose:
        print(f'Compacting a {matrix.shape} shaped matrix to {result.shape} shaped!')

    for i, idx in enumerate(compact_idx):
        result[..., i, :] = matrix[..., idx, compact_idx]

    # 确保变更被保存到磁盘
    if fn:
        result.flush()
    return result

# Modified: add multichannel support
def spreadM(c_mat, compact_idx, full_size, convert_int=True, verbose=False):
    """
    Spreads the matrix according to the index list (a reversed operation to compactM).
    """

    new_shape = list(c_mat.shape)

    new_shape[-1] = full_size
    new_shape[-2] = full_size

    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    result = np.memmap(tmp_dir + '/result.dat', dtype='float64', mode='w+', shape=tuple(new_shape))
    # result = np.zeros(new_shape).astype(c_mat.dtype)

    if convert_int: result = result.astype(np.int)
    if verbose: print('Spreading a', c_mat.shape, 'shaped matrix to', result.shape, 'shaped!')
    for i, s_idx in enumerate(compact_idx):
        result[..., s_idx, compact_idx] = c_mat[..., i, :]
    return result

except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}

# Modified: add multichannel support
def together(matlist, indices, corp=0, species='hsa', tag='HiC'):
    """
    Constructs a full dense matrix.
    """
    chr_nums = sorted(list(np.unique(indices[:, 0])))
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]: chr_nums[-1] = except_chr[species][chr_nums[-1]]
    print(f'{tag} data contain {chr_nums} chromosomes')
    c, h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    tmp_dir = 'tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:, 0] == num)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0, 1]
        # full_mat = np.zeros((c, width, width))
        shape = (c, width, width)
        full_mat = np.memmap(tmp_dir+f'/full_mat_{n}.dat', dtype='float64', mode='w+', shape=shape)
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                _, h, w = sub.shape
            full_mat[:, i:i + h, j:j + w] = sub
        results[n] = full_mat
    return results

def dense2tag(matrix):
    """
    Converts a square matrix (dense) to coo-based tag matrix.
    """
    matrix = np.rint(matrix, dtype=float).astype('int')
    matrix = np.triu(matrix)
    tag_len = np.sum(matrix)
    tag_mat = np.zeros((tag_len, 2), dtype=int)
    coo_mat = coo_matrix(matrix)
    row, col, data = coo_mat.row, coo_mat.col, coo_mat.data
    start_idx = 0
    for i in range(len(row)):
        end_idx = start_idx + data[i]
        tag_mat[start_idx:end_idx, :] = (row[i], col[i])
        start_idx = end_idx
    return tag_mat, tag_len

def tag2dense(tag, nsize):
    """
    Coverts a coo-based tag matrix to densed square matrix.
    """
    coo_data, data = np.unique(tag, axis=0, return_counts=True)
    row, col = coo_data[:, 0], coo_data[:, 1]
    dense_mat = coo_matrix((data, (row, col)), shape=(nsize, nsize)).toarray()
    dense_mat = dense_mat + np.triu(dense_mat, k=1).T
    return dense_mat

def pooling(mat, scale, pool_type='max', return_array=False, verbose=True):
    mat = torch.tensor(mat).float()
    if len(mat.shape) == 2:
        mat.unsqueeze_(0)  # need to add channel dimension
    if scale > 1:
        if pool_type == 'avg':
            out = F.avg_pool2d(mat, scale)
        elif pool_type == 'max':
            out = F.max_pool2d(mat, scale)
    else:
        out = mat
    if return_array:
        out = out.squeeze().numpy()
    if verbose:
        print('({}, {}) sized matrix is {} pooled to ({}, {}) size, with {}x{} down scale.'.format(*mat.shape[-2:],
                                                                                                   pool_type,
                                                                                                   *out.shape[-2:],
                                                                                                   scale, scale))
    return out
