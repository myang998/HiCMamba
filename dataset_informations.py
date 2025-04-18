
# 1. the Root directory for all raw and processed data
root_dir = 'data_processing/data/hic_data'  # Example of root directory name

# 2. the folder to save files in preprocessing
RAW_dir = 'Datasets_NPZ/raw/'   # Example of directory name for raw data
hic_matrix_dir = 'mat'
data_dir = 'data'
multichannel_matrix_dir = 'multichannel_mat_new'
res_map = {'5kb': 5_000, '10kb': 10_000, '25kb': 25_000, '50kb': 50_000, '100kb': 100_000, '250kb': 250_000,
           '500kb': 500_000, '1mb': 1_000_000}

# 'train' and 'valid' can be changed for different train/valid set splitting
set_dict = {
    'all' : (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22),
    'test': (4, 14, 16, 20),
    # 'train': (1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22),
    # 'valid': (2, 6, 10, 12),
    }

abandon_chromosome_dict = {
    'GM12878' : [],
    'K562' : [9],
}

chromosome_size = {
'hg19':
    { 'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566 },
'hg38':
    { 'chrM': 16569, 'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309, 'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415}
}