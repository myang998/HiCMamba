# HiCMamba
HiCMamba: Enhancing Hi-C Resolution and Identifying 3D Genome Structures with State Space Modeling

## Data processing

1. **Download raw Hi-C data and set the environment variables**
* Download the raw Hi-C data from [GSE62525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525). [GM12878](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%5F30%2Ehic)
and [K562](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fcombined%5F30%2Ehic) are used in this work.
* Create the `RAW_dir` directory under your root directory and then unzip the raw Hi-C data into this directory.
* Set the related variables in `dataset_information.py`, including `root_dir` and `RAW_dir`.
* Also, set the directory name to store different data file.
    * RAW_dir: stores raw hic data
    * hic_matrix_dir: stores the hic matrices in npz format.
    * data_dir: stores the data for training, validation and test.

2. **Run `data_processing.Preprocess.py` to process the raw Hi-C data and generate data for training and testing**

* The script consists of four key step:
    1. Read raw data files from RAW_dir, save them in numpy matrix style(.npz files) in `hic_matrix_dir`.
    2. Read the high-coverage numpy matrices and downsampling them.
    3. Data normalization. 
    4. Data division, transforming the data into 40 * 40 submatrices.

* The script can be executed using the following command:
```
python -m data_processing.Preprocess -c GM12878
python -m data_processing.Preprocess -c K562
```

* optional
The well preprocessed data is accessible at [Google Drive](https://drive.google.com/file/d/1V1eYC9RcCEz6jTrOZ3-9DhNQBk34X8f6/view?usp=drive_link). You can use the preprocessed data through downloading the data and then moving this data into data_processing directory.

## Run HiCMamba
1. **Training**
```
python train.py
```
2. **Testing**
```
python test.py
```

## Requirements
- Python 3.8.18
- Pytorch 1.13.0+cu117
- causal-conv1d 1.0.0
- mamba_ssm 1.0.1
- Numpy 1.24.4
- Scipy 1.10.1
- Pandas 2.0.3
- Scikit-learn 1.3.2
- Matplotlib 3.7.5
- tqdm 4.66.2

The causal-conv1d and mamba_ssm are strongly recommended downloaded from the [BaiduNetdisk Link](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k) provided by [VM-UNet](https://github.com/JCruan519/VM-UNet). And then install the packages using:
```
pip install xxx.whl
```

## Acknowledgments
We express our gratitude to the authors of [HiCARN](https://github.com/OluwadareLab/HiCARN) for sharing their open-source code.