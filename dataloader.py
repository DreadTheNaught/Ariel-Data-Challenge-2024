import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from natsort import natsorted
import preprocessing.utils
import numpy as np
import time

class PreprocessAndLoad(Dataset):
    def __init__(self, root, flag, split, a_transform=None, f_transform=None):
        super().__init__()
        self.f_data = []
        self.a_data = []
        self.a_transform = a_transform
        self.f_transform = f_transform
        self.root_dir = root
        self.cut_inf, self.cut_sup = 39, 321
        self.files = natsorted(os.listdir(os.path.join(self.root_dir, split)))
        self.adc = pd.read_csv(os.path.join(root, f'{split}_adc_info.csv'))
        axis = pd.read_parquet(os.path.join(self.root_dir, 'axis_info.parquet'))
        self.integration_airs = axis['AIRS-CH0-integration_time'].dropna().values
        self.integration_fgs1 = np.ones(axis.shape[0]) * 0.1
        self.flag = flag
        self.split = split

        if self.split == "train":
            self.labels = pd.read_csv(os.path.join(root, f'{split}_labels.csv'))

        for planet in self.files:
            self.a_data.append((os.path.join(self.root_dir, split, planet, "AIRS-CH0_signal.parquet"), int(planet)))
            self.f_data.append((os.path.join(self.root_dir, split, planet, "FGS1_signal.parquet"), int(planet)))

    def __len__(self):
        return len(self.a_data)  # Assuming both AIRS and FGS1 data have the same length

    def __getitem__(self, index):
        start = time.time()
        AIRS_data_path, a_planet = self.a_data[index]
        FGS1_data_path, f_planet = self.f_data[index]

        AIRS_data = pd.read_parquet(AIRS_data_path)
        FGS1_data = pd.read_parquet(FGS1_data_path)
        
        AIRS_data = AIRS_data.values.astype(np.float64).reshape((AIRS_data.shape[0], 32, 356))
        FGS1_data = FGS1_data.values.astype(np.float64).reshape((FGS1_data.shape[0], 32, 32))
        print(f'FGS1 - {FGS1_data.shape}')
        adc_info = self.adc.loc[self.adc['planet_id'] == a_planet].iloc[0]
        print('before ACD')
        AIRS_data = preprocessing.utils.adc_convert(
            AIRS_data,
            float(adc_info['AIRS-CH0_adc_gain']),
            float(adc_info['AIRS-CH0_adc_offset'])
        )
        FGS1_data = preprocessing.utils.adc_convert(
            FGS1_data,
            float(adc_info['FGS1_adc_gain']),
            float(adc_info['FGS1_adc_offset'])
        )
        print(f'FGS1 - {FGS1_data.shape}')
        print('AFTER ACD')

        AIRS_data = AIRS_data[:, :, self.cut_inf:self.cut_sup]

        calibration_files = {
            'flat': 'flat.parquet',
            'dark': 'dark.parquet',
            'dead': 'dead.parquet',
            'linear_corr': 'linear_corr.parquet'
        }

        def load_calibration_data(calibration_folder,planet_id, calibration_type, cut_range=None,reshape =None):
            path = os.path.join(self.root_dir, f'{self.split}/{planet_id}/{calibration_folder}/{calibration_files[calibration_type]}')
            data = pd.read_parquet(path).values.astype(np.float64)
            if reshape:
                data = data.reshape(reshape)
            if cut_range:
                data = data[:, cut_range[0]:cut_range[1]]
            return data

        flat_airs = load_calibration_data(
            "AIRS-CH0_calibration",a_planet, 
            'flat', (self.cut_inf, self.cut_sup),
            (32, 356)
            )

        dark_airs = load_calibration_data(
            "AIRS-CH0_calibration",a_planet, 
            'dark', (self.cut_inf, self.cut_sup),
            (32, 356))

        dead_airs = load_calibration_data(
            "AIRS-CH0_calibration",a_planet, 
            'dead', (self.cut_inf, self.cut_sup),
            (32, 356))

        linear_corr_airs = load_calibration_data(
            "AIRS-CH0_calibration",a_planet, 
            'linear_corr',
            reshape=(6, 32, 356))[:,:,self.cut_inf:self.cut_sup]

        flat_fgs1 = load_calibration_data(
            "FGS1_calibration",f_planet, 
            'flat',reshape=(32,32))

        dark_fgs1 = load_calibration_data(
            "FGS1_calibration",f_planet, 
            'dark',reshape=(32,32))

        dead_fgs1 = load_calibration_data(
            "FGS1_calibration",f_planet, 
            'dead',reshape=(32,32))

        linear_corr_fgs1 = load_calibration_data(
            "FGS1_calibration",f_planet, 
            'linear_corr',reshape=(6,32,32))

        if self.flag['MASK']:
            print('are We going in ?')
            AIRS_data = preprocessing.utils.mask_hot_dead(AIRS_data, dead_airs, dark_airs)
            FGS1_data = preprocessing.utils.mask_hot_dead(FGS1_data, dead_fgs1, dark_fgs1)
            print('applied mask')
        if self.flag["NLCORR"]:
            print('starting linear correction')
            AIRS_data = preprocessing.utils.apply_linear_corr(linear_corr_airs, AIRS_data)
            FGS1_data = preprocessing.utils.apply_linear_corr(linear_corr_fgs1, FGS1_data)
            print('applied linear correction')

        if self.flag["DARK"]:
            AIRS_data = preprocessing.utils.clean_dark(AIRS_data, dead_airs, dark_airs, self.integration_airs)
            FGS1_data = preprocessing.utils.clean_dark(FGS1_data, dead_fgs1, dark_fgs1, self.integration_fgs1)
            print('corrected dark current')

        AIRS_data = preprocessing.utils.get_cds(AIRS_data)
        FGS1_data = preprocessing.utils.get_cds(FGS1_data)

        if self.a_transform:
            AIRS_data = self.a_transform(AIRS_data)
        if self.f_transform:
            FGS1_data = self.f_transform(FGS1_data)

        if self.flag["TIME_BINNING"]:
            AIRS_data = preprocessing.utils.bin_obs(AIRS_data, binning=30)
            FGS1_data = preprocessing.utils.bin_obs(FGS1_data, binning=30 * 12)
            print('done binning')
        else:
            AIRS_data = AIRS_data.transpose(0, 2, 1)
            FGS1_data = FGS1_data.transpose(0, 2, 1)

        if self.flag["FLAT"]:
            AIRS_data = preprocessing.utils.correct_flat_field(flat_airs, dead_airs, AIRS_data)
            FGS1_data = preprocessing.utils.correct_flat_field(flat_fgs1, dead_fgs1, FGS1_data)
            print('flat')

        if self.split == "train":

            label = self.labels[self.labels['planet_id'] == a_planet]
            
            data = (AIRS_data, FGS1_data,a_planet)
            print('made data')
        else:  # For test split
            data = (AIRS_data, FGS1_data)
        end = time.time()
        print(f'total time - {end-start}')
        return data
    
class LoadPreprocessed(Dataset):
    def __init__(self, data_dir, labels_dir, a_transform=None, f_transform=None):
        super().__init__()
        self.f_data = []
        self.a_data = []
        self.a_transform = a_transform
        self.f_transform = f_transform
        self.root_dir = data_dir

        self.files = os.listdir(self.root_dir)
        self.files = natsorted(self.files)

        self.labels = pd.read_csv(labels_dir)

        for planet in self.files:

            self.a_data.append(os.path.join(
                self.root_dir, planet, "AIRS.pth"))
            self.f_data.append(os.path.join(
                self.root_dir, planet, "FGS1.pth"))

    def __len__(self):
        if len(self.a_data) == len(self.f_data):
            return len(self.a_data)

    def __getitem__(self, index):
        AIRS_data_path = self.a_data[index]
        FGS1_data_path = self.f_data[index]

        AIRS_data = torch.load(AIRS_data_path)
        FGS1_data = torch.load(FGS1_data_path)

        # AIRS_data = AIRS_data.values
        # FGS1_data = FGS1_data.values
        labels = torch.tensor(self.labels.iloc[index, 1:],dtype=torch.float16)
        if self.a_transform:
            AIRS_data = self.a_transform(AIRS_data)
        if self.f_transform:
            FGS1_data = self.f_transform(FGS1_data)
        data = {"signals" : {"FGS1" : FGS1_data, "AIRS" : AIRS_data}, "labels" : labels}
        return data