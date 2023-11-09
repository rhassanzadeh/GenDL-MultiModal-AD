import numpy as np
import nibabel as nib
import hdf5storage

from torch.utils.data import Dataset
import torch

    
class sMRIDataset(Dataset): 
    def __init__(self, data):
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        subj_info = self.data.loc[idx]
        sMRI_dir = subj_info.sMRI_path
        label = subj_info.Diagnosis # 0: CN, 1: AD
        
        #load sMRI        
        sMRI = nib.load(sMRI_dir).get_fdata()
        
        # normalize to [-1, 1]
        sMRI = 2 * ((sMRI - np.min(sMRI)) / (np.max(sMRI) - np.min(sMRI))) - 1
        
        sMRI = torch.from_numpy(sMRI.astype('float32').reshape(1, 121, 145, 121))
        
        return sMRI
    
    
class FNCDataset(Dataset): 
    def __init__(self, data):
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        subj_info = self.data.loc[idx]
        FNC_dir = subj_info.FNC_path
        label = subj_info.Diagnosis # 0: CN, 1: AD
        
        #load FNC
        sFNC = hdf5storage.loadmat(FNC_dir)['sFNC'].astype('float32').reshape(1, 1378)
        sFNC = sFNC.reshape(1378)
        
        # normalize to [-1, 1]
        sFNC = 2 * ((sFNC - np.min(sFNC)) / (np.max(sFNC) - np.min(sFNC))) - 1
        
        sFNC = torch.from_numpy(sFNC)
        
        return sFNC
    
    
