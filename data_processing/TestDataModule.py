from cgi import test
from email.mime import base
import glob
import ntpath
from importlib.resources import path
import numpy as np
import pytorch_lightning as pl
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class TestDataModule(pl.LightningDataModule):
    def __init__(self, data_path, dataset, batch_size=1, split_file=None,num_workers=8, stage='test'):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.split_file = split_file
        self.dataset = dataset
        self.num_workers = num_workers
        self.stage = stage

    def setup(self, stage = None):
        img_paths = []
        OD_paths = []
        OC_paths = []
        names = []

        for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*.png')):
            img_paths.insert(len(img_paths),path)
            n = ntpath.basename(path)
            names.append(n)
        try:
            for path in sorted(glob.glob(self.data_path + '/images/' + self.dataset + '/*/*.png')):
                img_paths.insert(len(img_paths),path)
                n = ntpath.basename(path)
                names.append(n)
        except:
            print('NO TEST FOLDER')
        self.img_paths = img_paths
        for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*.png')):
                OD_paths.insert(len(OD_paths),path)
        for path in sorted(glob.glob(self.data_path + '/OC/' + self.dataset + '/*.png')):
                OC_paths.insert(len(OC_paths),path)
        try: #EN CASO DE NO POSEER MASCARAS PARA EL DATASET 
            
            for path in sorted(glob.glob(self.data_path + '/OD1/' + self.dataset + '/*/*.png')):
                OD_paths.insert(len(OD_paths),path)
            
            for path in sorted(glob.glob(self.data_path + '/OC/' + self.dataset + '/*/*.png')):
                OC_paths.insert(len(OC_paths),path)
        except:
            print('NO MASK FOR THIS DATASET')


        if stage == 'test' and self.split_file:
            #print("ëNTRO ACA")
            test_names = np.array((self.split_file['split']['test']).split(','))
            print('Test names: ', test_names)
            self.test_data = TestDataset(img_paths,OD_path= OD_paths, OC_path=OC_paths, split_names= test_names,stage='test')
        elif self.split_file is None:
            print(' No split_file were found, we will predict all the images')
            self.test_data = TestDataset(img_paths,OD_path= OD_paths, OC_path=OC_paths,split_names=names)

        if stage == "predict":
            self.pred_data = TestDataset(self.img_paths,stage=self.stage)
    
    def test_dataloader(self):
        '''Return DataLoader for Testing Data here'''

        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        ''' return the DataLoader for Prediction Data here'''

        return DataLoader(self.pred_data, batch_size=self.batch_size, num_workers=self.num_workers)

class TestDataset(Dataset):

    def __init__(self, img_paths, OD_path = None, OC_path = None, split_names= None,stage='test'):
        self.stage = stage
        if stage == 'test':
            paths = []
            OD_masks = []
            OC_masks = []
            names = split_names
            #OBTENGO LOS PATHS THE LAS IMAGENES QUE PERTENECEN AL GRUPO QUE QUIERO
            for i in range(len(img_paths)): 
                base_name = ntpath.basename(img_paths[i])
                #print('BASE NAME: ', base_name)
                #print('BASE NAME: ', names)


                if (base_name in names)or (('Test/' + base_name)in names): 
                    #print('ENTRO:')

                    paths.insert(len(paths),img_paths[i])

                    if OD_path:
                        OD_masks.append(OD_path[i])
                    if OC_path:
                        OC_masks.append(OC_path[i])

            print(len(paths))
            print(len(OD_masks))
            print(len(OC_masks))
        else:
            paths = img_paths
            names = []
            for p in paths:
                base_name = ntpath.basename(p)
                names.append(base_name)

        self.paths = paths
        if OD_path:
            self.OD_masks = OD_masks
        else:
            self.OD_masks = None
        if OC_path:
            self.OC_masks = OC_masks
        else:
            self.OC_masks = None
        self.names= names
        #print(self.names)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):

        img = Image.open(self.paths[index])
        shape = img.size

        if self.OD_masks:
            OD_mask = np.array(Image.open(self.OD_masks[index]))
            
            OD_mask[OD_mask == 255] = 1
        
        if self.OC_masks:
            OC_mask = np.array(Image.open(self.OC_masks[index]))
            OC_mask[OC_mask == 255] = 1

        #print(self.paths[index], self.OD_masks[index], self.OC_masks[index])
        if self.stage == 'test':
            mask = np.array(OD_mask)
            mask = mask  + np.array(OC_mask)
        else:
            mask = []
        #print(self.names[index])
        return np.array(img), mask, self.names[index],shape

