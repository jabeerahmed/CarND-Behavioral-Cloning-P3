#!/usr/bin/env python

from util_image import AddFlippedImages, AddSideImage
from util_csv import WriteCSVFile, ReadCSVFile, CheckPathError, LOG_FILE, IMG_FILE
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage import io as io
from random import sample
from os.path import join, abspath, basename, dirname, exists, splitext, isdir, isfile
import numpy as np
import os
import pandas as pd

from Timer import Timer as time

from random import sample

TYPES = ['center']#, 'left', 'right'
IM_TYPES = ['im_center']

isCropped, CropRng, isGray = 'isCropped', 'CropRng', 'isGray'
BaseDir, LogFile, ImgDir = 'BaseDir', 'LogFile', 'ImgDir'

def load_images(imlist, basepath=''):
    imgs = []
    for path in imlist: imgs.append(io.imread(join(basepath, path)))
    return imgs
        
def imshow(im): plt.imshow(im)

def crop_image_x(image, rng): return np.take(image, rng, axis=0)

def crop_images_x(images, rng): return [ crop_image_x(im, rng) for im in images]

#==============================================================================
#  Data Base
#==============================================================================

from pandas import DataFrame

class DataBase(DataFrame):

    # temporary properties
    _internal_names = pd.DataFrame._internal_names + ['internal_cache']
    _internal_names_set = set(_internal_names)

    # normal properties
    _metadata = ['params']

    def __init__(self, *args, **kw):
        DataFrame.__init__(self, *args, **kw)
        self.params = {}

    @property
    def _constructor(self):
        return DataBase

    @property
    def imshape(self): return self.im_center[0].shape        


    def crop(self, rng=range(60, 140)):
        h = self.imshape[0]
        dh = rng.stop - rng.start
        assert (rng.stop < h), "Incorrect dimension: Current height = {}, and Range is {}".format(h, str(rng))
        assert (h > dh),       "Incorrect dimension: Current height = {}, and Range is {}".format(h, str(rng))
        for t in IM_TYPES: self[t] = crop_images_x(self[t], rng)
        self.params[isCropped] = True
        self.params[CropRng] = rng

        
    def convert_to_gray(self):
        for t in IM_TYPES: self[t] = [rgb2gray(im) for im in self[t]]
        self.params[isGray] = True
        

    def load_db_images(db, basedir='', types=TYPES):
        for t in types: db['im_'+t] = load_images(db[t], basepath=basedir)
        return db
    
    
    def CreateFromFile(path, types=TYPES):
        path = path if isdir(path) else dirname(path)
        logfile, imgdir = CheckPathError(path)
        db = DataBase(DataBase.load_db_images(ReadCSVFile(path), basedir=path, types=types))
        db.params[LogFile] = logfile
        db.params[ImgDir]  = imgdir
        db.params[BaseDir] = path
        db.params[isCropped] = False
        db.params[CropRng] = range(db.imshape[0])
        db.params[isGray] = False
        return db

    
    def WriteToFile(data, path):
        assert exists(path) == False, "Path Exists " + path
        log, img = join(path, LOG_FILE), join(path, IMG_FILE)
        os.makedirs(img, exist_ok=True)
        WriteCSVFile(data, log)
        for i, row in data.iterrows():
            im = row.im_center
            im_path = join(path, row.center)
            io.imsave(im_path, im)


    def orgIndx(self): return DataBase.GetOrgIndx(self)

    
    def sideIndx(self): return DataBase.GetSideIndx(self) 
    
    
    def flipIndx(self): return DataBase.GetFlippedIndx(self)
    
    
    def grab(self, boolarray, pcent=1.0):
        boolarray = np.copy(np.array(boolarray))
        ba = np.argwhere(boolarray)
        notBa = boolarray == False
        n  = int(len(ba) * (1.0-pcent))
        inds = np.array(sample(list(ba), n))
        print('len BA = {}, n = {}, maxInds = {}'.format(len(ba), n, np.max(inds)))
        boolarray[inds] = False
#        return boolarray, notBa
        tot = boolarray | notBa
        return self[tot]


    def GetOrgIndx(db):
        flip = np.array(db.IsFlipped if 'IsFlipped' in db else [False]*len(db))
        hasSide = np.array(db.SideImage if 'SideImage' in db else [True]*len(db))
        org = (flip==False) & (hasSide==True)
        return org

    
    def GetSideIndx(db):            
        flip = np.array(db.IsFlipped if 'IsFlipped' in db else [False]*len(db))
        hasSide = np.array(db.SideImage if 'SideImage' in db else [True]*len(db))
        side = (flip==False) & (hasSide==False)
        return side


    def GetFlippedIndx(db):
        if 'IsFlipped' not in db: return np.array([False]*len(db))
        flip = np.array(db.IsFlipped if 'IsFlipped' in db else [False]*len(db))
        hasSide = np.array(db.SideImage if 'SideImage' in db else [True]*len(db))
        flip = (flip==True) & (hasSide==True)
        return flip
    
    def GetFlippedAndSideIndx(db):
        if 'IsFlipped' not in db or 'SideImage' not in db : return np.array([False]*len(db))
        flip = np.array(db.IsFlipped)
        hasSide = np.array(db.SideImage)
        flip = (flip==True) & (hasSide==False)
        return flip

    
    def HasParams(db): return hasattr(db,'params') 


    def GetParams(db):
        return db.params if DataBase.HasParams(db) else {}
    
        
#==============================================================================
# Trainer
#==============================================================================

def CreateCenterOnly(db, rng=None, num_pcent=1.0, copy=False):
    imshape = db.im_center[0].shape
    num_ims = len(db)
    X_train = np.zeros((num_ims,) + imshape, dtype=np.float32)
    y_train = np.array(db.steer)
    for i, img in enumerate(db['im_center']): X_train[i] = img

    ## if range defined
    if (rng is not None):
        X_train = np.take(X_train, rng, axis=0)
        y_train = np.take(y_train, rng, axis=0)
        return (X_train, y_train)

    ## If num percent defined
    num_pcent = np.clip(num_pcent, 0, 1)
    if (num_pcent != 1.0):
        num_ims = int(int(num_ims) * num_pcent)
        rng = sample(range(len(X_train)), num_ims)
        X_train = np.take(X_train, rng, axis=0)
        y_train = np.take(y_train, rng, axis=0)
        return (X_train, y_train)

    ## default is all
    return (X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

class Trainer(object):
     
    
    def __init__(self, db, rng=None, num_pcent=1.0, copy=False):
        self.X_train, self.y_train = CreateCenterOnly(db, rng=rng, num_pcent=num_pcent, copy=copy)
        # if (copy): self.X_train, self.y_train = np.copy(X_train), np.copy(y_train)
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=self.imshape))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        self.model = model
        self.params = DataBase.GetParams(db)


    @property
    def imshape(self): return self.X_train[0].shape


    def train(self, val_split=0.2, shuffle=True, epoch=2, verbose=0, path='model.h5', args={}):
        res = self.model.fit(self.X_train, self.y_train, validation_split=val_split, shuffle=shuffle, nb_epoch=epoch, verbose=verbose, **args)
        self.model.save(path)
        return res


#==============================================================================
# Nvidia Trainer
#==============================================================================

from keras.layers import Conv2D, ELU, Dropout, Cropping2D, MaxPooling2D, Convolution2D
from keras.optimizers import Adam

class NVidia1(Trainer):
    
    def __init__(self, db, rng=None, num_pcent=1.0, copy=False):
        self.X_train, self.y_train = CreateCenterOnly(db, rng=rng, num_pcent=num_pcent, copy=copy)
        self.params = DataBase.GetParams(db)

        crop_h = self.params[CropRng] if (self.params is not None) and (CropRng in self.params) else (60, 140)
        top_crop, bot_crop = crop_h[0], self.imshape[0] - crop_h[1]

        model = Sequential()

        # apply crop range
        model.add(Cropping2D(cropping=((top_crop, bot_crop), (0, 0)), input_shape=(160, 320, 3)))

        # Normalize
        model.add(Lambda(lambda x: x / 255.0 - 0.5))

        # Conv1
        model.add(Conv2D(filters=12, kernel_size=5, strides=2, padding="valid"))
        model.add(ELU())
        # model.add(Dropout(.5))

        # Conv2
        model.add(Conv2D(filters=13, kernel_size=5, strides=2, padding="valid"))
        model.add(ELU())
        # model.add(Dropout(.5))

        # # Conv3
        # model.add(Conv2D(filters=24, kernel_size=5, strides=2, padding="valid"))
        # model.add(ELU())
        # model.add(Dropout(.5))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

        model.add(Flatten())

        # Fully connected layer 1
        model.add(Dense(1320))
        model.add(Dropout(.5))
        model.add(ELU())

        # Fully connected layer 2
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())

        # Fully connected layer 2
        model.add(Dense(50))
        model.add(ELU())

        model.add(Dense(1))

        adam = Adam(lr=0.001)
        model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

        self.model = model
        model.summary()


class NVidia2(Trainer):
    def __init__(self, db, rng=None, num_pcent=1.0, copy=False):
        self.X_train, self.y_train = CreateCenterOnly(db, rng=rng, num_pcent=num_pcent, copy=copy)
        self.params = DataBase.GetParams(db)

        crop_h = self.params[CropRng] if (self.params is not None) and (CropRng in self.params) else (60, 140)
        top_crop, bot_crop = crop_h[0], self.imshape[0] - crop_h[1]

        model = Sequential()

        # apply crop range
        model.add(Cropping2D(cropping=((top_crop, bot_crop), (0, 0)), input_shape=(160, 320, 3)))
        # Normalise the data
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))

        # Conv layer 1
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())

        # Conv layer 2
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())

        # Conv layer 3
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))

        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())

        # Fully connected layer 1
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())

        # Fully connected layer 2
        model.add(Dense(50))
        model.add(ELU())

        model.add(Dense(1))

        adam = Adam(lr=0.0001)

        model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
        self.model = model
        model.summary()


#%%

#==============================================================================
# Main
#==============================================================================


if __name__ == '__main__':
    path = 'data/data/flip'
#    main('data/data/org')
    root = dirname(path)
#    data = DataBase.CreateFromFile(path)
#    flip = AddFlippedImages(data)
#    side1= AddSideImage(data, left_steer=(0.2,0.02), right_steer=(-0.2,0.02), basedir=path)
#    side2= AddSideImage(flip, left_steer=(0.2,0.02), right_steer=(-0.2,0.02), basedir=path)

#    DataBase.WriteToFile(data,root+'/base')
#    DataBase.WriteToFile(flip,root+'/flip')
#    DataBase.WriteToFile(side1,root+'/side_only')
#    DataBase.WriteToFile(side2,root+'/side_flip')

    data = time.run(lambda: DataBase.CreateFromFile('data/data/base'))
    
    


# #%%
#
# from pandas import DataFrame
#
# class SubclassedDataFrame2(DataFrame):
#
#     # temporary properties
#     _internal_names = pd.DataFrame._internal_names + ['internal_cache']
#     _internal_names_set = set(_internal_names)
#
#     # normal properties
#     _metadata = ['added_property']
#
#     @property
#     def _constructor(self):
#         return SubclassedDataFrame2
#
# from pandas import DataFrame
#
# class SubData(DataFrame):
#
#     # temporary properties
#     _internal_names = pd.DataFrame._internal_names + ['internal_cache']
#     _internal_names_set = set(_internal_names)
#
#     # normal properties
#     _metadata = ['params']
#
#     def __init__(self, *args, **kw):
#         DataFrame.__init__(self, *args, **kw)
#         self.params = {}
#
#     @property
#     def _constructor(self):
#         return SubData
