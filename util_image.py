#!/usr/bin/env python


from skimage.color import rgb2gray
from skimage import io
import numpy as np
import pandas as pd
import os
from random import gauss

def Rgb2Gray(images): return [rgb2gray(img) for img in images]


IsFlipped = 'IsFlipped'
FlipImage = 'FlipImage'
SideImage = 'SideImage'

def AddFlippedImages(data, skip_range=(-0.2, 0.2)):
    def newname(oldname):
        dr = os.path.dirname(oldname)
        bn = os.path.basename(oldname)
        return os.path.join(dr, 'flipped_'+bn)
    
    data = pd.DataFrame(data, copy=True)
    if IsFlipped not in data: data[IsFlipped] = pd.Series([False]*len(data), index=data.index)
    if FlipImage not in data: data[FlipImage] = pd.Series([None ]*len(data), index=data.index)    

    rows = []
    for i, row in data.iterrows():
        if row.IsFlipped == False and row.FlipImage is None:
            if (skip_range[0] <= row.steer <= skip_range[1]):
                row.im_center = np.fliplr(row.im_center)
                row.center    = newname(row.center)
                row.steer     = -1*row.steer
                row.IsFlipped = True
                row.FlipImage = row.center
                data.loc[row.name, FlipImage] = row.center
                rows.append(row)

    if len(rows)>0: 
        data2 = pd.DataFrame(rows, copy=True)
        data = data.append(data2, ignore_index=True)

    return data


def AddSideImage(data, left_steer=(0.2,0.05), right_steer=(-0.2,0.05), basedir=None):

    if basedir is None:
        basedir = ''
        if hasattr(data, 'params'): basedir = data.params('BaseDir')

    def newname(oldname):
        dr = os.path.dirname(oldname)
        bn = os.path.basename(oldname)
        return os.path.join(dr, 'sideim_'+bn)

    def newrow(row, key, steer, bdir):
        row = pd.Series(row, copy=True)
        row.im_center = io.imread(os.path.join(bdir, row[key]))
        row[SideImage] = False
        row.center = newname(row[key])
        row.steer = steer
        return row

    data = pd.DataFrame(data, copy=True)
    if SideImage not in data: data[SideImage] = pd.Series([False]*len(data), index=data.index)

    rows = []
    for i, row in data.iterrows():
        notSide = row.center.find('sideim_') == -1
        if notSide and row[SideImage] == False:
            data.loc[row.name, SideImage] = True
            rows.append(newrow(row, 'left', gauss(*left_steer), basedir))
            rows.append(newrow(row, 'right',gauss(*right_steer), basedir))

    if len(rows)>0: 
        data2 = pd.DataFrame(rows, copy=True)
        data = data.append(data2, ignore_index=True)
        
    return data

#%%


