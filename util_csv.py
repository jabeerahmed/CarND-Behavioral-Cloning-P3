#!/usr/bin/env python
#%%

import os
import pandas as pd
import shutil
import numpy as np

LOG_FILE='driving_log.csv'
IMG_FILE='IMG'

def NewFileName(filename):
    f, ext = os.path.splitext(filename)
    for i in range(1000):
        f_new = f + '_{:03d}'.format(i)
        if (os.path.exists(f_new + ext) == False):
            f = f_new
            break
    return f+ext


def ReadCSVFile(filename):
    if (os.path.isdir(filename)): filename, _ = CheckPathError(filename)
    data = pd.read_csv(filename)
    if 'steer' in data.columns: return data
    return pd.read_csv(filename, names=['center', 'left', 'right', 'steer', 'gas', 'brake', 'speed'])


def SetNewImagePath(db, path=''):
    for i, row in db.iterrows():
        db.at[i, 'center'] = os.path.join(path, os.path.basename(row.center))
        db.at[i, 'left']   = os.path.join(path, os.path.basename(row.left))
        db.at[i, 'right']  = os.path.join(path, os.path.basename(row.right))
    return db

def WriteCSVFile(db, newPath):
    if (os.path.isdir(newPath)): newPath, _ = CheckPathError(newPath)
    tmpPath = None
    
    cols = [i for i in db.columns if (i not in ['im_center', 'im_right', 'im_left'])]
    
    if (os.path.exists(newPath)):
        tmpPath = NewFileName(newPath)
        os.rename(newPath, tmpPath)

    db.to_csv(newPath, columns=cols, index=False, header=True, index_label=None)
    return tmpPath
    

def FixMissingFiles(db, path):
    rows = []
    for i, row in recv.iterrows():
        im = row.center
        if not os.path.exists(os.path.join(path, im)): rows.append(i)

    s = pd.Series(rows)
    return db.drop(s)
    
#def write(db, cols=[]):db.to_csv('./test.csv', columns=cols, index=False, header=True, index_label=cols)
#%%

import argparse


def SetPath(in_args):
    parser = argparse.ArgumentParser(description='Process some integers.', prog='util_csv.SetPath')
    parser.add_argument('path',     type=str, metavar="path",           help='path to the log csv file')
    parser.add_argument('new_path', type=str, metavar="new_path",       help='new path to add to every line')
    parser.add_argument('dst',      action='store_const', const=None,   help='destination file path')
    args = parser.parse_args(in_args)

    if args.dst is None: args.dst = args.path
    db = ReadCSVFile(args.path)
    db = SetNewImagePath(db, args.new_path)
    bak = WriteCSVFile(db, args.dst)
    if (bak is not None): print("Old file saved to "+bak)


def CheckPathError(path):
    log  = os.path.join(path, LOG_FILE)
    img = os.path.join(path, IMG_FILE)
    assert os.path.exists(path) and os.path.isdir(path), "Path Not Found: " + path
    assert os.path.exists(log)  and os.path.isfile(log), "Log File Not Found: " + log
    assert os.path.exists(img)  and os.path.isdir(img), "IMG Not Found: " + img
    return log, img


def CopyDataToPath(src, dst):
    new_dir = dst
    log, img = CheckPathError(src)
    new_log, new_img = os.path.join(new_dir, LOG_FILE), os.path.join(new_dir, IMG_FILE)
    os.mkdir(new_dir)
    shutil.copytree(img, new_img), shutil.copy(log, new_log)


def MoveDataToPath(src, dst):
    new_dir = dst
    log, img = CheckPathError(src)
    new_log, new_img = os.path.join(new_dir, LOG_FILE), os.path.join(new_dir, IMG_FILE)
    os.mkdir(new_dir)
    os.rename(img, new_img), os.rename(log, new_log)


def SetRelativePath(in_args):
    parser = argparse.ArgumentParser(description='Sets all the image paths to relative paths.',
                                     prog='util_csv.SetRelativePath')
    parser.add_argument('path', help='path to the log csv file')
    args = parser.parse_args(in_args)
    path = args.path
    log, img = CheckPathError(path)

    db = ReadCSVFile(path)
    db = SetNewImagePath(db, IMG_FILE)
    bak = WriteCSVFile(db, log)
    if bak is not None: print("Old file saved to "+bak)


def OrganizeNewDataSet(in_args):
    parser = argparse.ArgumentParser(description='Process some integers.',
                                     prog='util_csv.OrganizeNewDataSet')
    parser.add_argument('path', help='path to the log csv file')
    path = parser.parse_args(in_args).path
    new_dir, bak_dir = os.path.join(path, 'org'), os.path.join(path, 'bak')
    CopyDataToPath(path, bak_dir)
    MoveDataToPath(path, new_dir)
    SetRelativePath([new_dir])

#%%

def main():
    cmds = {'setpath': SetPath, 'relpath': SetRelativePath, 'orgdata': OrganizeNewDataSet}
    parser = argparse.ArgumentParser(description='Process some integers.', prog='util_csv')
    parser.add_argument('cmd', choices=cmds.keys(),
                        metavar="sub_command={}".format(list(cmds.keys())),
                        help='choose the command')
    parser.add_argument('cmd_args', nargs='+', metavar='cmd_args', help='command args')
    args = parser.parse_args()
    cmds[args.cmd](args.cmd_args)

if __name__ == '__main__': main()