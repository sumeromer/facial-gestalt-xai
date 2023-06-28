#!/usr/bin/env python
import sys
import pathlib
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import cv2
from skimage import transform as trans

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
    else:
        ratio = float(image_size)/128.0
    dst = arcface_dst * ratio
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M

def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts

def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class NIHFacesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, metadata_file, fold, split, mean_bgr=None, image_size=224, flip=False):
        
        self.root_dir = pathlib.Path(root_dir)
        self.fold = fold
        self.split = split
        self.image_size = image_size
        df = pd.read_csv(metadata_file) # also included in the repository (./metadata/partitions.csv)
        if mean_bgr is None:
            mean_bgr = self.channel_mean(df)
        else:
            mean_bgr = np.array(mean_bgr)
        # from resnet50_ft.prototxt: np.array([91.4953, 103.8827, 131.0912])
        self.mean_bgr = mean_bgr
            
        # categories
        labels_txt = sorted([f.split('Slide')[0] for f in df['image_name']])
        self.categories = np.array(['22q11DS', 'Angelman', 'BWS', 'CdLS', 'Down', 'KS', 'NS', 'PWS', 'RSTS1', 'Unaffected', 'WHS', 'WS']) #np.unique(labels_txt)
        
        # data frame: train/test set of selected partition (one of 5-fold)
        if split=='all-images':
            self.df = df
            logger.info('NIH-Faces, all-images, #%d'%(self.df.shape[0]))
            logger.info('VGG Face-2 mean BGR=[%2.3f, %2.3f, %2.3f])'%(self.mean_bgr[0],
                                                                     self.mean_bgr[1],
                                                                     self.mean_bgr[2]))
        else:
            self.df = df.iloc[np.argwhere(df[self.fold].to_numpy()==self.split).squeeze(), :].reset_index(drop=True)
            logger.info('NIH-Faces %s / %s / #%d'%(fold, split, self.df.shape[0]))
            logger.info('NIH-Faces mean BGR=[%2.3f, %2.3f, %2.3f])'%(self.mean_bgr[0],
                                                                     self.mean_bgr[1],
                                                                     self.mean_bgr[2]))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        
        # read image, face parsing map, and 5-point RetinaFace landmarks
        image_path = pathlib.Path(self.root_dir, 'images', self.df.iloc[i]['image_name']) 
        image = np.asarray(Image.open(image_path.as_posix()).convert('RGB'))
        image = image[:, :, ::-1]  # RGB -> BGR (pretrained ResNet/VGGFace2 model was trained on BGR images)

        segmap_fpath = pathlib.Path(self.root_dir, 'features', 'segmaps', self.df.iloc[i]['image_name']) 
        segmap = np.asarray(Image.open(segmap_fpath.as_posix()))

        landmark = self.df.loc[i, ['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4', 'x5','y5']].to_numpy().astype(np.float32).reshape(5, 2)       
        image, M = norm_crop2(image, landmark, image_size=self.image_size , mode='arcface')
        segmap, _ = norm_crop2(segmap, landmark, image_size=self.image_size , mode='arcface')

        # transform landmark locations (x,y) on normalized image
        landmark_transformed = trans_points2d(landmark, M)
        # label
        label = np.argwhere(self.categories==image_path.stem.split('Slide')[0]).squeeze()
            
        # horizontal flip only in training set 
        if self.split=='train' and self.flip==True:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = torchvision.transforms.functional.hflip(Image.fromarray(image))
                segmap = torchvision.transforms.functional.hflip(Image.fromarray(segmap))
                
        # mean subtraction
        image = np.asarray(image).astype(np.float32) - self.mean_bgr
        # transform to torch tensor
        image = torchvision.transforms.functional.to_tensor(image).float()
        segmap = np.array(segmap).astype(np.uint8) # torchvision.transforms.functional.to_tensor(segmap).int()

        return image, label, segmap, image_path.stem, landmark_transformed
    
    def channel_mean(self, df):
        # calculate BGR channel mean over given fold's train set
        values = []
        for i in range(0, df.shape[0]):
            if df.iloc[i][self.fold]=='train':
                image_path = pathlib.Path(self.root_dir, 'images', df.iloc[i]['image_name']) 
                image = np.asarray(Image.open(image_path.as_posix()).convert('RGB'))
                image = image[:, :, ::-1]  # RGB -> BGR (pretrained ResNet/VGGFace2 model was trained on BGR images)
                landmark = df.loc[i, ['x1','y1', 'x2','y2', 'x3','y3', 'x4','y4', 'x5','y5']].to_numpy().astype(np.float32).reshape(5, 2)       
                image, M = norm_crop2(image, landmark, image_size=self.image_size , mode='arcface')
                values.append(list(np.mean(image.reshape(image.shape[0]*image.shape[1], 3), axis=0)))      
        return np.mean(np.array(values), axis=0)
    

