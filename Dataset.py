import torch.nn as nn
from torchvision import models
from torchvision import transforms
#from lib.rpn_util import *
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import os
import math
from PIL import Image
import re
from easydict import EasyDict as edict
import cv2


def read_A2D2_label(file, P, use_3d_for_2d=False):
    """
    Reads the A2D" label file from disc.

    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []

    text_file = open(file, 'r')

    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         class_id
       1    truncated    Integer (0,1,2,3) indicating truncation state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    
    # step 1: getting P Frame from C frame
    # Facts:
    # - share same origin in 0
    # - y/z plane of C and x/y plane of P coincide
    # - C: right handed system, X pointing along optical axis, y pointing left, z pointing up
    # - P: right handed system, Z pointing into image plane, y pointing down, x pointing right
    A_0P = np.zeros((3,3))
    A_0P[:,2] = A_0C[:,0] # P's z-axis equals C's x-axis 
    A_0P[:,0] = -A_0C[:,1] # P's x-axis equals C's y-axis 
    A_0P[:,1] = -A_0C[:,2] # P's y-axis equals C's z-axis 

    t_P_0 = t_C_0 # translation is the same
    #print(A_0P)
    '''
    for line in text_file:

        pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
                              + '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n')
                             .replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))

        parsed = pattern.fullmatch(line)
        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            cls = parsed.group(1)
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = (float(parsed.group(5)))
            y = (float(parsed.group(6)))
            x2 = (float(parsed.group(7)))
            y2 = (float(parsed.group(8)))

            width = x2 - x + 1
            height = y2 - y + 1

            l3d = (float(parsed.group(9)))
            w3d = (float(parsed.group(11)))
            h3d = (float(parsed.group(10)))


            cx3d = (float(parsed.group(12))) # center of car in 3d
            cy3d = (float(parsed.group(13))) # bottom of car in 3d
            cz3d = (float(parsed.group(14))) # center of car in 3d
            rotY = float(parsed.group(15))

            # actually center the box
            cz3d += (h3d*0.4)
            cy3d += (w3d*0.25)
            cx3d += (l3d*0.25)
            

            elevation = (1.65 - cy3d)

            coord3d = (P) @ (np.array([cx3d, cy3d, cz3d, 1]))
   
            # store the projected instead
            cx3d_2d = coord3d[0]
            cy3d_2d = coord3d[1]
            cz3d_2d = coord3d[2]

            cx = cx3d_2d / cz3d_2d
            cy = cy3d_2d / cz3d_2d
            #cx= uv[0]
            #cy= uv[1]
            
            # encode occlusion with range estimation
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            if occ == 0: vis = 1
            elif occ == 1: vis = 0.66
            elif occ == 2: vis = 0.33
            else: vis = 0.0

            while rotY > math.pi: rotY -= math.pi * 2
            while rotY < (-math.pi):rotY += math.pi * 2

            # recompute alpha
            alpha = convertRot2Alpha(rotY, cz3d, cx3d)

            obj.elevation = elevation
            obj.cls = cls
            obj.occ = occ > 0
            obj.ign = ign
            obj.visibility = vis
            obj.trunc = trunc
            obj.alpha = alpha
            obj.rotY = rotY

            # is there an extra field? (assume to be track)
            if len(parsed.groups()) >= 16 and parsed.group(16).isdigit(): obj.track = int(parsed.group(16))

            obj.bbox_full = [x, y, width, height]
            #print(obj.bbox_full)
            #obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, rotY, cx3d, cy3d, cz3d, alpha]
            #print(obj.bbox_3d)
            obj.center_3d = [cx3d, cy3d, cz3d]
            #print(obj.center_3d)

            gts.append(obj)

    text_file.close()

    return gts
def convertRot2Alpha(ry3d, z3d, x3d):

    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
    #alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha

class A2D2_3D_det_Dataset(torch.utils.data.Dataset):
    """
    A single Dataset class is used for the 3D monocular object detection ,
    which implements the __init__ and __get__ functions from PyTorch.
    """

    def __init__(self, image_dir, label_dir, csv_file,transform= None):
        imdb = []
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.annotations = pd.read_csv(csv_file)
        
        self.transform = None
        
        """
        # frame naming convention
        # 0 -> ego frame
        # C -> camera mount frame
        # P -> camera frame uesd for projection
        # L -> Lidar mount frame

        # our test point, in 0 
        #X_Test_0 = np.array([10, 0, 0])

        # C Frame
        self.A_0C = np.array([[0.9997595802751106,    -0.021482638328964472, 0.00439066],
                 [0.021372184646008877,   0.9994877740097232,   0.02382057],
                 [-0.004900139956328717, -0.02372100030596073,  0.99970661]
                ]) # "x/y-axis" in "view" become columns; 3rd column = cross(x,y)
        self.t_C_0 = np.array([1.711045726422736, -5.735179668849011e-09, 0.9431449279047172]) # aka "origin" in "view"

        # L Frame
        self.A_0L = np.array([[-0.007840892071133887, 0.9807203123118539,    0.19525929],
                 [-0.9999275831501491,  -0.005906897013169535, -0.01048509],
                 [-0.00912956006775139, -0.19532736005643384,   0.98069561]
                ]) # "x/y-axis" in "view" become columns; 3rd column = cross(x,y)
        self.t_L_0 = np.array([1.7183999999999997, 3.8163916471489756e-17, 1.1201406050000002]) # aka "origin" in "view"

        # Projection Matrix
        self.K = np.array([[1687.3369140625,               0.0, 965.4341405582381],
              [            0.0, 1783.428466796875, 684.4193604186803],
              [            0.0,               0.0,               1.0]])

        
        """
        self.P = np.array([[ 1.00145047e+03, -1.66583918e+03,  3.52945654e+01, -1.74681545e+03],
                 [ 6.76424384e+02, -2.78547356e+01, -1.78625897e+03,  5.27308035e+02],
                    [ 9.99759581e-01,  2.13721864e-02, -4.90013971e-03, -1.70601282e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self.P_inv = np.linalg.inv(self.P)
    def __getitem__(self, index):
        imobj= edict()
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 2])
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        #image = Image.open(img_path).convert("RGB")
        image= cv2.imread(img_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image= cv2.resize(image, (480,304), interpolation = cv2.INTER_AREA)
        image = np.swapaxes(image, 1,2)
        image = np.swapaxes(image, 0,1)
        #gt= read_A2D2_label(label_path, self.A_0C,self.t_C_0,self.A_0L, self.t_L_0,self.K)
        gt = read_A2D2_label(label_path, self.P)
        imobj.P= self.P
        imobj.p2_inv= self.P_inv
        imobj.gts= gt
        imobj.image= image

        return imobj

    def __len__(self):
        return len(self.annotations)
    @staticmethod
    def collate(batch):
        """
        Defines the methodology for PyTorch to collate the objects
        of a batch together, for some reason PyTorch doesn't function
        this way by default.
        """

        imgs = []
        imobjs = []

        # go through each batch
        for sample in batch:
            
            # append images and object dictionaries
            imgs.append(sample[0])
            imobjs.append(sample[1])

        # stack images
        #imgs = np.array(imgs)
        #imgs = torch.tensor(imgs)

        return imgs, imobjs

