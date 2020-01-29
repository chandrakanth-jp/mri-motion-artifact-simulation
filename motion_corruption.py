# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:11:49 2020

@author: jayach01
"""

import SimpleITK as sitk
import sys, os
import random
import numpy as np
from utilities_motion_corruption import load_image, affine_translate, affine_rotate, create_transform, affine_transformation, manipulate_kspace_columns, image_reconstruction, write_image, generate_normal_motion_ixi


input_dir = '/data' # Enter path to directory containing dicom image volumes
print(os.path.exists(input_dir))

for path,subdir,files in os.walk(input_dir):
        for image in sorted(files):
            filepath = path+'/'+image
            if '.dcm' in filepath:
                image, array = load_image(filepath)
                imagesize = image.GetSize()
                print(filepath)
                print(imagesize)
                motion_table=generate_normal_motion_ixi(imagesize)
                np.save(path+'/motion_table',motion_table)
                corrupted_data,original_data = manipulate_kspace_columns(motion_table,filepath,imagesize)
                corrupted_img, original_img = image_reconstruction(corrupted_data,original_data)
                write_image(path+'/',corrupted_img)
        
