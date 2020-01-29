# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:09:58 2020

@author: jayach01
"""

import numpy as np
import glob
import os
from scipy import fftpack,ndimage
import cmath
from scipy.ndimage import rotate
import SimpleITK as sitk
from matplotlib import pyplot as plt
from math import acos, atan2, cos, pi, sin
from numpy.linalg import norm
from nibabel import quaternions
import math
from skimage.measure import compare_ssim as ssim
import random


def load_image(filepath):
    input_img = sitk.ReadImage(filepath)
    input_array = sitk.GetArrayFromImage(input_img)
    return input_img, input_array

def affine_translate(transform, x_translation, y_translation,z_translation): 
    new_transform = sitk.AffineTransform(transform)
    new_transform.SetTranslation((x_translation, y_translation,z_translation))
    return new_transform

def affine_rotate(transform, rotation_matrix,dimension=3):
    new_transform = sitk.AffineTransform(transform)
    matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
    new_matrix = np.dot(rotation_matrix,matrix)
    new_transform.SetMatrix(new_matrix.ravel())
    return new_transform

def create_transform(table,z,k):
        shift = table[z,k,0]
        angle = (table[z,k,1])
        axis  = table[z,k,2]
        rot_matrix = quaternions.angle_axis2mat(angle,axis)
        affine_transform = sitk.AffineTransform(3)
        affine_transform = affine_translate(affine_transform, shift[0], shift[1], shift[2])
        combined_transform = affine_rotate(affine_transform, rot_matrix)
        return combined_transform

def affine_transformation(input_img, transform_matrix):
    center=input_img.TransformContinuousIndexToPhysicalPoint((input_img.GetSize()[0]/2,input_img.GetSize()[1]/2,input_img.GetSize()[2]/2))
    transform_matrix.SetCenter(center)
    transformed_image = sitk.Resample(input_img,input_img.GetSize(),transform_matrix,sitk.sitkLinear,input_img.GetOrigin(),input_img.GetSpacing(), input_img.GetDirection())
    return transformed_image



  
def manipulate_kspace_columns(table,filepath,imagesize):
    input_img, input_array = load_image(filepath) # Load image and array
    
    print("Generating K-Space of original image")
    print('-'*30)
    
    original = np.fft.fftn(input_array,axes=(-2,-1))
    original = np.fft.fftshift(original,axes=(-2, -1))
    copy1 = np.fft.fftn(input_array,axes=(-2, -1))
    copy1 = np.fft.fftshift(copy1,axes=(-2, -1))
    print("K-Space Generated")    

    print("Image transformations and K-Space manipulation....")

    for z in range(np.shape(table)[0]):
        for k in range(np.shape(table)[1]):
            shift = table[z,k,0]

            if np.sum(abs(shift[0]+shift[1]+shift[2]))!=0:
                #print(z,k)
                transform = create_transform(table,z,k)
                tf_img = affine_transformation(input_img, transform)
                tf_array = sitk.GetArrayFromImage(tf_img)

                coil1_dist_array =  tf_array*np.ones((imagesize[1],imagesize[0]))
                coil1_dist_kspace = np.fft.fft2(coil1_dist_array[z])
                coil1_dist_kspace = np.fft.fftshift(coil1_dist_kspace,axes=(-2, -1))
                copy1[z,:,k] = coil1_dist_kspace[:,k] # Substitute Original K-space lines 
    
    
        if z % 10 == 0:
            print('Done: {0}/{1} Slices'.format(z, np.shape(input_array)[0]))
    print("K-Space manipulation complete")
    return copy1, original       




def image_reconstruction(corrupted_k_space, original_k_space):
    img_c1=np.zeros((np.shape(corrupted_k_space)))
    original_img=np.zeros((np.shape(original_k_space)))

    for i in range(np.shape(corrupted_k_space)[0]):
        coil1_inv =  np.fft.ifft2(corrupted_k_space[i])
        coil1_inv = np.abs(coil1_inv)
        img_c1[i] = coil1_inv

    for i in range(np.shape(original_k_space)[0]):
        coil1_inv =  np.fft.ifft2(original_k_space[i])
        coil1_inv = np.abs(coil1_inv)
        original_img[i] = coil1_inv
    return  img_c1, original_img

def write_image(folder, image):
    coil1_img = sitk.GetImageFromArray(image)

    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkUInt16)
    corrupted_img = castFilter.Execute(coil1_img)
    sitk.WriteImage(corrupted_img, folder+"corrupted_img.dcm")
    
def generate_normal_motion_ixi(imagesize):
    motion_table = np.array([np.zeros(3),np.zeros(1),np.zeros(3)])
    trajectory = [[motion_table],]*imagesize[2]*imagesize[0]
    trajectory = np.array(trajectory)
    trajectory = np.reshape(trajectory,(imagesize[2],imagesize[0],3))
    #trajectory = trajectory*0
    trajectory[:,:,2] = trajectory[:,:,2]+1
    print(np.shape(trajectory))
    for i in range(imagesize[2]):
        num = list(range(0,130)) + list(range(180,300)) 
        motion_events = np.random.randint(2,high=6)
        k_space_lines = random.sample(num,motion_events) # Select random k-space lines for manipulation of raw-data
        for row in k_space_lines:
           shift = np.array([np.random.normal(loc=0,scale=1),np.random.normal(loc=0,scale=1),0])
           rot = np.random.normal(loc=0, scale=np.pi/120)
           axes =np.array([0,0,1])
          
           trajectory[i,row,0] = shift
           trajectory[i,row,1] = rot
           trajectory[i,row,2] = axes
                
    #table = np.reshape(table,(imagesize[2],imagesize[0],3))
    return trajectory
    
        

    