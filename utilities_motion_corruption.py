# -*- coding: utf-8 -*-
"""
@author: chandrakanth
"""

import numpy as np
import SimpleITK as sitk
from nibabel import quaternions
import random


def load_image(filepath):
    """ Loads an 3D image
    Args:
        param1: path to the image file
        
    Returns:
            The image volume, corresponding numpy array   
    """
    input_img = sitk.ReadImage(filepath)
    input_array = sitk.GetArrayFromImage(input_img)
    return input_img, input_array

def affine_translate(transform, x_translation, y_translation,z_translation): 
    """ Creates a transform for 3D translation
    
    Args:
        param1: 3D transform object
        param2: translation in x-axis
        param3: translation in y-axis
        param4: translation in z-axis
        
    Returns:
            Affine translation object for 3D translation
    """
    new_transform = sitk.AffineTransform(transform)
    new_transform.SetTranslation((x_translation, y_translation,z_translation))
    return new_transform

def affine_rotate(transform, rotation_matrix,dimension=3):
    """ Creates a transform object for 3D rotation, combines 3D translation and rotation transforms
    
    Args:
        param1: Affine translation object for 3D translation
        param2: rotation_matrix for 3D rotation
        param3: dimension of transformation
        
    Returns:                  
        Transformation object which combines 3D translation and rotation
     """
     
    matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
    new_matrix = np.dot(rotation_matrix,matrix)
    transform.SetMatrix(new_matrix.ravel())
    return transform

def create_transform(table,slice,row):
    """ Iterates through the motion data table and generates transformation matrices
    
    Args:
        param1: motion_trajectory_data (numpy array)
        param2: slice number
        param3: row number (k-space line)
    
    Returns:
            transformation object corresponding to a 4x4 transformation matrix
    """
    shift = table[slice,row,0]
    angle = (table[slice,row,1])
    axis  = table[slice,row,2]
    rot_matrix = quaternions.angle_axis2mat(angle,axis)
    affine_transform = affine_translate(sitk.AffineTransform(3), shift[0], shift[1], shift[2])
    combined_transform = affine_rotate(affine_transform, rot_matrix)
    return combined_transform

def affine_transformation(input_img, transform_matrix):
    """ Performs affine(rigid) transformation on the input image volume
    
    Args:
        param1: input image volume
        param2: transformation object corresponding to a 4x4 transformation matrix
        
    Returns:
            Transformed image
    
    """
    center=input_img.TransformContinuousIndexToPhysicalPoint((input_img.GetSize()[0]/2,input_img.GetSize()[1]/2,input_img.GetSize()[2]/2))
    transform_matrix.SetCenter(center)
    transformed_image = sitk.Resample(input_img,input_img.GetSize(),transform_matrix,sitk.sitkLinear,input_img.GetOrigin(),input_img.GetSpacing(), input_img.GetDirection())
    return transformed_image



  
def manipulate_kspace_columns(table,filepath,imagesize):
    """ Generates a k-space corresponding to a motion corrupted acquisition by of merging the k-space lines
        from the transfromed image with that of the original uncorrupted image
        
    Args:
        param1: motion trajectory data/motion table (numpy array)
        param2: file path of the image volume
        param3: size of the input image
    
    Returns:
            motion corrupted 3D k-space
        
    """
    
    input_img, input_array = load_image(filepath) # Load image and array
    
    print("Generating K-Space of original image")
    print('-'*30)
    
    img_fft = np.fft.fftn(input_array,axes=(-2,-1))
    img_fft = np.fft.fftshift(img_fft,axes=(-2, -1))
    print("K-Space Generated")    

    print("Image transformations and K-Space manipulation....")

    for slice in range(np.shape(table)[0]):
        for row in range(np.shape(table)[1]):
            shift = table[slice,row,0]

            if np.sum(abs(shift[0]+shift[1]+shift[2]))!=0:
                transform = create_transform(table,slice,row)
                transformed_img = affine_transformation(input_img, transform)
                transformed_array = sitk.GetArrayFromImage(transformed_img)

                coil1_dist_array =  transformed_array*np.ones((imagesize[1],imagesize[0]))
                coil1_dist_kspace = np.fft.fft2(coil1_dist_array[slice])
                coil1_dist_kspace = np.fft.fftshift(coil1_dist_kspace,axes=(-2, -1))
                img_fft[slice,:,row] = coil1_dist_kspace[:,row] # Substitute Original K-space lines 
    
    
        if slice % 10 == 0:
            print('Done: {0}/{1} Slices'.format(slice, np.shape(input_array)[0]))
    print("K-Space manipulation complete")
    return img_fft       




def image_reconstruction(corrupted_k_space):
    """Reconstructs motion corrupted 3D image from corrupted k-space
    
    Args:
        param1: 3D corrupted k-space
    
    Returns:
            Motion corrupted 3D image
    """
    img_array=np.zeros((np.shape(corrupted_k_space)))
    for i in range(np.shape(corrupted_k_space)[0]):
        img =  np.fft.ifft2(corrupted_k_space[i])
        img = np.abs(img)
        img_array[i] = img
    return  img_array

def write_image(folder, image):
    """ Writes image to directory
    
    Args:
        param1: path to image folder
        param2: image 
        
    """
    coil1_img = sitk.GetImageFromArray(image)
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkUInt16)
    corrupted_img = castFilter.Execute(coil1_img)
    sitk.WriteImage(corrupted_img, folder+"/corrupted_image.dcm")
    
def generate_motion_trajectory(imagesize, x_shift=4, y_shift=4, z_shift=0, rotation=np.pi/60 ):
    """ Generates a 3D random motion trajectory, each row specifies the transformation parameters
        for each k-space line
    
    Args:
        param1: image size(tuple)
        param2: maximum shift in x-axis in mm
        param3: maximum shift in y-axis in mm
        param3: maximum shift in z-axis in mm
        param4: maximum angle of rotation 
        
    Returns:
            Numpy array, motion trajectory data
    """
    motion_table = np.array([np.zeros(3),np.zeros(1),np.zeros(3)]) # creating a template for entering parmeter values for affine transformation
    trajectory = [[motion_table],]*imagesize[2]*imagesize[0]
    trajectory = np.array(trajectory)
    trajectory = np.reshape(trajectory,(imagesize[2],imagesize[0],3))
    
    trajectory[:,:,2] = trajectory[:,:,2]+1 # Adding 1 to avoid creation of invalid transformation 
    for slice in range(imagesize[2]):
        num = list(range(0,int(imagesize[0]/2-10))) + list(range(int(imagesize[0]/2+10),imagesize[0])) 
        motion_events = np.random.randint(2,high=6)
        k_space_lines = random.sample(num,motion_events) # Select random k-space lines for manipulation of raw-data
        for row in k_space_lines:
           shift = np.array([np.random.normal(loc=0,scale=x_shift),np.random.normal(loc=0,scale=y_shift),z_shift])
           rot = np.random.normal(loc=0, scale=rotation)
           axes =np.array([0,0,1])
          
           trajectory[slice,row,0] = shift
           trajectory[slice,row,1] = rot
           trajectory[slice,row,2] = axes
                
    return trajectory
    
        

    