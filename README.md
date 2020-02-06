# mri-motion-artifact-simulation
Python framework for simulation of motion artifacts in MR images

This repository introduces a Python framework for simulation of subject motion in MR images. The framework enables simulation of macroscopic
3D motion of the subject with a flexible motion trajectory for cartesian MRI. Rigid body assumption is made for subject motion and only inter-scan motion is simulated.

![MRI Motion Simulation Algorithm](https://github.com/chandrakanth-jp/mri-motion-artifact-simulation/blob/master/images/motion_simulation_algorithm.png)



## Prerequisites

The following libraries are required for the framework:  
 - SimpleITK
 - nibabel
 
 The code should be compatible with Python version 3.5 
 
 ## Installation
 
 cd to the project root directory, activate virtualenv and run
 
```bash
pip install -r requirements.txt
```

