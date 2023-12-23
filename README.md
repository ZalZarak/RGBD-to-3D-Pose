# RGBD-to-3D-Pose

## What is this repository about?

This repository extracts 3D-coordinates of joint positions of a humanoid using 
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and a [IntelRealSense Depth-Camera](https://www.intelrealsense.com/).
With those joints it simulates a humanoid having spheres and cylinders as limbs in [PyBullet](https://pybullet.org/).

It is designed detect humans for collision avoidance for robots (proof of concept). 
It is designed to work in this project: [IR-DRL](https://github.com/ignc-research/IR-DRL).   

It may work for your purpose too. Feel free to use it. 

## Installation
1. - Install OpenPose as described here: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md.    
   - Make sure to enable BUILD_PYTHON in CMake.
   - Enable custom python executable (e.g. conda env):
        - Add Entry in CMake Gui: Name: PYTHON_EXECUTABLE, Type: file path
        - Set it to your python executable
   - This is easier on Linux. For Windows, you can consider following tips: [openpose_installation_tips.txt](openpose_installation_tips.txt)
   - Install all dependencies
     - ```pip install -r requirements.txt```
   - Depending on your OS, provide the correct path files in your config under OpenPoseHandler.
   - Optional: Set custom flags for OpenPose in the config, e.g. greater precision
        - Here are all possible flags, not all of them are useful for this purpose: [openpose_flags.md](openpose_flags.md)
   - Configurate

## Modules

### [Main](src/main.py)

Runs the code as configured. Either runs Perceptor or Simulator or starts the debug tools.

### [Config](src/config.py)

Configures the entire pipeline. For the explanation of the different settings visit [config_explanation.yaml](config/config_explanation.yaml)

### [Perceptor](src/perceptor.py)

Here, the main logic is implemented. It receives a RGBD-Stream from a IntelRealSense camera or from a 
playback file. It pushes the RGB-Frame to OpenPose and extracts the 3D-coordinates of the joints using
the depth information. It translates and rotates those 3d-coordinates as configured and forwards them
to Simulator to render them into a simple humanoid shape in real-time in PyBullet. It can save the 
stream or 3D-coordinates of the joints with timestamps to a file for latter replay. If you want to 
replay a stream (.bag file), make sure to enter the same resolution and fps as it was recorded with.

It can either run as main process and start Simulator as subprocess or the other way around.

Now, some joints might be occluded by objects in front of them but still recognized by OpenPose, which
would produce inaccurate 3D-Positions (with the depth of the object instead of the joint).   
The program performs the following to increase accuracy:
- It validates each joint through the connections defined in the config under _Perceptor/connections_hr_.
  A joint is marked as valid, if it validates through at least one of its connections
- Each connection has a defined accepted length range under _Perceptor/lengths_hr_ and a maximal depth
  deviation under _Perceptor/depth_hr_.   
  - The length is valid if the distance between the joints is within the range. The depth deviation is 
  valid if the z-length (abs(joint1[z]-joint2[z])) is not higher than the maximum depth deviation.   
  - The connection is valid if length __and__ depth are valid.
- You can define a color range under Perceptor/color_range and define applicable joints under Perceptor/color_validation_joints_hr.
  - If the pixel's color, where the joint is, is within the color range, it is marked as valid without
  validation. This is useful if you know the color of certain joints, f.e. if you wear green gloves.
  - The color range must be set precisely and barely mark pixels as within the range which do not belong
  to the associated limb. It is very different in different lighting situations and is influenced by the time of day, 
  season and cloud cover, among other things.
- To increase robustness, you can define a search area for each joint under _Perceptor/search_areas_hr_.
  - The search area is defined by a tuple (d, s). The search area is a square with the length 2d+1
    centered at the pixel in question. s determines the skip/density, e.g. only each s'th pixel will be in the
    search area.
  - The program will search in the search area of unvalidated joints to find an accepted color or an
    accepted length and depth. If it succeeds, the joint will be marked as valid and have the old
    x,y-coordinates and the new z-coordinate.
  - The search area shouldn't be too big and only around the joint.
- Use the debug mode to visualize and adjust lengths, depths, color ranges and search areas.

Unvalidated joints will be set to (0,0,0).


### [Simulator](src/simulator.py)

The Simulator simulates the humanoid in 3D with simple spheres and cylinders. It either receives them
from Perceptor in real-time or from a file.    
It can be integrated into other programs. Collision detection works between the humanoid and other
objects. The collision humanoid collision filter group mask is (-1, 1, 0).   
Current joint positions are saved in self.joints.
Pybullet IDs of currently valid/invalid limbs are saved in self.limb_list_valid and self.limb_list_invalid.

- The limbs are defined in the config under Simulator/limbs, Simulator/radii and Simulator/lengths.
- If a limb has a single joint, it becomes a sphere, if it has two joints, it becomes a cylinder. 
- A sphere's or cylinder's radius is defined under radii. A cylinder's length is defined under lengths.


### [OpenPoseHandler](src/openpose_handler.py)

Extract human joint positions from RGBD-Frames. For best results it should receive an upright frame, 
this can be set in the config under Perceptor/flip.

Insert the paths of your OpenPose installation under OpenPoseHandler in the config.

You can add and change the parameters but not all of them are useful. Find them here: 
[File](openpose_flags.md) and [Link](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/demo_advanced.md).


### [Debug](src/debug.py)

The debug mode offers visualization for the lengths and depths between joints, the color range and 
search areas and a GUI to set those. 

Different modes can be configured in the config under Debug. It takes all other necessary 
configuration from Perceptor. 

For length and depth it will give you some statistics about meausured length/depth per joint. This is
just meant as help. 

### [Helper](src/helper.py)

Some helper methods

### [Custom 3D joint transforms](src/custom_3d_joint_transforms.py)

Here you can define your own transforms for extracted 3d joints.
- functions must take and return np.ndarray([25, 3]).
- This will be applied after 3d joints are extracted and validated.
- But before any other transforms, e.g. the joints are in camera space, not in "real" space.
- So, neither flips (90° camera rotations), nor translations nor rotation were applied at this point.
- undetected joints are (0,0,0).
- unvalidated joints have z=0, e.g. (x,y,0).
- This should stay this way, e.g. (0,0,0) should stay (0,0,0) and (x,y,0) should stay (x,y,0).
- A joint with (x,y,z) with z!=0 must not become (a,b,0).

### [Config](src/config.py)

__Change the used config here!__


# Configuration

Go to [config_explanation](config/config_explanation.yaml) for a detailed description of each parameter.

### How to configure length ranges, maximum depth?

- Run Debug in the length/depth mode. Move naturally, as you would do during your task.     
  At the end, you get some statistics about length/depth of your joint connections. This is just
  meant as help and orientation.   
  __Known Bug__: In this mode, OpenPose works poorly. Set flip=0 for a better performance.
- The camera depth sensor isn't very precise. You will need to set your ranges in a way, that will 
  accept connections in normal positions, where the depth value is well distinguishable.
  - Example: If you point your hand at the camera, OpenPose is likely to detect your elbow joint, 
    but the depth won't be precise enough.
  - Great differences in depth over few pixels are poorly detected.
  - Look at the values and try to estimate a good lower and upper bound. You can use custom_connections
    to display only some connections.
  - For each connection:
    - Take an object and cover one joint. Move the object closer to the camera, while covering only one joint.
      Set the upper value to the value when the imprecision of the depth becomes "unacceptable" 
      (if it's lower than before).
    - Take an object and cover both joints. Move the object closer to the camera, while covering both joints. 
      Set the lower value to the value when the imprecision of the depth becomes "unacceptable" 
      (if it's higher than before).
  - Try balancing precision against robustness.


### How to configure color range?

- Run Debug in color mode.
- Set the color range in a way that your colored limb (f.e. green gloves) is properly visible but the
  rest is black. Move your limb to different angles as this influences the perceived color strongly.
- If you have to decide: Better make your limb less visible than make other stuff better visible.  
  You still need to balance that.
- __IMPORTANT__: You need to do this everytime anew, as the color is strongly affected by different
  lightning, e.g. time, sun angle, cloud coverage etc.


### How to configure search area?

- Run Debug in search area mode.
- Adjust the deviation so that it is approximately the size of the joint:
  - For example: The deviation for the torso should be bigger than for the elbow.
- Adjust the skip/density depending on how crucial the joint is and how big the search area of it is:
  - Hands are usually important, so the density is higher.
  - Torso has a high deviation, so density can be lower.
  - Balance it to your needs.
- High densities and high deviations may increase lag (not measured).

### Why does the color stream look so weird?

The reason for that can be found in the alignment configuration: By default (2) the color stream alings 
to the depth stream. Since the depth-stream has noise/shadows, those parts are black too. 
Luckily, OpenPose doesn't mind those.   

With my camera model, I also get big black bars at the edges of the color-stream. The depth-stream has a
wider viewing angle, so that the color stream gets adjusted onto it, e.g. it appears smaller, but it keeps
its viewing angle.

The alternative is to align depth to color (1) and color will look fine. The problem is that, the 
deprojection from 2d to 3d (e.g. the 3d joint positions) will become very imprecise in 
the x- and y- directions. This is a knows issue.   
This [forum post](https://github.com/IntelRealSense/librealsense/issues/10438) suggests, that the issue is
fixed if color is aligned to depth.

The issue might not persist with other camera models (especially with newer models). I only noticed this 
error when I switched my camera.

You can use the "view coordinates" debug mode to show coordinates of some joints. Compare them to the
coordinates obtained in IntelRealSense Viewer. There enable both depth and color stream and switch to 3d.
Hover over the joint with the mouse and inspect the coordinates. If both match while aligning depth to color
(1), using this setting should be fine. Otherwise, align color to depth (2). You may also compre the
coordinates for this setting.

### How to generate geometry separately?

This is useful, if you have your own methods to generate, track geometry and don't want to rewrite your code.

1. Set ```build_geometry``` to ```False```
2. For each limb: Generate it. Let ```p_id``` be the pybullet object id of this limb.
3. Set ```limbs_pb``` dict at ```(joint_id, )``` for spheres and ```(joint_id_1, joint_id_2)``` for cylinders
to ```p_id```. You may use ```joint_map``` in the config for that, which is also saved under the same name
in Simulator object. For RWrist and RElbow it would look like 
```simulator.limbs_pb[(simulator.joint_map["RElbow"], simulator.joint_map["RWrist"])] = p_id```.   
__Make very sure, that ```joint_id_1 < joint_id_2```__.   
````lengths, radii, limbs```` from the config are also saved under the same name in simulator object.
4. Set the Collision Filter Group Mask to ```(-1, 1, 0)``` for limbs:   
```p.setCollisionFilterGroupMask(p_id, -1, 1, 0)```