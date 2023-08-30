# Repository description

This is a repository for my USRA term project on 3D reconstruction and semantic mapping.

The goal of the project was to implement an egocentric mapping working on AI Habitat, the interactive 3D simulation for training robots/agents on a virtual 3D setting. While the virtual 3D setting provides advantages on training (in the way that the robot never breaks by falling off the stairs, or that agents can move a thousand step per second), using a virtual simulator involves its own challenges such as the use of large 3D data, or, even configuring the projects. In fact, working with the configuration was where most of the struggle was on this project, as an undergraduate student with little experience. 

This repository successfully runs with the ground truth semantic segmentation marked on the top-down 2D map. Although the intention was to extend the project to semantic segmentation on 3D simulation, I decided to pursue a career outside of machine learning while working on the project.

![results](https://user-images.githubusercontent.com/47484587/197925919-4f9f9780-fe68-4567-8538-108844eaec81.gif)

## How to run the test code

### 1. Installing habitat (skip if already installed)

**Setup**
- Conda setup: https://github.com/facebookresearch/habitat-sim#preparing-conda-env
- Installing habitat: https://github.com/facebookresearch/habitat-sim#conda-install-habitat-sim

**Data**
- Download test data: https://github.com/facebookresearch/habitat-sim#testing
- Or use any test scenes.

### 2. Test data path

- Download path should match example_agent.py:

```python
# Specifying scene path
backend_cfg.scene_id = (
  "data/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb"
)
```

Or this path in example_agent.py should be modified to match the actual data path.

### 3. Starting the program

```
$ python3 example_agent.py
```
No argument required.

## Parameters

### 1. Initialization
```python
SEM.SemMap(
  cell_dim_meters: np.ndarray = np.ndarray([0.2, 0.2, 0.2]),
  initial_map_size: np.ndarray = np.ndarray([50, 50, 15]),
  toggle_resize_map : bool = True
)
```
Initializes the Semantic map agent object.

**Parameters**

- [Optional] `cell_dim_meters`: Cell widths of each grids (meters). Default: [0.2,0.2,0.2]
- [Optional] `initial_map_size`: Initial map size in number of cells for each dimension. Default: [50,10,15]
- [Optional] `toggle_resize_map`: Whether to resize map when needed. Default: True
  - If True, the map is resized when any observation occurs outside of map.
  - If False, observations outside of map will be ignored.                                                                                                                                                                                                                                                                                                                                        

### 2. Adding frames
```python
SemMap.integrate_frame(
  depth: np.ndarray,
  semantic: np.ndarray,
  resolution: np.ndarray,
  position: np.ndarray,
  quat: np.ndarray, 
  hfov: int = 90,
)
```
Calling integrate_frame() for observation frames will add information to the map.

**Parameters**
- `depth`: Depth frame obtained from a depth sensor.
- `semantic`: Semantic segmentation obtained from a semantic sensor.
- `position`: Position of the camera.
- `resolution`: Resolution of the frames. This has to be same for both depth and semantic frames.
- `quat`: Rotation of the camera.
- `hfov`: Field of view. Default: 90

## Getting data from SemMap

### 3. Get grid map data
```python
SemMap.get_gridmap()
```
Returns
- `grid_map`: np.ndarray (3d)
  - Each entry in 3d array is an integer code representing the location on the grid map.
  - 0: Unobserved
  - 1: Empty
  - 2: Not empty (This will later be replaced with semantic segmentation code)
- `cell_dim_meters`: Cell widths of each grids (meters)
- `top_left_above`: Coordinate of the top-left-above end (first cell) of the map.
- `bottom-right_below`: Coordinate of the bottom-right-below end (last cell) of the map.

### 4. Display 2D topdown map
```python
SemMap.display_topdown(
  height_min: int,
  height_max: int,
)
```
Displays a top-down map for the specified height range. Each grid will be included in the top-down map only if the center of the grid is within the height range.

**Parameters**
- `height_min`: int, minimum height to display
- `height_max`: int, maximum height to display

### 5. Save 2D topdown map
```python
SemMap.save_topdown(
  height_min: int,
  height_max: int,
  filename: string
)
```
Saves a top-down map for the specified height range as an image file. Each grid will be included in the top-down map only if the center of the grid is within the height range.

**Parameters**
- `height_min`: int, minimum height to display
- `height_max`: int, maximum height to display
- `filename`: String, name of the file. Has to be unique for each call.

### 6. Save a video of topdown maps
```python
SemMap.save_video(
  filename: string
)
```
Generates a video of saved topdown maps.

**Parameters**
- `filename`: String, name of the file

## 3D Map
- Display for 3D map is currently not implemented for performance reasons.

## Error handling
- Currently there is no error handling for 3D reconstruction. Instead, SemMap assumes all the depth frames are correct.
- Error correction for semantic segmentation is not implemented yet, but will be implemented in the future.


## Credits

slam.py, depth_utils.py, fmm_planner.py, rotation_utils.py are from Map plan baseline - https://github.com/s-gupta/map-plan-baseline with small modificaitons on slam.py

depth_agent.py is from https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/stereo_agent.py, with modifications.

fog_of_war from https://github.com/facebookresearch/habitat-lab/blob/cc9e4a9adc06950f7511745d681726efc0d1a85a/habitat-lab/habitat/utils/visualizations/fog_of_war.py
