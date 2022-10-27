![results](https://user-images.githubusercontent.com/47484587/197925919-4f9f9780-fe68-4567-8538-108844eaec81.gif)

## How to run

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
  cell_dim_meters: np.ndarray = np.ndarray([0.2,0.2,0.2])
  map_width_meter: int = 10,
)
```
Initializes the Semantic map agent object.

**Parameters**
- [Optional] `cell_dim_meters`: Cell widths of each grids. Default:
- [Optional] `map_width_meter`: Initial size of the map in meters. Default: 10.
  - When larger maps are needed, the agent will automatically resize the map, so there is no need to specify this parameter unless (1) it is taking to long to resize the map or (2) smaller map is needed.

### 2. Adding frames
```python
SemMap.add_frame(
  depth: np.ndarray,
  semantic: np.ndarray,
  position: np.ndarray,
  quat: np.ndarray, 
)
```
Calling add_frame() for observation frames will add information to the map.

**Parameters**
- `depth`: depth frame obtained from a depth sensor.
- `quat`: rotation of the camera.
- `position`: position of the camera.

## Getting data from SemMap

### 3. Get grid map data
```python
SemMap.get_gridmap()
```
Returns
- `grid_map`: np.ndarray (2d)
  - Each entry in 2d array is an integer code representing the location on the grid map.
  - 0: Unobserved
  - 1: Empty
  - 2: Not empty (This will later be replaced with semantic segmentation code)
- `cell_dim_meters`: represents how many grids are in 1 meter.

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


## Credits

slam.py, depth_utils.py, fmm_planner.py, rotation_utils.py are from Map plan baseline - https://github.com/s-gupta/map-plan-baseline with small modificaitons on slam.py

depth_agent.py is from https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/stereo_agent.py, with modifications.

fog_of_war from https://github.com/facebookresearch/habitat-lab/blob/cc9e4a9adc06950f7511745d681726efc0d1a85a/habitat-lab/habitat/utils/visualizations/fog_of_war.py
