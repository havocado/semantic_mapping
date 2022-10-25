![skokloster_castle_resize](https://user-images.githubusercontent.com/47484587/196324233-92771932-eb4d-489a-8da0-66f9df7d92a4.gif)

## How to run

#### 1. Installing habitat (skip if already installed)

Conda setup: https://github.com/facebookresearch/habitat-sim#preparing-conda-env

Installing habitat: https://github.com/facebookresearch/habitat-sim#conda-install-habitat-sim

Downloading test data: https://github.com/facebookresearch/habitat-sim#testing

#### 2. Test data path

- Download path should match example_agent.py line 58:

```python
# Specifying scene path
backend_cfg.scene_id = (
  "data/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb"
)
```

Or this path in example_agent.py should be modified to match the actual data path.

#### 3. Starting the program

```
$ python3 example_agent.py
```
No argument required.

## Parameters

#### 1. Initialization
```python
SEM.SemMapAgent(
  agent_config: habitat_sim.AgentConfiguration, 
  initial_location: np.ndarray, 
  display_figures: bool = False, 
  save_figures: bool = True, 
  grid_per_meter: int = 5, 
  map_width_meter: int = 10,
  slice_range_below: float = -1.0, # 0 or negative
  slice_range_above: float = 0.0, # 0 or positive
)
```
Initializes the Semantic map agent object.

**Parameters**
- `agent_config`: habitat_sim.AgentConfiguration object.
- `initial_location`: Initial agent location, obtained by `sim.last_state().position`. This will also be the center of the generated map.
- [Optional] `display_figures`: Option to display figures. Default: False
  - If `True`, the program will wait for the user to click on the generated figure after it displays each figures. 
  - If `False`, the figures may be still generated depending on `save_figures`, but the figures will disappear immediately instead of waiting for the use to click.
- [Optional] `save_figures`: Option to save figures as results. Default: True
  - To save results, `save_result()` also has to be called at the end of the program
  - Note: Saving figures will significantly slow down the program, as this function will save image files for each frame and generate a video.
- [Optional] `grid_per_meter`: Number of grid per meters (integer, at least 1) Default: 5
  - Rounding errors may apply.
- [Optional] `map_width_meter`: Initial size of the map in meters. Default: 10.
  - When larger maps are needed, the agent will automatically resize the map, so there is no need to specify this parameter unless (1) it is taking to long to resize the map or (2) smaller map is needed.
- [Optional] `slice_range_below`: The vertical display range for 2D maps, relative to the camera. 0 or negative.
- [Optional] `slice_range_above`: The vertical display range for 2D maps, relative to the camera. 0 or positive.

#### 2. Adding frames
```python
SemMapAgent.act(
  obs, # observation returned from sim.step
  quat: np.ndarray, 
  position: np.ndarray,
)
```
Calling act() after each frames will add information to the map.

**Parameters**
- `obs`: observation returned from `sim.step`
- `quat`: rotation returned from `sim.last_state().rotation`
- `position`: position returned from `sim.last_state().position`

#### 3. (Optional) Save result as video
```python
SemMapAgent.save_result(filename)
```
Saves the results to destination.

**Parameters**
- `filename`: String, destination filename

## Credits

slam.py, depth_utils.py, fmm_planner.py, rotation_utils.py are from Map plan baseline - https://github.com/s-gupta/map-plan-baseline with small modificaitons on slam.py

depth_agent.py is from https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/stereo_agent.py, with modifications.

fog_of_war from https://github.com/facebookresearch/habitat-lab/blob/cc9e4a9adc06950f7511745d681726efc0d1a85a/habitat-lab/habitat/utils/visualizations/fog_of_war.py
