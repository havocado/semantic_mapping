## How to run

#### 1. Installing habitat (skip if already installed)

Conda setup: https://github.com/facebookresearch/habitat-sim#preparing-conda-env

Installing habitat: https://github.com/facebookresearch/habitat-sim#conda-install-habitat-sim

Downloading test data: https://github.com/facebookresearch/habitat-sim#testing

#### 2. Test data path

- Download path should match depth_agent.py line 58:

```python
# Speficying path
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = (
    "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)
```

Or this path in example_agent.py should be modified to match the actual data path.

#### 3. Starting the program

```
$ python3 example_agent.py
```
No argument required.

## Parameters

.


## Credits

slam.py, depth_utils.py, fmm_planner.py, rotation_utils.py are from Map plan baseline - https://github.com/s-gupta/map-plan-baseline with small modificaitons on slam.py

depth_agent.py is from https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/stereo_agent.py, with modifications.

fog_of_war from https://github.com/facebookresearch/habitat-lab/blob/cc9e4a9adc06950f7511745d681726efc0d1a85a/habitat-lab/habitat/utils/visualizations/fog_of_war.py
