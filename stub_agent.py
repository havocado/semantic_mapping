import numpy as np
import random
import habitat_sim
import SemMap

# Setting up backeng config
backend_cfg = habitat_sim.SimulatorConfiguration()
# Specifying scene path
backend_cfg.scene_id = (
    "data/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb"
)
backend_cfg.scene_dataset_config_file = (
    "data/hm3d_annotated_example_basis.scene_dataset_config.json"
)

# Setting up depth sensor (Required)
depth_sensor = habitat_sim.CameraSensorSpec()
depth_sensor.uuid = "depth"
depth_sensor.resolution = [512, 512]
depth_sensor.position = 1.5 * habitat_sim.geo.UP
depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

# Setting up rgb sensor
rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "rgb"
rgb_sensor.resolution = [512, 512]
rgb_sensor.position = 1.5 * habitat_sim.geo.UP
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR

# Setting up semantic sensor
sem_sensor = habitat_sim.CameraSensorSpec()
sem_sensor.uuid = "semantic"
sem_sensor.resolution = [512, 512]
sem_sensor.position = 1.5 * habitat_sim.geo.UP
sem_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC

# Setting up agent config
agent_config = habitat_sim.AgentConfiguration()
agent_config.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
    ),
}
agent_config.sensor_specifications = [
    depth_sensor, 
    rgb_sensor, 
    sem_sensor]

# Create sim
sim = habitat_sim.Simulator(
    habitat_sim.Configuration(backend_cfg, [agent_config]))

semantic_mapper = SemMap.SemanticMapper(
  cell_dim_meters=np.array([0.2, 0.2, 0.2]),
  initial_map_size=np.array([50, 50, 30]),
  toggle_resize_map=True)

def _action(sim):
  num_acts = 10
  for act_no in range(num_acts):
    # Decide on action
    action_rand = random.randint(0,100)
    if action_rand <= 60:
        action_code = 0
        obs = sim.step("move_forward")
        print("Frame ", act_no, ": Move forward")
    elif action_rand <= 80:
        action_code = 1
        obs = sim.step("turn_left")
        print("Frame ", act_no, ": Turn left")
    elif action_rand <= 100:
        action_code = 2
        obs = sim.step("turn_right")
        print("Frame ", act_no, ": Turn right")

    # Pass parameters to SemanticMapper.integrate_frame()
    # Resolution is expected to be same for all sensors
    depth = obs['depth']
    semantic = obs['semantic']
    resolution = agent_config.sensor_specifications[0].resolution
    position = position = sim.last_state().position
    quat = sim.last_state().rotation

    semantic_mapper.integrate_frame(depth, semantic, resolution, position, quat)

    # Display
    semantic_mapper.display_topdown(height_min=0.2, height_max=1.5)

_action(sim)