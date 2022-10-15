import os
import habitat_sim
import SEM
import cv2 as _cv2
import numpy as np
import random
import quaternion
import matplotlib as plt

display = False

# Setting up display
if (display):
  _cv2.namedWindow("sensor_depth")

# Setting up backeng config
backend_cfg = habitat_sim.SimulatorConfiguration()
# Specifying scene path
backend_cfg.scene_id = (
  "../map_plan_baseline_modified/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)

# Setting up sensor
depth_sensor = habitat_sim.CameraSensorSpec()
depth_sensor.uuid = "depth"
depth_sensor.resolution = [512, 512]
depth_sensor.position = 1.5 * habitat_sim.geo.UP
depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

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
agent_config.sensor_specifications = [depth_sensor]

sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))

topdown_position = sim.last_state().position[0:3]
location = np.zeros(3)
location[0] = topdown_position[2] * (-1)
location[1] = topdown_position[0]
location[2] = sim.last_state().position[0]

semantic_agent = SEM.SemMapAgent(agent_config, location)


def _action(sim):
  num_acts = 100
  for act_no in range(num_acts):
    action_rand = random.randint(0,100)
    if action_rand <= 60:
        action_code = 0
        obs = sim.step("move_forward")
    elif action_rand <= 80:
        action_code = 1
        obs = sim.step("turn_left")
    elif action_rand <= 100:
        action_code = 2
        obs = sim.step("turn_right")
    print("action_code: ", action_code)

    theta = quaternion.as_rotation_vector(sim.last_state().rotation)
    theta = theta[1]

    topdown_position = sim.last_state().position[0:3]
    location = np.zeros(2)
    location[0] = topdown_position[2] * (-1)
    location[1] = topdown_position[0]
    print("location: ", location)
    print("theta: ", theta)

    # TODO: Get rid of this
    obs["depth"] = obs["depth"][:,:,np.newaxis]

    semantic_agent.act(obs, theta, location, (act_no == num_acts-1))

_action(sim)