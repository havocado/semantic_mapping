import os
import habitat_sim
import SEM
import cv2 as _cv2
import numpy as np
import random

import matplotlib as plt

# Setting up backeng config
backend_cfg = habitat_sim.SimulatorConfiguration()
# Specifying scene path
backend_cfg.scene_id = (
  "data/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb"
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

# Create SemMapAgent
initial_position = sim.last_state().position[0:3]
semantic_agent = SEM.SemMapAgent(
    agent_config, 
    initial_position, 
    display_figures=False, 
    save_figures=True)

def _action(sim):
  num_acts = 10
  for act_no in range(num_acts):
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
        print("Frame ", act_no, ": Turn right")

    # Passing parameters to SemMapAgent
    obs["depth"] = obs["depth"][:,:,np.newaxis]
    quat = sim.last_state().rotation
    position = sim.last_state().position[:3]
    semantic_agent.act(obs, quat, position)

_action(sim)

# Save result as video
# Need to set save_figures=True when initializing SemMapAgent,
# otherwise the saved video will be empty
semantic_agent.save_result("result_videos/results.gif")