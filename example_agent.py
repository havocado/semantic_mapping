import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import csv
import habitat_sim
import SemMap
import imageio
import semantic_segmentation

# Options
save_figure_as_gif = True
wait_after_display = False

# Setting up backeng config
backend_cfg = habitat_sim.SimulatorConfiguration()
# Specifying scene path
backend_cfg.scene_id = (
    "data/hm3d/00366-fxbzYAGkrtm/fxbzYAGkrtm.basis.glb"
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

# Setting up rgb sensor (Optional)
rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "rgb"
rgb_sensor.resolution = [512, 512]
rgb_sensor.position = 1.5 * habitat_sim.geo.UP
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR

# Setting up semantic sensor (Can be replaced semantic segmentation)
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

# Create sim object
sim = habitat_sim.Simulator(
    habitat_sim.Configuration(backend_cfg, [agent_config]))

# Create SemanticMapper object
semantic_mapper = SemMap.SemanticMapper(
  cell_dim_meters=np.array([0.2, 0.2, 0.2]),
  initial_map_size=np.array([100, 100, 100]),
  toggle_resize_map=True)

# Setup semantic segmentation
sem_seg = semantic_segmentation.SemanticSegmentation()

# Display related.
# Here ax0 will show rgb image
# ax1 will show semantic frame
# and ax2 will show topdown view of semantic map
fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
ax0.axis('off')
ax1.axis('off')
ax2.axis('off')

# Read csv file to get color map
# delimeter: comma
cmap_container = {}
with open('data/hm3d/00366-fxbzYAGkrtm/fxbzYAGkrtm.semantic.txt', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # Skip first line as it contains header
    next(reader)
    contents_csv = list(reader)
    # +2 for extra 2 codes for unknown and unannotated
    cmap_container = np.ndarray([len(contents_csv)+3, 4])
    # First 3 codes hardcoded here
    # -2: unknown, handled by 
    cmap_container[0] = [1., 1., 1., 1.] # color for -2, empty
    cmap_container[1] = [0., 0., 0., 1.] # color for -1, unannotated
    cmap_container[2] = [0.5, 0.5, 0.5, 1.] # color for 0, unobserved
    for row in contents_csv:
        print("row ", row)
        if (len(row) < 4):
            continue
        cmap_container[int(row[0])+2] = [
            int(row[1][0:2], 16)/255.0,
            int(row[1][2:4], 16)/255.0,
            int(row[1][4:6], 16)/255.0,
            1.0]
cmap_semantic = matplotlib.colors.ListedColormap(cmap_container.reshape(-1,4))
# Normalization function that maps 0, 1, 2, ... to -2, -1, 0 ...
# This is required for displaying semantic map
norm_semantic = matplotlib.colors.Normalize(vmin=-2, vmax=len(cmap_container)-2)

result_imgs = []

def _action(sim):
  num_acts = 30
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

    # Get semantic map
    # Pass parameters to SemanticMapper.integrate_frame() here
    # Resolution is expected to be same for all sensors (depth, rgb, semantic)
    depth = obs['depth']

    #semantic = obs['semantic']
    rgb = obs['rgb']
    # cut out the alpha channel
    rgb = rgb[:, :, 0:3]

    # Run semantic segmentation
    seg_result = sem_seg.run(im=rgb)
    print("seg_result: ", seg_result['instances'].pred_masks)
    if (seg_result['instances'].pred_masks.shape[0] == 0):
        # No object detected
        semantic = np.ndarray((512, 512), dtype=np.int32)
        semantic.fill(-1)
    else:
        semantic = np.argmax(seg_result['instances'].pred_masks, axis=0)
        semantic = semantic.cpu().detach().numpy()
        # TODO: Handle configuration here

    resolution = agent_config.sensor_specifications[0].resolution
    position = position = sim.last_state().position
    quat = sim.last_state().rotation

    semantic_mapper.integrate_frame(depth, semantic, resolution, position, quat)

    # Display
    ax0.imshow(rgb)
    ax1.imshow(semantic, norm=norm_semantic, cmap=cmap_semantic)
    semantic_mapper.display_topdown(ax2, height_min=0.2, height_max=1.5, norm=norm_semantic, color=cmap_semantic)
    ax2.axis('off')

    if (save_figure_as_gif):
        filename = "results/results_"+str(act_no)+".jpg"
        fig.savefig(filename)
        result_imgs.append(filename)

    if (wait_after_display):
        plt.waitforbuttonpress()
    plt.cla()

    # TODO: Remove
    
    if (np.mod(act_no, 3) == 0):
        filename = "results/rgb_results_"+str(act_no)+".jpg"
        plt.imsave(filename, obs['rgb'])
    

_action(sim)

# Save gif
if (save_figure_as_gif):
    images = []
    for imgname in result_imgs:
      img = imageio.imread(imgname)
      print("appending ", imgname)
      images.append(img)
    imageio.mimsave("result_videos/result.gif", images, fps=5)