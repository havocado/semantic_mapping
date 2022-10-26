import habitat_sim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import quaternion
import imageio
import fog_of_war

class SemMapAgent(object):
  def __init__(
    self, 
    agent_config: habitat_sim.AgentConfiguration, 
    initial_location: np.ndarray, 
    display_figures: bool = False, 
    save_figures: bool = True, 
    grid_per_meter: int = 5, 
    map_width_meter: int = 10,
    slice_range_below: float = -1.0, # 0 or negative
    slice_range_above: float = 0.0, # 0 or positive
  ):
    self.frame_count = 0
    self.resolution = agent_config.sensor_specifications[0].resolution
    
    # Init location info
    initial_location = self.habitat_position_to_2d(initial_location)
    self.agent_location = np.array([0,0,0],np.float32)
    self.initial_location = initial_location[0:2]
    self.elevation = initial_location[2]
    self.camera_height = agent_config.sensor_specifications[0].position[2]

    # Init map related location info
    # Gridmap: 0(unobserved)/1(empty)/other code (segmentation)
    self.grid_per_meter = grid_per_meter
    self.map_width_meter = map_width_meter
    self.map_grid_width = np.round(self.map_width_meter*self.grid_per_meter).astype(int)
    self.grid_map = np.zeros([self.map_grid_width, self.map_grid_width])
    self.grid_map.fill(0)
    # map_center is index of center of map
    self.map_center = np.array([
      np.round(self.map_grid_width/2.).astype(int), 
      np.round(self.map_grid_width/2.).astype(int)])

    # Init plot figure
    self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(1, 3)
    self.fig.set_size_inches(21, 7)
    self.fig.subplots_adjust(wspace=0, hspace=0)
    plt.ion()

    self.sem_cmap = matplotlib.colors.ListedColormap(np.random.rand(1000,3))
    self.sem_cmap.colors[0] = np.array([0.5,0.5,0.5]) # unobserved
    self.sem_cmap.colors[1] = np.array([1.,1.,1.]) # empty

    # Init camera info
    self.res_width = self.resolution[0]
    self.res_height = self.resolution[1]
    self.camera_hfov = agent_config.sensor_specifications[0].hfov
    self.camera_xc = (self.res_width-1.0) / 2
    self.camera_zc = (self.res_height-1.0) / 2
    # TODO: replace 90 with self.camera_hfov (I don't know how to do it)
    self.camera_f = (self.res_width / 2.) / np.tan(np.deg2rad(90/ 2.))

    # Init 2D map slicing height info
    # 2D map will display heights between: 
    #   [elevation+slice_range_below, elevation+slice_range_above]
    self.slice_range_below = slice_range_below # Should be 0 or negative
    self.slice_range_above = slice_range_above # Should be 0 or positive
    
    # Init agent related info
    self.all_marked_points = np.array([])
    self.all_agent_marks = np.zeros([1,2])

    self.display_test_figs = display_figures
    self.save_test_figs = save_figures

    self.result_imgs = []

  def act(
    self, 
    obs, # observation returned from sim.step
    quat: np.ndarray, 
    position: np.ndarray,
  ) -> None:
    self.frame_count = self.frame_count+1
    depth = obs['depth']
    depth = depth[:,:,np.newaxis]
    rgb = obs['rgb']
    semantic = obs['semantic']
    theta = self.quat_to_topdown_theta(quat)
    location = self.habitat_position_to_2d(position)
    self._update_agent_location(theta, location)
    coords = self._unproject_to_world(depth)
    self._add_to_map(coords)
    self.grid_map = fog_of_war.reveal_fog_of_war(
      self.grid_map, 
      np.array(self._xy_to_grid_index(
        self.agent_location[0], self.agent_location[1])), 
      self.agent_location[2])

    # Display figures.
    # Note: figures are not actually displayed until plt.show()
    if (self.display_test_figs or self.save_test_figs):
      self._display_sensor_output_1(rgb)
      self._display_sensor_output_2(semantic)
      self._display_map()
    plt.show()

    # Save figures for each frame. Required for creating gif.
    if (self.save_test_figs):
      filename = "results/results_"+str(self.frame_count)+".jpg"
      self.fig.savefig(filename)
      self.result_imgs.append(filename)

    # Wait for button press
    if (self.display_test_figs):
      plt.waitforbuttonpress()
    plt.cla() # TODO: replace with something faster

  def save_result(self, filename):
    # Save gif on last frame
    if (self.save_test_figs):
      self._save_gif(filename)

  def _update_agent_location(self, theta, location):
    self.agent_location[0:2] = (location[:2] - self.initial_location)
    self.agent_location[2] = theta

  def _unproject_to_world(self, depth):
    # Filter out errors
    depth = depth[:,:,0]*1
    depth[depth==0] = np.NaN
    camera_coords = self._depth2camera(depth)
    geocentric_coords = self._camera2geocentric(camera_coords)
    world_coords = self._geocentric2world(geocentric_coords)
    return world_coords
  
  def _add_to_map(self, coords):
    # 2D map will display heights between: 
    #   [elevation+slice_range_below, elevation+slice_range_above]
    sliced_coords = coords[coords[:,:,2]>self.slice_range_below]
    sliced_coords = sliced_coords[sliced_coords[:,2]<self.slice_range_above]

    # Check if all coordinates are within grid size.
    # If any coordinate go out of grid size, resize the map.
    grid_indices = self._xy_to_grid_index(sliced_coords[:,0], sliced_coords[:,1])
    if (len(grid_indices[0])==0): return
    while (
      not(np.min(grid_indices[0])>=0 
      and np.max(grid_indices[0])<=self.map_grid_width 
      and np.min(grid_indices[1])>=0 
      and np.max(grid_indices[1])<=self.map_grid_width)
    ):
      self._resize_map()
      grid_indices = self._xy_to_grid_index(sliced_coords[:,0], sliced_coords[:,1])

    # Add new data to map
    self.grid_map[tuple(grid_indices)] = 2
    # Add agent location info
    self.all_agent_marks = np.concatenate((
      self.all_agent_marks, 
      self.agent_location[0:2].reshape(1,2)), 
      axis=0)
    
  def _display_sensor_output_1(self, output, is_depth=False):
    if (is_depth):
      self.ax0.imshow(output/10.0, cmap='gray')
    else:
      self.ax0.imshow(output)
    self.ax0.axis('off')

  def _display_sensor_output_2(self, output, is_depth=False):
    if (is_depth):
      self.ax1.imshow(output/10.0, cmap='gray')
    else:
      self.ax1.imshow(output, cmap=self.sem_cmap, vmin=0, vmax=999)
    self.ax1.axis('off')

  def _display_map(self):
    self.ax2.imshow(self.grid_map, cmap=self.sem_cmap, vmin=0, vmax=999)
    self.ax2.plot(
      self._x_to_grid_index(self.all_agent_marks[:,0]), 
      self._y_to_grid_index(self.all_agent_marks[:,1]), 
      linestyle='-', 
      color='green')
    self.ax2.scatter(
      self._x_to_grid_index(self.agent_location[0]), 
      self._y_to_grid_index(self.agent_location[1]), 
      marker='*', 
      color='red')
    self.ax2.axis('off')
    
  def _save_gif(self, filename):
    images = []
    for imgname in self.result_imgs:
      img = imageio.imread(imgname)
      print("appending ", imgname)
      images.append(img)
    imageio.mimsave(filename, images, fps=5)

  def _depth2camera(self, depth):
    # TODO: Rewrite and organize.
    x, z = np.meshgrid(
      np.arange(depth.shape[-1]), np.arange(depth.shape[-2]-1, -1, -1))
    X = (x-self.camera_xc) * depth / self.camera_f
    Z = (z-self.camera_zc) * depth / self.camera_f
    XYZ = np.concatenate(
      (X[...,np.newaxis], depth[...,np.newaxis], Z[...,np.newaxis]), 
      axis=X.ndim)
    return XYZ

  def _camera2geocentric(self, camera_coords):
    # Building rotation matrix
    R = self._get_rotation_matrix([1.,0.,0.], angle=np.deg2rad(self.elevation))

    geocentric_coords = np.matmul(
      camera_coords.reshape(-1,3), R.T).reshape(camera_coords.shape)
    geocentric_coords[...,2] = geocentric_coords[...,2] + self.camera_height
    return geocentric_coords

  def _geocentric2world(self, geocentric_coords):
    R = self._get_rotation_matrix([0.,0.,1.], angle=self.agent_location[2])
    world_coord = np.matmul(
      geocentric_coords.reshape(-1,3), R.T).reshape(geocentric_coords.shape)
    world_coord[:,:,0] = world_coord[:,:,0] + self.agent_location[1]
    world_coord[:,:,1] = world_coord[:,:,1] + self.agent_location[0]
    return world_coord

  def _resize_map(self):
    # Resize map by *1.2
    resize_scale = 1.2
    # Recalculate parameters
    new_map_width = np.round(self.map_width_meter * resize_scale).astype(int)
    new_map_grid_width = np.round(new_map_width * self.grid_per_meter).astype(int)
    new_grid_map = np.zeros([new_map_grid_width, new_map_grid_width])
    
    # Copy data to new grid map
    new_grid_map.fill(0)
    new_map_center = np.array([
      np.round(new_map_grid_width/2.).astype(int), 
      np.round(new_map_grid_width/2.).astype(int)])
    old_offset = new_map_center[0:2]-self.map_center[0:2]
    new_grid_map[
      old_offset[0]:old_offset[0]+self.map_grid_width, 
      old_offset[1]:old_offset[1]+self.map_grid_width
    ] = self.grid_map

    # Update map
    self.map_width_meter = new_map_width
    self.map_grid_width = new_map_grid_width
    self.map_center = new_map_center
    self.grid_map = new_grid_map

  def _get_rotation_matrix(self, ax_, angle):
    # TODO: Fix everything here
    ANGLE_EPS = 0.001
    ax = self._normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
      S_hat = np.array(
          [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
          dtype=np.float32)
      R = np.eye(3) + np.sin(angle)*S_hat + \
          (1-np.cos(angle))*(np.linalg.matrix_power(S_hat, 2))
    else:
      R = np.eye(3)
    return R

  def habitat_position_to_2d(self, position):
    # Convert 3d habitat coord to 2d coord and elevation
    # Position: sim.last_state().position[0:3]
    location = np.zeros(3)
    location[0] = position[2] * (-1)
    location[1] = position[0]
    location[2] = position[1] # elevation
    return location

  def quat_to_topdown_theta(self, quat):
    theta = quaternion.as_rotation_vector(quat)
    theta = theta[1]
    return theta

  def _normalize(self, v):
    return v / np.linalg.norm(v)

  # TODO: clean up below 2 functions
  def _xy_to_grid_index(self, x, y):
    return [
      np.round(x*self.grid_per_meter).astype(int)+self.map_center[0], 
      np.round(y*self.grid_per_meter).astype(int)+self.map_center[1]]
  def _x_to_grid_index(self, x):
    return np.round(x*self.grid_per_meter).astype(int)+self.map_center[0]
  def _y_to_grid_index(self, y):
    return np.round(y*self.grid_per_meter).astype(int)+self.map_center[1]
