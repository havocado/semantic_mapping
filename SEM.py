import numpy as np
import matplotlib.pyplot as plt
import imageio
import fog_of_war

class SemMapAgent(object):
  def __init__(self, agent_config, initial_location):
    self.frame_count = 0
    self.resolution = agent_config.sensor_specifications[0].resolution
    
    # Init location info
    self.agent_location = np.array([0,0,0],np.float32)
    self.initial_location = initial_location[0:2]
    self.elevation = initial_location[2] # TODO: Verify!!
    self.camera_height = agent_config.sensor_specifications[0].position[2] # TODO: Verify!!

    # Init map related location info
    self.map_grid_size = 0.2
    self.map_width = 10
    self.map_grid_width = np.round(self.map_width / self.map_grid_size).astype(int)
    self.grid_map = np.zeros([self.map_grid_width, self.map_grid_width])
    self.grid_map.fill(0.5)
    # map_center is index of center of map
    self.map_center = np.array([np.round(self.map_grid_width/2.).astype(int), np.round(self.map_grid_width/2.).astype(int)]) # TODO: Fix this to be middle of drawn map

    # Init plot figure
    self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2)
    self.fig.set_size_inches(13.5, 7)
    self.fig.subplots_adjust(wspace=0, hspace=0)
    plt.ion()

    # Init camera info
    self.res_width = self.resolution[0]
    self.res_height = self.resolution[1]
    self.camera_hfov = agent_config.sensor_specifications[0].hfov
    self.camera_xc = (self.res_width-1.0) / 2
    self.camera_zc = (self.res_height-1.0) / 2
    # TODO: replace 90 with self.camera_hfov
    self.camera_f = (self.res_width / 2.) / np.tan(np.deg2rad(90/ 2.))
    
    self.all_marked_points = np.array([])
    self.all_agent_marks = np.zeros([1,2])

    self.done = False

    # Params for testing
    self.display_test_figs = False
    self.save_test_figs = True

    self.pngs = []

  def act(self, obs, theta, location, done):
    self.frame_count = self.frame_count+1
    depth = obs['depth']
    theta = theta
    self.done = done
    self.update_agent_location(theta, location)
    coords = self.unproject_to_world(depth)
    self.add_to_map(coords)
    self.grid_map = fog_of_war.reveal_fog_of_war(self.grid_map, np.array(self.xy_to_grid_index(self.agent_location[0], self.agent_location[1])), self.agent_location[2])

    if (self.display_test_figs or self.save_test_figs or self.done):
      self.display_map(depth)
    if (self.done and self.save_test_figs):
      self.save_gif()

  def update_agent_location(self, theta, location):
    self.agent_location[0:2] = (location - self.initial_location)
    self.agent_location[2] = theta

  def unproject_to_world(self, depth):
    # Filter out errors
    depth = depth[:,:,0]*1
    depth[depth==0] = np.NaN
    camera_coords = self.depth2camera(depth)
    geocentric_coords = self.camera2geocentric(camera_coords)
    world_coords = self.geocentric2world(geocentric_coords)
    return world_coords
  
  def add_to_map(self, coords):
    # Slice where Y is between -1 and 1.
    sliced_coords = coords[coords[:,:,2]>-1]
    sliced_coords = sliced_coords[sliced_coords[:,2]<1]

    # Check if all coordinates are within grid size.
    grid_indices = self.xy_to_grid_index(sliced_coords[:,0], sliced_coords[:,1])
    while (not(np.min(grid_indices[0])>=0 and np.max(grid_indices[0])<=self.map_grid_width and np.min(grid_indices[1])>=0 and np.max(grid_indices[1])<=self.map_grid_width)):
      self.resize_map()
      grid_indices = self.xy_to_grid_index(sliced_coords[:,0], sliced_coords[:,1])

    # Add new data to map
    self.grid_map[tuple(grid_indices)] = 1
    # Add agent location info
    self.all_agent_marks = np.concatenate((self.all_agent_marks, self.agent_location[0:2].reshape(1,2)), axis=0)
    
  def display_map(self, depth):
    self.ax1.imshow(1-(self.grid_map), cmap='gray', vmin=0, vmax=1)
    self.ax0.imshow(depth/10.0, cmap='gray')
    self.ax1.plot(self.x_to_grid_index(self.all_agent_marks[:,0]), self.y_to_grid_index(self.all_agent_marks[:,1]), linestyle='-', color='green')
    self.ax1.scatter(self.x_to_grid_index(self.agent_location[0]), self.y_to_grid_index(self.agent_location[1]), marker='*', color='red')
    self.ax0.axis('off')
    self.ax1.axis('off')

    # Saving
    if (self.save_test_figs):
      self.fig.savefig("results/results_"+str(self.frame_count)+".jpg")
      self.pngs.append("results/results_"+str(self.frame_count)+".jpg")
    
    plt.show()
    if (self.display_test_figs):
      plt.waitforbuttonpress()
    plt.cla() # TODO: replace with something faster
    
  def save_gif(self):
    images = []
    for png in self.pngs:
      img = imageio.imread(png)
      print("appending ", png)
      images.append(img)
    imageio.mimsave('temp.gif', images, fps=5)

  def depth2camera(self, depth):
    # TODO: Rewrite and organize.
    x, z = np.meshgrid(np.arange(depth.shape[-1]), np.arange(depth.shape[-2]-1, -1, -1))
    X = (x-self.camera_xc) * depth / self.camera_f
    Z = (z-self.camera_zc) * depth / self.camera_f
    XYZ = np.concatenate((X[...,np.newaxis], depth[...,np.newaxis], Z[...,np.newaxis]), axis=X.ndim)
    return XYZ

    """fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, depth, Z, marker='.')
    ax.set_label('X')
    ax.set_label('Y = depth')
    ax.set_label('Z')
    plt.show()"""

  def camera2geocentric(self, camera_coords):
    # Building rotation matrix
    R = self.get_rotation_matrix([1.,0.,0.], angle=np.deg2rad(self.elevation))

    geocentric_coords = np.matmul(camera_coords.reshape(-1,3), R.T).reshape(camera_coords.shape)
    geocentric_coords[...,2] = geocentric_coords[...,2] + self.camera_height
    return geocentric_coords

  def geocentric2world(self, geocentric_coords):
    R = self.get_rotation_matrix([0.,0.,1.], angle=self.agent_location[2])
    world_coord = np.matmul(geocentric_coords.reshape(-1,3), R.T).reshape(geocentric_coords.shape)
    world_coord[:,:,0] = world_coord[:,:,0] + self.agent_location[1]
    world_coord[:,:,1] = world_coord[:,:,1] + self.agent_location[0]
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.azim = 0
    ax.elev = 0
    ax.scatter(world_coord[:,:,0], world_coord[:,:,1], world_coord[:,:,2], marker='.', s=0.05)
    ax.set_label('X')
    ax.set_label('Y = depth')
    ax.set_label('Z')
    plt.show()"""
    return world_coord

  def resize_map(self):
    # Resize map by *1.2
    resize_scale = 1.2
    # Recalculate parameters
    new_map_width = np.round(self.map_width * resize_scale).astype(int)
    new_map_grid_width = np.round(new_map_width / self.map_grid_size).astype(int)
    new_grid_map = np.zeros([new_map_grid_width, new_map_grid_width])
    
    # Copy data to new grid map
    new_grid_map.fill(0.5)
    new_map_center = np.array([np.round(new_map_grid_width/2.).astype(int), np.round(new_map_grid_width/2.).astype(int)])
    old_offset = new_map_center[0:2]-self.map_center[0:2]
    new_grid_map[old_offset[0]:old_offset[0]+self.map_grid_width, old_offset[1]:old_offset[1]+self.map_grid_width] = self.grid_map

    # Update map
    self.map_width = new_map_width
    self.map_grid_width = new_map_grid_width
    self.map_center = new_map_center
    self.grid_map = new_grid_map

  def get_rotation_matrix(self, ax_, angle):
    # TODO: Fix everything here
    ANGLE_EPS = 0.001
    ax = self.normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
      S_hat = np.array(
          [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
          dtype=np.float32)
      R = np.eye(3) + np.sin(angle)*S_hat + \
          (1-np.cos(angle))*(np.linalg.matrix_power(S_hat, 2))
    else:
      R = np.eye(3)
    return R


  def normalize(self, v):
    return v / np.linalg.norm(v)

  # TODO: clean up below 2 functions
  def xy_to_grid_index(self, x, y):
    return [np.round(x/self.map_grid_size).astype(int)+self.map_center[0], np.round(y/self.map_grid_size).astype(int)+self.map_center[1]]
  def x_to_grid_index(self, x):
    return np.round(x/self.map_grid_size).astype(int)+self.map_center[0]
  def y_to_grid_index(self, y):
    return np.round(y/self.map_grid_size).astype(int)+self.map_center[1]
