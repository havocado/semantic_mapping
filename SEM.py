import numpy as np
import matplotlib.pyplot as plt

class SemMapAgent(object):
  def __init__(self, agent_config, initial_location):
    self.resolution = agent_config.sensor_specifications[0].resolution
    
    # Init location info
    self.agent_location = np.array([0,0,0],np.float32)
    self.initial_location = initial_location[0:2]
    self.elevation = initial_location[2] # TODO: Verify!!
    self.camera_height = agent_config.sensor_specifications[0].position[2] # TODO: Verify!!

    # Init map related location info
    self.map_center = np.array([5.0,5.0]); # TODO: Fix this to be middle of drawn map

    # Init camera info
    self.res_width = self.resolution[0]
    self.res_height = self.resolution[1]
    self.camera_hfov = agent_config.sensor_specifications[0].hfov
    self.camera_xc = (self.res_width-1.0) / 2
    self.camera_zc = (self.res_height-1.0) / 2
    # TODO: replace 90 with self.camera_hfov.
    self.camera_f = (self.res_width / 2.) / np.tan(np.deg2rad(90/ 2.))
    
    self.all_marked_points = np.array([])
    self.all_agent_marks = self.map_center.reshape(1,2)

    self.done = False
    self.display_test_figs = False
    ## TODO: Fill this.

  def act(self, obs, theta, location, done):
    depth = obs['depth']
    self.done = done
    self.update_agent_location(theta, location)
    coords = self.unproject_to_world(depth)
    self.add_to_map(coords)

  def update_agent_location(self, theta, location):
    self.agent_location[0:2] = self.map_center + (location - self.initial_location)
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
    
    if (self.all_marked_points.size == 0):
      self.all_marked_points = sliced_coords
    else:
      self.all_marked_points = np.concatenate((self.all_marked_points, sliced_coords), axis=0)

    self.all_agent_marks = np.concatenate((self.all_agent_marks, self.agent_location[0:2].reshape(1,2)), axis=0)
    
    #ax.azim = 0
    #ax.elev = 0
    if (self.display_test_figs or self.done):
      fig = plt.figure()
      plt.scatter(self.all_marked_points[:,0], self.all_marked_points[:,1], marker='.')
      plt.scatter(self.all_agent_marks[:,0], self.all_agent_marks[:,1], marker='*', color='gray')
      plt.scatter(self.agent_location[0], self.agent_location[1], marker='*', color='red')
      plt.show()
    print("---")

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
    # TODO: Fix this part
    R = self.get_rotation_matrix([0.,0.,1.], angle=self.agent_location[2]-np.pi/2.)
    world_coord = np.matmul(geocentric_coords.reshape(-1,3), R.T).reshape(geocentric_coords.shape)
    world_coord[:,:,0] = world_coord[:,:,0] + self.agent_location[0]
    world_coord[:,:,1] = world_coord[:,:,1] + self.agent_location[1]
    
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
