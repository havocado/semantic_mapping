import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import quaternion

# Grid map description
# -2 - empty
# -1 - obstacle, unannotated
# 0 - unobserved
# >=1 - semantic segmentation id
# Saved value: +2 to above.
class SemanticMap(object):
  def __init__(
    self,
    cell_dim_meters,
    initial_map_size
  ):
    # Save map dimension information
    self.cell_dim_meters = cell_dim_meters
    self.initial_map_size = initial_map_size

    # Construct grid array
    self.gridmap = np.ndarray(initial_map_size) # TODO: Fix
    self.gridmap.fill(0)

    # _init_on_first_frame will take care of other initializations
    self.is_empty = True

  # Takes an array of n points and adds to the map.
  # Input:  world_coords (n*3)
  #         semantic_code (n*1)
  def add_points_to_map(
    self,
    world_coords: np.ndarray,
    semantic_code: np.ndarray
  ):
    if (self.is_empty):
      self._init_on_first_frame(world_coords)
      self.is_empty = False

    grid_ind = self._world_coord_to_grid_index(world_coords).astype(int)

    # TODO: Add error correction
    # TODO: Add resizing
    for i in range(grid_ind.shape[0]):
      # check if out of bound
      if (grid_ind[i,0] < 0 or grid_ind[i,0] >= self.gridmap.shape[0] or
          grid_ind[i,1] < 0 or grid_ind[i,1] >= self.gridmap.shape[1] or
          grid_ind[i,2] < 0 or grid_ind[i,2] >= self.gridmap.shape[2]):
        #print("Out of bound: ", grid_ind[i,:])
        continue

      #print("semantic_code[i]: ", semantic_code[i])
      if semantic_code[i] < 1:
        self.gridmap[grid_ind[i,0],grid_ind[i,1],grid_ind[i,2]] = -1
      else:
        self.gridmap[grid_ind[i,0],grid_ind[i,1],grid_ind[i,2]] = semantic_code[i]
    
# Decide the map position on first frame
  def _init_on_first_frame(
    self,
    world_coords: np.ndarray,
  ):
    bbox_min = np.nanmin(world_coords, 0)
    bbox_max = np.nanmax(world_coords, 0)
    bbox_center = np.round((bbox_max - bbox_min)/2)
    self.map_center_world_coord = bbox_center
    self.map_center_grid_index = np.round(self.initial_map_size/2)
    self.map_min_world_coord = (self.map_center_world_coord 
        - np.multiply(self.map_center_grid_index, self.cell_dim_meters))
    self.map_max_world_coord = (self.map_min_world_coord 
        + np.multiply(self.initial_map_size, self.cell_dim_meters))

  def _world_coord_to_grid_index(self, world_coords):
    filtered_coords = world_coords[~np.isnan(world_coords).any(axis=1)]
    relative_coord = filtered_coords - np.transpose(self.map_min_world_coord[:,None])
    result_ind = np.round(relative_coord/np.transpose(self.cell_dim_meters[:,None])).astype(int)
    return result_ind

# Wrapper class for Semantic Map
class SemanticMapper(object):
  def __init__(
    self,
    cell_dim_meters: np.ndarray = np.array([0.2, 0.2, 0.2]),
    initial_map_size: np.ndarray = np.array([50, 50, 15]),
    toggle_resize_map : bool = True
  ):
    self.cell_dim_meters = cell_dim_meters
    self.initial_map_size = initial_map_size
    self.toggle_resize_map = toggle_resize_map

    # Init Frame counter
    self.frame_count = 0

    # Init Map
    self.map = SemanticMap(cell_dim_meters, initial_map_size)

    # Init color map
    

  def integrate_frame(
    self,
    depth: np.ndarray, # Depth frame
    semantic: np.ndarray, # Semantic frame
    resolution: np.ndarray,
    position: np.ndarray,
    quat: np.ndarray, 
    hfov: int = 90,
  ):
    print("integrate_frame() called")
    self.frame_count = self.frame_count + 1

    semantic[semantic<1] = 0
    
    # 1. Set agent location
    self._update_agent_location(position, quat)

    # 2. Unproject to world using the location
    world_coords = self._unproject_to_world(depth, resolution, hfov, position, quat)

    # 3. Add coord to map
    self._add_to_map(world_coords, semantic)


  def get_gridmap(self):
    print("get_gridmap() called")

  def display_topdown(
    self,
    axis, # plt.axis
    height_min: int,
    height_max: int,
    norm: matplotlib.colors.Normalize,
    color: matplotlib.colors.ListedColormap
  ):
    
    """
    gridmap = self.map.gridmap # TODO: Fix this later
    displaymap = np.ndarray(gridmap.shape[0], gridmap.shape[1])
    height_top_ind = height_max.
    """
    # STUB
    gridmap = self.map.gridmap # TODO: Fix this later    
    height_stub = np.round(gridmap.shape[2]/3).astype(int)
    displaymap = gridmap[:,:,height_stub]
    axis.imshow(displaymap, norm=norm, cmap=color)


  def save_topdown(
    self,
    height_min: int,
    height_max: int,
    filename: str,
  ):
    print("save_topdown() called")

  def _update_agent_location(self, position, quat):
    print("_update_agent_location() called")

  def _unproject_to_world(self, depth, resolution, hfov, position, quat):
    print("_unproject_to_world() called")
    depth = depth[:,:,np.newaxis]
    depth = self._filter_depth_errors(depth)
    camera_coords = self._depth2camera(depth, resolution, hfov)
    geocentric_coords = self._camera2geocentric(camera_coords, position)
    world_coords = self._geocentric2world(geocentric_coords, position, quat)
    return world_coords

  def _filter_depth_errors(self, depth):
    depth = depth[:,:,0]*1
    depth[depth==0] = np.NaN
    return depth

  def _depth2camera(self, depth, resolution, hfov):
    # Credit: Modified based on https://github.com/s-gupta/map-plan-baseline
    res_width = resolution[0]
    res_height = resolution[1]
    camera_xc = (res_width-1.0) / 2
    camera_zc = (res_height-1.0) / 2
    camera_f = (res_width / 2.) / np.tan(np.deg2rad(hfov/ 2.))

    x, z = np.meshgrid(
      np.arange(depth.shape[-1]), np.arange(depth.shape[-2]-1, -1, -1))
    X = (x-camera_xc) * depth / camera_f
    Z = (z-camera_zc) * depth / camera_f
    XYZ = np.concatenate(
      (X[...,np.newaxis], depth[...,np.newaxis], Z[...,np.newaxis]), 
      axis=X.ndim)
    return XYZ

  def _camera2geocentric(self, camera_coords, position):
    # Credit: Modified based on https://github.com/s-gupta/map-plan-baseline
    R = self._get_rotation_matrix([1.,0.,0.], angle=np.deg2rad(position[2]))
    geocentric_coords = np.matmul(
      camera_coords.reshape(-1,3), R.T).reshape(camera_coords.shape)
    geocentric_coords[...,2] = geocentric_coords[...,2] + position[2]
    return geocentric_coords

  def _geocentric2world(self, geocentric_coords, position, quat):
    # Credit: Modified based on https://github.com/s-gupta/map-plan-baseline
    theta = self._quat_to_topdown_theta(quat)
    R = self._get_rotation_matrix([0.,0.,1.], angle=theta)
    world_coord = np.matmul(
      geocentric_coords.reshape(-1,3), R.T).reshape(geocentric_coords.shape)
    world_coord[:,:,0] = world_coord[:,:,0] + position[0]
    world_coord[:,:,1] = world_coord[:,:,1] + position[1]
    return world_coord

  def _get_rotation_matrix(self, ax_, angle):
    # Credit: Modified based on https://github.com/s-gupta/map-plan-baseline
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

  def _quat_to_topdown_theta(self, quat):
    theta = quaternion.as_rotation_vector(quat)
    theta = theta[1]
    return theta

  def _normalize(self, v):
    return v / np.linalg.norm(v)

  def _add_to_map(self, world_coords, semantic_codes):
    world_coords = world_coords.reshape(-1, 3)
    semantic_codes = semantic_codes.reshape(-1, 1)
    self.map.add_points_to_map(world_coords, semantic_codes)