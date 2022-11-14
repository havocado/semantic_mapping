import numpy as np

class SemanticMap(object):
  def __init__(
    self,
    cell_dim_meters,
    initial_map_size
  ):
    print("init() for SemanticMap called")
    self.cell_dim_meters = cell_dim_meters
    self.gridmap = np.ndarray(initial_map_size) # TODO: Fix

  def add_to_map(self):
    print("add_to_map() called")

# Wrapper class for Semantic Map
class SemanticMapper(object):
  def __init__(
    self,
    cell_dim_meters: np.ndarray = np.ndarray([0.2, 0.2, 0.2]),
    initial_map_size: np.ndarray = np.ndarray([50, 50, 15]),
    toggle_resize_map : bool = True
  ):
    self.cell_dim_meters = cell_dim_meters
    self.initial_map_size = initial_map_size
    self.toggle_resize_map = toggle_resize_map

    #
    self.frame_count = 0

    # Initialize Map object
    self.map = SemanticMap(cell_dim_meters, initial_map_size)

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
    
    # 1. Set agent location
    self._update_agent_location(position)

    # 2. Unproject to world using the location
    coords = self._unproject_to_world(depth, quat)

    # 3. Add coord to map
    self._add_to_map(coords)


  def get_gridmap(self):
    print("get_gridmap() called")

  def display_topdown(
    height_min: int,
    height_max: int,
  ):
    print("display_topdown() called")

  def save_topdown(
    self,
    height_min: int,
    height_max: int,
    filename: str,
  ):
    print("save_topdown() called")

  def _unproject_to_world(self, depth):
    print("_unproject_to_world() called")