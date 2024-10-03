import os
import random
import multiprocessing
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import mmap
import pickle
import zstandard as zstd
import io
import multiprocessing

class LargeTextDataset(ABC):
  """Abstract class for handling large text datasets."""
  
  def __init__(self, folder_path, saved_rel_path='samples.jsonl'):
    self.folder_path = folder_path
    self.stats_dict_path = os.path.join(folder_path, '.stats_dict.pkl')
    os.makedirs(os.path.join(folder_path, 'outputs'))
    self.saved_path = os.path.join(folder_path, 'outputs', saved_rel_path)

    self.stats_dict = {}
    if os.path.exists(self.stats_dict_path):
      with open(self.stats_dict_path, 'rb') as f:
        self.stats_dict = pickle.load(f)

    self._preprocess_files()
  
  @staticmethod
  def get_file_stream(filepath: str):
      """
      Returns a seekable file-like object for both compressed and uncompressed files.
      """
      if filepath.endswith('.zst'):
          with open(filepath, 'rb') as f:
              dctx = zstd.ZstdDecompressor()
              decompressed_data = dctx.stream_reader(f).read()
              return io.BytesIO(decompressed_data)
      else:
          return open(filepath, 'rb')
        
  @staticmethod
  def _search_newline_pos(filepath: str):
    """Given a text file, find all newline char positions.

    This essentially implements `wc -l` but returns the positions.
    This is useful since once we know all where the `\n` positions
    are, we can then use them *uniformly* sample the lines in the 
    file with O(1) space instead of O(n).

    The peak mem usage of this function is dominated by the length
    of the result;  the file is not loaded into the memory.
    """

    newline_positions = []
    with open(filepath, 'r+b') as f:
      # Memory-map the file, size 0 means the whole file
      with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        file_size = mm.size()
        current_pos = 0
        # Initialize tqdm with total file size
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  desc="Processing") as pbar:
          while True:
            pos = mm.find(b'\n', current_pos)
            if pos == -1:
              # Update the progress bar to the end
              pbar.update(file_size - current_pos)
              break
            newline_positions.append(pos)
            # Calculate the number of bytes processed since last update
            bytes_processed = pos - current_pos + 1  # +1 to move past the '\n'
            pbar.update(bytes_processed)
            current_pos = pos + 1  # Move to the next byte after '\n'

    return newline_positions

  def get_line(self, file_name: str, line_number: int):
      """
      Retrieve a specific line from the file based on byte offsets.
      """
      filepath = os.path.join(self.folder_path, file_name)
      start_offset = self.get_line_start_offset(file_name, line_number)
      end_offset = self.stats_dict[file_name][line_number]
      with self.get_file_stream(filepath) as f:
          f.seek(start_offset)
          if end_offset is not None:
              line = f.read(end_offset - start_offset)
          else:
              line = f.readline()
      return line.decode('utf-8').rstrip('\n')
  
  def _preprocess_files(self):
    """
      Compute or load the dictionary line offsets for each file in the dataset. The dictionary has 
      keys of file_names, values of a list of offsets of newlines. Note that the set of shard paths
      are different from the set of keys of loaded dictionary, if that case, then recompute and save the dictionary. 
    """
    
    # List all files in the folder excluding .pkl files
    all_files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f)) and not f.endswith('.pkl')]

    # Remove entries from stats_dict that are no longer present in the folder
    files_to_remove = [key for key in self.stats_dict.keys() if key not in all_files]
    for file_name in files_to_remove:
        del self.stats_dict[file_name]

    self.total_lines = 0
    self.file_weights, self.file_lengths = {}, {}
    # Process any files that are not in stats_dict
    for file_name in all_files:
      if file_name not in self.stats_dict:
        file_path = os.path.join(self.folder_path, file_name)
        newline_positions = self._search_newline_pos(file_path)
        self.stats_dict[file_name] = newline_positions

      num_lines = len(self.stats_dict[file_name])
      self.file_weights[file_name], self.file_lengths[file_name] = num_lines, num_lines
      self.total_lines += num_lines

    # Update the weights of the files
    for file_name in self.file_weights:
      self.file_weights[file_name] /= self.total_lines
    
    # Save the updated stats_dict to file
    with open(self.stats_dict_path, 'wb') as f:
      pickle.dump(self.stats_dict, f)
  
  def sample_line_with_seed(self, seed):
    """
    Sample a single line from the dataset using a specific seed.
    """
    # Create a new random number generator with the given seed
    rng = random.Random(seed)

    # Sample a file based on the weights
    file_name = rng.choices(
        population=list(self.file_weights.keys()),
        weights=list(self.file_weights.values()),
        k=1
    )[0]

    # Sample a line from the selected file
    num_lines = self.file_lengths[file_name]
    line_number = rng.randint(0, num_lines - 1)

    # Retrieve the line
    return self.get_line(file_name, line_number)
  
  # def sample_line(self, _):
  #   # Sample a file based on the weights
  #   file_name = random.choices(
  #       population=list(self.file_weights.keys()),
  #       weights=list(self.file_weights.values()),
  #       k=1
  #   )[0]

  #   # Sample a line from the selected file
  #   num_lines = self.file_lengths[file_name]
  #   line_number = random.randint(0, num_lines - 1)

  #   # Retrieve the line
  #   return self.get_line(file_name, line_number)
    
  def random_sample(self, num_samples, saved=False, num_workers=None, seed=42):
    """
      Efficiently uniformly sample random lines from the sharded dataset.

      Args:
          num_samples (int): Number of samples to retrieve.

      Returns:
          list of str: A list containing the randomly sampled lines.
    """
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    print(f'Using {num_workers} workers for sampling.')
    
    # Generate a list of seeds, one for each sample
    rng = random.Random(seed)
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_samples)]

    # Use multiprocessing to sample lines in parallel
    with multiprocessing.Pool(num_workers) as pool:
        sampled_lines = pool.map(self.sample_line_with_seed, seeds)

    # Optionally save the sampled lines to a file
    if saved:
        with open(self.saved_path, 'w') as f:
            for line in sampled_lines:
                f.write(line + '\n')

    return sampled_lines

  def get_line_start_offset(self, file_name: str, line_number: int):
    """Returns the starting offset for a specific line number."""
    return 0 if line_number == 0 else self.stats_dict[file_name][line_number - 1] + 1

    
  # def random_sample(self, num_samples, saved=False):
  #   """
  #     Efficiently uniformly sample random lines from the sharded dataset.

  #     Args:
  #         num_samples (int): Number of samples to retrieve.

  #     Returns:
  #         list of str: A list containing the randomly sampled lines.
  #   """
  #   sampled_lines = []

  #   for _ in range(num_samples):
  #       # Sample a file based on the weights
  #       file_name = random.choices(
  #           population=list(self.file_weights.keys()),
  #           weights=list(self.file_weights.values()),
  #           k=1
  #       )[0]

  #       # Sample a line from the selected file
  #       num_lines = self.file_lengths[file_name]
  #       line_number = random.randint(0, num_lines - 1)

  #       # Retrieve the line
  #       sampled_lines.append(self.get_line(file_name, line_number))

  #   # Optionally save the sampled lines to a file
  #   if saved:
  #       with open(self.saved_path, 'w') as f:
  #           for line in sampled_lines:
  #               f.write(line + '\n')

  #   return sampled_lines
  

# Specific dataset classes
class ThePileDataset(LargeTextDataset):
  def load_dataset(self):
    # Implementation if needed
    pass

class DolmaDataset(LargeTextDataset):

  def load_dataset(self):
    # Implementation if needed
    pass


# --------------------------------------------------
### test code below; can delete after code is ready

  # @staticmethod
  # def _process_single_file(file_path):
  #   """
  #       Compute or load line offsets for a single file.

  #       Args:
  #           file_path (str): Path to the text file.
  #           cache_dir (str): Path to the cache directory.

  #       Returns:
  #           list of int: List of byte offsets where each line starts.
  #       """
        

  #   # file_name = os.path.basename(file_path)
  #   # cache_file = os.path.join(cache_dir, f"{file_name}index.npy")

  #   # if os.path.exists(cache_file):
  #   #   # Load offsets from cache
  #   #   offsets = np.load(cache_file).tolist()
  #   # else:
  #     # Compute offsets and save to cache
  #     # offsets = []
  #     # offset = 0
  #     # with open(file_path, 'rb') as f:
  #     #   while True:
  #     #     line = f.readline()
  #     #     if not line:
  #     #       break
  #     #     offsets.append(offset)
  #     #     offset += len(line)
  #     # # Save offsets to cache
  #     # np.save(cache_file, np.array(offsets, dtype=np.int64))
  #   return offsets


# def read_from_pretrain_shard(filepath: str):

#   # Open a text file in read-write mode
#   with open('example.txt', 'r+b') as f:
#     # Memory-map the file, size 0 means the whole file
#     mm = mmap.mmap(f.fileno(), 0)

#     # Move the cursor to the start of the file
#     mm.seek(0)

#     # Read the first 10 bytes
#     print(mm.read(10))  # Adjust the number as needed

#     # Move the cursor to a different position
#     mm.seek(20)

#     # Read the next 10 bytes from that position
#     print(mm.read(10))

#     # # You can also write data
#     # mm.seek(0)
#     # mm.write(b'Hello!')  # Writes "Hello!" at the start

#     # Don't forget to close the mmap object
#     mm.close()
if __name__ == '__main__':
  dataset = LargeTextNewDataset(folder_path='test_data')
  