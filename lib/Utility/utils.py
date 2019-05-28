########################
# importing libraries
########################
# system libraries
import sys
import os
import shutil
import subprocess
import torch
import torch.utils.data


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_checkpoint(state, is_best, save_path=None, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


class GPUMem:
    def __init__(self, is_gpu):
        self.is_gpu = is_gpu
        if self.is_gpu:
            self.total_mem = self._get_total_gpu_memory()

    def _get_total_gpu_memory(self):
        total_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.total",
                                             "--format=csv,noheader,nounits"])

        return float(total_mem[0:-1])  # gets rid of "\n" and converts string to float

    def get_mem_util(self):
        if self.is_gpu:
            # Check for memory of GPU ID 0 as this usually is the one with the heaviest use
            free_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"])
            free_mem = float(free_mem[0:-1])    # gets rid of "\n" and converts string to float
            mem_util = 1 - (free_mem / self.total_mem)
        else:
            mem_util = 0
        return mem_util
