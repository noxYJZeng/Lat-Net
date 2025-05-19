
import os
import time
import subprocess
from termcolor import colored

class Process:
    def __init__(self, cmd):
        self.cmd = cmd if isinstance(cmd, list) else cmd.split()
        self.status = "Not Started"
        self.gpu = -1
        self.process = None
        self.return_status = "NONE"
        self.run_time = 0
        self.pid = None
        self.start_time = None

    def start(self, gpu=0):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env
        )
        self.pid = self.process.pid
        self.status = "Running"
        self.start_time = time.time()
        self.gpu = gpu

    def update_status(self):
        if self.status == "Running":
            self.run_time = time.time() - self.start_time
            ret = self.process.poll()
            if ret is not None:
                self.status = "Finished"
                self.return_status = "SUCCESS" if ret == 0 else "FAIL"

    def get_pid(self):
        return self.pid

    def get_status(self):
        return self.status

    def get_gpu(self):
        return self.gpu

    def print_info(self):
        print(colored(f"CMD: {' '.join(self.cmd)}", 'blue'))
        print(colored(f"Status: {self.status}", 'blue'))
        print(colored("Return Status: ", 'blue') + 
              colored(self.return_status, 'green' if self.return_status == 'SUCCESS' else 
                                            'red' if self.return_status == 'FAIL' else 
                                            'yellow'))
        print(colored(f"Run Time: {self.run_time:.2f} seconds", 'blue'))
