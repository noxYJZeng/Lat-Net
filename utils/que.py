import time
import process

class Que:
    def __init__(self, available_gpus=[0, 1]):
        self.pl = []
        self.running_pl = []
        self.available_gpus = available_gpus

    def enque_file(self, file_name):
        with open(file_name, 'r') as f:
            cmd_list = [line.strip() for line in f if line.strip()]
        for cmd in cmd_list:
            self.pl.append(process.Process(cmd.split()))

    def start_next(self, gpu):
        for task in self.pl:
            if task.get_status() == "Not Started":
                task.start(gpu)
                break

    def find_free_gpu(self):
        used_gpus = [task.get_gpu() for task in self.pl if task.get_status() == "Running"]
        return list(set(self.available_gpus) - set(used_gpus))

    def update_pl_status(self):
        for task in self.pl:
            task.update_status()

    def print_que_status(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 40)
        print("QUEUE STATUS")
        print("=" * 40)
        for task in self.pl:
            task.print_info()
        print("=" * 40)

    def start_que_runner(self):
        while True:
            time.sleep(1)
            free_gpus = self.find_free_gpu()
            for gpu in free_gpus:
                self.start_next(gpu)
            self.update_pl_status()
            self.print_que_status()
