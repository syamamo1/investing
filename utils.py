import time
from tqdm import tqdm


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def get_time(self):
        if self.end_time is None:
            self.end_time = time.time()

        return self.end_time - self.start_time


class Timers():
    def __init__(self, name='', use_tqdm=False):
        self.timers = {}
        self.name = name
        self.counts = 0
        self.current = None
        self.use_tqdm = use_tqdm

    def start(self, ide=None):
        # Name already in timers
        if ide in self.timers:
            self.timers[ide].start()

        # New timer
        elif ide is not None:
            self.timers[ide] = Timer()
            self.timers[ide].start()

        # New unnamed timer
        elif ide is None:
            self.timers[self.counts] = Timer()
            self.timers[self.counts].start()
            ide = self.counts
            self.counts += 1

        self.current = ide

    def end(self, ide=None):
        # Name passed in
        if ide is not None:
            self.timers[ide].end()

        # No name passed in
        elif ide is None:
            self.timers[self.current].end()

    # Print all timers
    def show(self):
        for ide, timer in self.timers.items():
            message = f'{self.name} | {ide}: {timer.get_time(): .2f}s'

            if self.use_tqdm:
                tqdm.write(message)
            else:
                print(message)