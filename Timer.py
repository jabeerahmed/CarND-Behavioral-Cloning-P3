import time

class Timer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs  = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        self.mins  = self.secs / 60.0
        self.hours = self.mins / 60.0
        if self.verbose: print('elapsed time: %f ms' % self.msecs)
        
    def printTime(self):        
        t, u = self.hours, 'hrs'        
        if (self.mins <  60): t, u = self.mins, 'mins'
        if (self.secs < 300): t, u = self.secs, 'secs'        
        if (self.msecs<1000): t, u = self.msecs,'ms'        
        print('---------------------------------------------------------------')
        print('elapsed time: {:5.3f} {}'.format(t, u))
        print('---------------------------------------------------------------')

    def run(func, args={}):
        ret = None
        with Timer() as timer: ret = func(**args)
        timer.printTime()
        return ret