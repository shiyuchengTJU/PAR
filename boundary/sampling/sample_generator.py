from collections import deque
import threading
import randomgen

from .perlin import create_perlin_noise, calc_fade


class SampleGenerator:
    """
    Multi-threaded util that runs in the background and precalculates noise samples. To be used in with statement!

    Not a python "generator". Sorry about the name.
    """

    def __init__(self, shape, n_threads=1, queue_lengths=40, pixels=64):
        self.shape = shape
        self.n_threads = n_threads
        self.queue_lengths = queue_lengths
        self.queue_normal = deque()
        self.queue_perlin = deque()

        self.perlin_fade = calc_fade(shape[0])
        self.perlin_color = True

        # Manually reimplemented Queue locking to cover 2 deque instead of 1.
        self.lock = threading.Lock()
        self.cv_not_full = threading.Condition(self.lock)           # Producer condition
        self.cv_not_empty = threading.Condition(self.lock)          # Consumer condition
        self.is_running = True
        self.threads = []
        self.pixels = pixels
        for thread_id in range(n_threads):
            thread = threading.Thread(target=self._thread_fun, args=(thread_id,))
            thread.start()
            self.threads.append(thread)

    def __enter__(self):
        # Threads already started at __init__
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop all the threads.
        for thread in self.threads:
            thread.do_run = False

        self.is_running = False

        # Stop producers.
        with self.cv_not_full:
            self.cv_not_full.notify_all()

        # Stop consumers, if any.
        # Usually, Consumers are from the same thread that created and destroys this object. So there will be none waiting at this point.
        # If, however, there is a consumer from a background thread, they could be still waiting, and then they will receive an
        #  InterruptedError here.
        with self.cv_not_empty:
            self.cv_not_empty.notify_all()

        for thread in self.threads:
            thread.join()
        print("SampleGenerator: all threads stopped.")

    def _thread_fun(self, thread_id):
        # create a thread-specifc RNG
        rng = randomgen.RandomGenerator(randomgen.Xoroshiro128(seed=20 + thread_id))
        rnd_normal = None
        rnd_perlin = None

        t = threading.currentThread()
        while getattr(t, 'do_run', True):

            # Prepare one of each sampling patterns
            if rnd_normal is None:
                rnd_normal = rng.standard_normal(size=self.shape, dtype='float64')
            if rnd_perlin is None:
                rnd_perlin = create_perlin_noise(color=self.perlin_color, batch_size=1, normalize=True, precalc_fade=self.perlin_fade, pixels=self.pixels)[0]

            # Lock and put them into the queues.
            with self.cv_not_full:
                if len(self.queue_normal) >= self.queue_lengths and len(self.queue_perlin) >= self.queue_lengths:
                    self.cv_not_full.wait()

                # Fill one or both queues.
                if len(self.queue_normal) < self.queue_lengths:
                    self.queue_normal.append(rnd_normal)
                    rnd_normal = None
                if len(self.queue_perlin) < self.queue_lengths:
                    self.queue_perlin.append(rnd_perlin)
                    rnd_perlin = None

                self.cv_not_empty.notify_all()

    def get_normal(self):
        """
        Returns a std-normal noise vector - not normalized!
        """
        with self.cv_not_empty:
            while len(self.queue_normal) == 0:
                self.cv_not_empty.wait()
                if not self.is_running:
                    raise InterruptedError("Trying to consume an item, but the SampleGenerator was already shut down!")

            retval = self.queue_normal.popleft()
            self.cv_not_full.notify()
        return retval

    def get_perlin(self):
        """
        Returns a perlin noise vector, normalized to L2=1
        """
        with self.cv_not_empty:
            while len(self.queue_perlin) == 0:
                self.cv_not_empty.wait()
                if not self.is_running:
                    raise InterruptedError("Trying to consume an item, but the SampleGenerator was already shut down!")

            retval = self.queue_perlin.popleft()
            self.cv_not_full.notify()
        return retval