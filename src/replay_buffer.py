from collections import deque, namedtuple
import random
import numpy as np



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, buffer_size=int(2e6)):
        self.buffer = deque([], maxlen=buffer_size)
        self.idx = 0
    
    def push(self, *args): # s, a, r, s', done
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return [state, action, reward, next_state, done]

    def __len__(self):
        return len(self.buffer)