import signal
import sys
import numpy as np
from typing import Dict
import copy

import gym
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError


def interrupt_handler(signum, frame):
    print("Signal handler!!!")
    sys.exit(-2)


def _process_image(img, show_image=True):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(img, "bgr8")
    gray = bridge.imgmsg_to_cv2(img, "mono8")
    _, gray = cv.threshold(gray, 160, 255, cv.THRESH_BINARY)

    H, W, _ = image.shape
    clipped = gray[int(H*2/3):, :]

    cnt, _, _, centroids = cv.connectedComponentsWithStats(clipped)

    if cnt < 3:
        return None

    distances = [x[0] - W/2 for x in centroids]
    left_edge = centroids[distances.index(min(distances))]
    right_edge = centroids[distances.index(max(distances))]

    # print("EDGES:", left_edge, right_edge)

    if len(left_edge) == 0 or len(right_edge) == 0 or (left_edge == right_edge).all():
        return None

    if left_edge[0] < 30 or right_edge[0] > W - 30:
        return None

    goal_x = int((left_edge[0] + right_edge[0]) / 2)
    goal_y = int((left_edge[1] + right_edge[1]) / 2)
    error = goal_x - W / 2

    slope = error * (180.0 / W)

    clipped = cv.cvtColor(clipped, cv.COLOR_GRAY2RGB)
    pt0 = (int(W/2), int(H/3))
    pt1 = (goal_x, goal_y)
    pt2 = (int(W/2), goal_y)

    cv.arrowedLine(clipped, pt0, pt1, (0,255,0), 2, 8, 0, 0.03)
    cv.arrowedLine(clipped, pt0, pt2, (255,0,0), 2, 8, 0, 0.03)

    cv.imwrite("clipped.png", clipped)
    if show_image:
        cv.imshow("clipped", clipped)
        cv.waitKey(1)

    return slope/90.0



class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int,
                 size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )

    def __len__(self) -> int:
        return self.size



class ActionNoise():
    """
    The action noise base class
    """

    def __init__(self):
        super().__init__()

    def reset(self) -> None:
        """
        call end of episode reset for the noise
        """
        pass

    def __call__(self) -> np.ndarray:
        raise NotImplementedError()



class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise

    :param mean: the mean value of the noise
    :param sigma: the scale of the noise (std here)
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray, shape: int):
        self._mu = mean
        self._sigma = sigma
        self._shape = shape
        super().__init__()

    def __call__(self) -> np.ndarray:
        return self.sample()

    def sample(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma, self._shape)


class OUNoise(ActionNoise):
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def __call__(self) -> np.ndarray:
        return self.sample()

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state



"""## Environment
*ActionNormalizer* is an action wrapper class to normalize the action values
ranged in (-1. 1). Thanks to this class, we can make the agent simply select
action values within the zero centered range (-1, 1).
"""

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action
