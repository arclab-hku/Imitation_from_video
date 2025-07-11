"""An env wrapper that flattens the observation dictionary to an array."""
import numpy as np
import gym


class ActionScaleWrapper(gym.Env):
  """An env wrapper that flattens the observation dictionary to an array."""
  def __init__(self, gym_env, action_scale, default_pose):
    """Initializes the wrapper."""
    self.action_scale = action_scale
    self.default_pose = default_pose
    self._gym_env = gym_env

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)

  def reset(self, initial_motor_angles=None, reset_duration=0.0):
    return self._gym_env.reset(
        initial_motor_angles=initial_motor_angles,
        reset_duration=reset_duration)

  def step(self, action):
    """Steps the wrapped environment.
    Args:
      action: Numpy array. The input action from an NN agent.
    Returns:
      The tuple containing the flattened observation, the reward, the epsiode
        end indicator.
    """
    new_action = action * self.action_scale + self.default_pose
    return self._gym_env.step(new_action)

  def render(self, mode='human'):
    return self._gym_env.render(mode)
