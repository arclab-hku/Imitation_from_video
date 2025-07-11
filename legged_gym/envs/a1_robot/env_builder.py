from legged_gym.envs.a1_robot import locomotion_gym_config
from legged_gym.envs.a1_robot import locomotion_gym_env
from legged_gym.envs.a1_robot.env_wrappers import action_scale_wrapper
from legged_gym.envs.a1_robot.sensors import robot_sensors
from legged_gym.envs.a1_robot import a1
from legged_gym.envs.a1_robot import a1_robot


def build_env_isaac(sim_params, default_pose, obs_scales, action_scale,
                    use_real_robot=False,
                    realistic_sim=False):

  gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

  robot_class = a1_robot.A1Robot if use_real_robot else a1.A1
  robot_kwargs = {}

  if use_real_robot or realistic_sim:
    robot_kwargs["reset_func_name"] = "_SafeJointsReset"
    robot_kwargs["velocity_source"] = a1.VelocitySource.IMU_FOOT_CONTACT
  else:
    robot_kwargs["reset_func_name"] = "_PybulletReset"

  # Robot sensors.
  sensors = [
    robot_sensors.ProjectedGravitySensor(),
    robot_sensors.FakeCommandSensor(),
    robot_sensors.MotorAngleSensor(
          noisy_reading=False,
          num_motors=a1.NUM_MOTORS,
          default_pose=default_pose,
          scales=obs_scales.dof_pos),
    robot_sensors.MotorVelocitySensor(
          noisy_reading=False,
          num_motors=a1.NUM_MOTORS,
          scales=obs_scales.dof_vel),
    robot_sensors.LastActionSensor(
          num_actions=a1.NUM_MOTORS,
          scale=action_scale,
          default_pose=default_pose)
  ]

  env = locomotion_gym_env.LocomotionGymEnv(
      gym_config=gym_config,
      robot_class=robot_class,
      robot_kwargs=robot_kwargs,
      env_randomizers=[],
      robot_sensors=sensors,
      task=None)
  env = action_scale_wrapper.ActionScaleWrapper(env, action_scale, default_pose)
  return env

