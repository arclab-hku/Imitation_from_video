# ! this env is for training locomotion AMP policy

from time import time
import numpy as np
import os
import pygame

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .aliengo_amp_config import AliengoAMPCfg

from rl.datasets.motion_loader import AMPLoader

class AliengoAMP(LeggedRobot):
    cfg: AliengoAMPCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

        self._get_commands_from_joystick = self.cfg.env.get_commands_from_joystick
        if self._get_commands_from_joystick:
            pygame.init()
            self._p1 = pygame.joystick.Joystick(0)
            self._p1.init()
            print(f"Loaded joystick with {self._p1.get_numaxes()} axes.")

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.obs_dict, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return self.obs_dict

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        self.reset_buf |= self.time_out_buf

        if self.cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            height_cutoff = self.root_states[:, 2] < -0.05
            self.reset_buf |= height_cutoff

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # assign to obs struct
        self.obs_dict['obs'] = self.obs_buf
        self.obs_dict['privileged_obs'] = self.privileged_obs_buf.to(self.device)
        self.obs_dict['proprio_hist'] = self.proprio_hist_buf.to(self.device).flatten(1)

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        return env_ids, terminal_amp_states

    def reset_idx(self, env_ids):
        """ Reset some environments.
                    Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
                    [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
                    Logs episode info
                    Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        # root_pos[:, :2] = self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _init_buffers(self):
        super()._init_buffers()

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)

    def compute_observations(self):
        """
            Computes observations
        """
        if self._get_commands_from_joystick:
          for event in pygame.event.get():
            lin_vel_x = -1 * self._p1.get_axis(1)
            if lin_vel_x >= 0:
             lin_vel_x *= torch.abs(torch.tensor(self.command_ranges["lin_vel_x"][1]))
            else:
             lin_vel_x *= torch.abs(torch.tensor(self.command_ranges["lin_vel_x"][0]))

            lin_vel_y = -1 * self._p1.get_axis(3)
            if lin_vel_y >= 0:
             lin_vel_y *= torch.abs(torch.tensor(self.command_ranges["lin_vel_y"][1]))
            else:
             lin_vel_y *= torch.abs(torch.tensor(self.command_ranges["lin_vel_y"][0]))

            ang_vel = -1 * self._p1.get_axis(0)
            if ang_vel >= 0:
             ang_vel *= torch.abs(torch.tensor(self.command_ranges["ang_vel_yaw"][1]))
            else:
             ang_vel *= torch.abs(torch.tensor(self.command_ranges["ang_vel_yaw"][0]))

            self.commands[:, 0] = lin_vel_x
            self.commands[:, 1] = lin_vel_y
            self.commands[:, 2] = ang_vel

        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add other privileged information
        if self.cfg.privileged_info.enable_foot_contact:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, contact), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        # if self.num_obs == self.num_privileged_obs - 6:
        #     self.obs_buf = self.privileged_obs_buf[:, 6:]
        # if self.num_obs == self.num_privileged_obs - 3:
        #     self.obs_buf = self.privileged_obs_buf[:, 3:]
        # else:
        #     self.obs_buf = torch.clone(self.privileged_obs_buf)
        self.obs_buf = self.privileged_obs_buf[:, 3:3+45]

        # ! add proprioceptive observation history
        # get previous step's obs
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()

        # get current step's obs
        cur_obs_buf = self.obs_buf.clone().unsqueeze(1)

        # concatenate to get full history
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        # Note: if reset, then the history buffer are all filled with the current observation
        at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, :] = self.obs_buf[at_reset_env_ids].unsqueeze(1)

        self.proprio_hist_buf = self.obs_buf_lag_history[:, -self.include_history_steps:].clone()

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = []
        with torch.no_grad():
            for i, chain_ee in enumerate(self.chain_ee):
                foot_pos.append(
                    chain_ee.forward_kinematics(joint_pos[:, i * 3:i * 3 +
                                                3]).get_matrix()[:, :3, 3])
        foot_pos = torch.cat(foot_pos, dim=-1)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.root_states[:, 2:3]
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)





