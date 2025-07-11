# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from .trimesh import frame_trimesh, combine_trimeshes, box_trimesh
from isaacgym import gymtorch, gymapi, gymutil


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))

        self.frame_center = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
        self.num_goals = cfg.num_goals

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            self.vertices_ground, self.triangles_ground = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            print("Created {} vertices (only ground) ".format(self.vertices_ground.shape[0]))
            print("Created {} triangles (only ground) ".format(self.triangles_ground.shape[0]))

            # initialize
            self.vertices = self.vertices_ground
            self.triangles= self.triangles_ground

            if self.cfg.combine_frame:
                # ! create frame trimesh and combine them
                # ! create
                self.frame_vertices, self.frame_triangles = self.get_frame_trimeshes()
                # ! combine
                for i in range(len(self.frame_vertices)):
                    # add increment of the triangles
                    self.frame_triangles[i] = self.frame_triangles[i] + self.vertices.shape[0]
                    # combine the ground and the frame
                    self.vertices = np.concatenate((self.vertices, self.frame_vertices[i]), axis=0)
                    self.triangles = np.concatenate((self.triangles, self.frame_triangles[i]), axis=0)

                print("Created {} vertices (including frame) ".format(self.vertices.shape[0]))
                print("Created {} triangles (including frame) ".format(self.triangles.shape[0]))

    def get_frame_trimeshes(self):
        # create mulitple frame trimesh
        frame_vertices, frame_triangles = [], []
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                for k in range(self.cfg.num_goals):
                    # add border offset and hanging height to the frame center
                    center_position = self.frame_center[i, j, k] \
                                      + np.array([self.cfg.border_size, self.cfg.border_size, 0.0])
                    # create trimesh
                    frame_vertices_cur, frame_triangles_cur = frame_trimesh(size=np.array([self.cfg.frame_depth ,
                                                                                           self.cfg.frame_width ,
                                                                                           self.cfg.frame_height ]),
                                                                            inner_frame_scale= self.cfg.inner_frame_scale,
                                                                            center_position=center_position.astype(np.float32)
                                                                           )
                    frame_vertices.append(frame_vertices_cur)
                    frame_triangles.append(frame_triangles_cur)
        return frame_vertices, frame_triangles

    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005,
                                             downsampled_scale=self.cfg.downsampled_scale)

    def make_terrain(self, choice, difficulty):
        '''
        Set terrain properties: height_field_raw; goals

        Returns: terrain

        '''
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.length_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[8]:
            locomotion_rough_terrain(terrain)
            self.add_roughness(terrain)
        elif choice < self.proportions[9]:
            parkour_hurdle_terrain(terrain,
                                   num_stones=self.num_goals - 2,
                                   stone_len=0.1 + 0.3 * difficulty,
                                   hurdle_height_range=[0.1 + 0.1 * difficulty, 0.15 + 0.25 * difficulty],
                                   pad_height=0,
                                   x_range=[4.0, 4.1],
                                   y_range=self.cfg.y_range,
                                   half_valid_width=[0.4, 0.8],
                                   flat=True,
                                   )
            self.add_roughness(terrain)
        else:
            parkour_frame_terrain(terrain,
                                  first_frame_position=self.cfg.first_frame_position,
                                  num_stones=self.num_goals - 2,
                                  frame_interval_x=[self.cfg.frame_interval_x[0], self.cfg.frame_interval_x[1]],
                                  frame_interval_y=[self.cfg.frame_interval_y[0], self.cfg.frame_interval_y[1]],
                                )
            self.add_roughness(terrain)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        '''
        Set the private variable in this class: self.height_field_raw, self.env_origin_z(=0), self.goals
        '''
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        # self.terrain_type[i, j] = terrain.idx

        # ! for terrains with goals
        if hasattr(terrain, 'goals'):
            self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]

            # center of frame lie in the middle of the terrain goals
            # x----------||(first frame)-----X----||----X----||-----X----||----X  (X for goals, || for frame)
            first_frame_start_position = np.array([[self.cfg.first_frame_position, self.env_width/2]])
            # ! calculate the middle of every two frames (x{N} + x{N+1}) / 2 (N >= 1)
            temp = (terrain.goals[:-1] + terrain.goals[1:]) / 2
            # ! concat the first frame with the first frame
            frame_center = np.concatenate((first_frame_start_position, temp), axis=0)
            # ! calculate the frame in the world frame
            self.frame_center[i, j, :, :2] = frame_center + [i * self.env_length, j * self.env_width]
            # self.frame_center[i, j, :, 2] = self.cfg.frame_hanging_height
            self.frame_center[i, j, :, 2] = np.random.uniform(self.cfg.frame_hanging_height[0], self.cfg.frame_hanging_height[1])

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def locomotion_rough_terrain(terrain):
    terrain.height_field_raw = 0.0
    return terrain

def parkour_hurdle_terrain(terrain,
                           platform_len=4.0,
                           platform_height=0.,
                           num_stones=8,
                           stone_len=0.3,
                           x_range=[1.5, 2.4],
                           y_range=[-0.4, 0.4],
                           half_valid_width=[0.4, 0.8],
                           hurdle_height_range=[0.2, 0.3],
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    goals = np.zeros((num_stones + 2, 2))  # (num_goals, 2); 2 for x-y
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, :mid_y + rand_y - half_valid_width] = 0
            terrain.height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, mid_y + rand_y + half_valid_width:] = 0
        last_dis_x = dis_x
        goals[i + 1] = [dis_x, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

def parkour_frame_terrain(terrain,
                           first_frame_position=4.0,     # soft zone: where to set the first goal
                           num_stones=8,
                           frame_interval_x=[1.5, 2.4],
                           frame_interval_y=[-0.4, 0.4],
                           ):
    goals = np.zeros((num_stones + 2, 2))  # (num_goals, 2); 2 for x-y
    # terrain.height_field_raw[:] = -200
    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(frame_interval_x[0] / terrain.horizontal_scale)
    dis_x_max = round(frame_interval_x[1] / terrain.horizontal_scale)
    dis_y_min = round(frame_interval_y[0] / terrain.horizontal_scale)
    dis_y_max = round(frame_interval_y[1] / terrain.horizontal_scale)

    first_goal_position_x = round(first_frame_position / terrain.horizontal_scale) + np.random.randint(dis_x_min, dis_x_max)/2

    dis_x = first_goal_position_x
    goals[0] = [first_goal_position_x, mid_y]
    last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        last_dis_x = dis_x
        goals[i + 1] = [dis_x, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    terrain.goals = goals * terrain.horizontal_scale