import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR


VISUALIZE_RETARGETING = True

URDF_FILENAME = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

OUTPUT_DIR = "{LEGGED_GYM_ROOT_DIR}/datasets/mocap_motions_a1".format(
    LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

REF_POS_SCALE = 0.825
INIT_POS = np.array([0, 0, 0.32])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [6, 11, 16, 21]
SIM_HIP_JOINT_IDS = [2, 7, 12, 17]
SIM_ROOT_OFFSET = np.array([0, 0, -0.04])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, 0.06, 0.0]),
    np.array([0, -0.06, 0.0]),
    np.array([0, 0.06, 0.0]),
    np.array([0, -0.06, 0.0])
]
TOE_HEIGHT_OFFSET = 0.02

DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

FR_FOOT_NAME = "FR_foot"
FL_FOOT_NAME = "FL_foot"
HR_FOOT_NAME = "RR_foot"
HL_FOOT_NAME = "RL_foot"

MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    [
        "pace0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 162, 201, 1
    ],
    [
        "pace1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 201, 400, 1
    ],
    [
        "pace2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 400, 600, 1
    ],
    [
        "trot0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 448, 481, 1
    ],
    [
        "trot1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 400, 600, 1
    ],
    [
        "trot2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 480, 663, 1
    ],
    [
        "canter0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 430, 480, 1
    ],
    [
        "canter1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 380, 430, 1
    ],
    [
        "canter2",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 480, 566, 1
    ],
    [
        "right_turn0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 1085, 1124, 1.5
    ],
    [
        "right_turn1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 560, 670, 1.5
    ],
    [
        "left_turn0",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 2404, 2450, 1.5
    ],
    [
        "left_turn1",
        "{LEGGED_GYM_ROOT_DIR}/datasets/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt".format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), 120, 220, 1.5
    ]
]