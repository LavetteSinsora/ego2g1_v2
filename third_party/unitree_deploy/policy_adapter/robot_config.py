import numpy as np

# fmt: off
INIT_POSE = {
    'unitree_g1_wbc_dex1': np.array( [ 0.14457, -0.00584, -0.02911,  0.25332, -0.05492, -0.529,   -0.14641,
                                       0.44876, -0.07364, -0.05896, -0.0597,   0.1155,  -0.60938,  0.15445,
                                       0.,      -0.,      -0.,       0. ,      0.,       0.,       0.74,
                                       0.04764, 0.
    ], dtype=np.float32),

    'unitree_g1_wbc_dex1_no_image': np.array([ 0.14457, -0.00584, -0.02911,  0.25332, -0.05492, -0.529,   -0.14641,
                                               0.44876, -0.07364, -0.05896, -0.0597,   0.1155,  -0.60938,  0.15445,
                                               0.,      -0.,      -0.,       0. ,      0.,       0.,       0.74,
                                               0.04764, 0.
    ], dtype=np.float32),


    'unitree_g1_dex1': np.array([0.10559805, 0.02726714, -0.01210221, -0.33341318, -0.22513399, -0.02627627, -0.15437093,
                                 0.1273793 , -0.1674708 , -0.11544029, -0.40095493,  0.44332668,  0.11566751,  0.3936641,
                                 5.4, 5.4], dtype=np.float32),

    'unitree_g1_dex1_no_image': np.array([0.10559805, 0.02726714, -0.01210221, -0.33341318, -0.22513399, -0.02627627, -0.15437093,
                                          0.1273793 , -0.1674708 , -0.11544029, -0.40095493,  0.44332668,  0.11566751,  0.3936641,
                                          5.4, 5.4], dtype=np.float32),
}

CAM_KEY = {
    "unitree_g1_dex1": {
        "observation.images.cam_left_high": "cam_left_high",
        "observation.images.cam_right_high": "cam_right_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    },
    "unitree_g1_wbc_dex1": {
        "observation.images.cam_left_high": "cam_left_high",
        "observation.images.cam_right_high": "cam_right_high",
        "observation.images.cam_left_wrist": "cam_left_wrist",
        "observation.images.cam_right_wrist": "cam_right_wrist",
    },
    "unitree_g1_dex1_no_image": {},
    "unitree_g1_wbc_dex1_no_image": {},
}

OBS_STATE_KEYS = {
    "unitree_g1_dex1": {
        "left_arm": slice(0, 7),
        "right_arm": slice(7, 14),
        "left_hand": slice(14, 15),
        "right_hand": slice(15, 16),
    },
    "unitree_g1_dex1_no_image": {
        "left_arm": slice(0, 7),
        "right_arm": slice(7, 14),
        "left_hand": slice(14, 15),
        "right_hand": slice(15, 16),
    },
    "unitree_g1_wbc_dex1": {
        "left_arm": slice(0, 7),
        "right_arm": slice(7, 14),
        "waist": slice(14, 17),
        "left_leg": slice(17, 23),
        "right_leg": slice(23, 29),
        "left_hand": slice(29, 30),
        "right_hand": slice(30, 31),
    },
    "unitree_g1_wbc_dex1_no_image": {
        "left_arm": slice(0, 7),
        "right_arm": slice(7, 14),
        "waist": slice(14, 17),
        "left_leg": slice(17, 23),
        "right_leg": slice(23, 29),
        "left_hand": slice(29, 30),
        "right_hand": slice(30, 31),
    }
}

REAL_ACTION_KEYS = {
    "unitree_g1_dex1": {
        "dim": 16,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
        }
    },
    "unitree_g1_dex1_no_image": {
        "dim": 16,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
        }
    },
    "unitree_g1_wbc_dex1": {
        "dim": 23,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
            "command": slice(16, 23),
        }
    },
    "unitree_g1_wbc_dex1_no_image": {
        "dim": 23,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
            "command": slice(16, 23),
        }
    }
}

DATASET_ACTION_KEYS = {
    "unitree_g1_dex1": {
        "dim": 16,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
        }
    },
    "unitree_g1_dex1_no_image": {
        "dim": 16,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
        }
    },
    "unitree_g1_wbc_dex1": {
        "dim": 35,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
            "command": slice(28, 35),
        }
    },
    "unitree_g1_wbc_dex1_no_image": {
        "dim": 35,
        "keys": {
            "left_arm": slice(0, 7),
            "right_arm": slice(7, 14),
            "left_hand": slice(14, 15),
            "right_hand": slice(15, 16),
            "command": slice(28, 35),
        }
    }
}
# fmt: on
