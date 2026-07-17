"""Shared constants: OpenXR joint layout, Revo2 motor contract, model paths."""

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "revo2"

MJCF_PATH = {
    "left": ASSETS_DIR / "xml_left" / "brainco-lefthand-v2-patched.xml",
    "right": ASSETS_DIR / "xml_right" / "brainco-righthand-v2-patched.xml",
}

MJCF_PATH_FREE = {
    "left": ASSETS_DIR / "xml_left" / "brainco-lefthand-v2-patched-free.xml",
    "right": ASSETS_DIR / "xml_right" / "brainco-righthand-v2-patched-free.xml",
}

# --- OpenXR XR_EXT_hand_tracking joint indices (Pico 26-joint layout) ---
XR_PALM = 0
XR_WRIST = 1
XR_THUMB_TIP = 5
XR_INDEX_TIP = 10
XR_MIDDLE_TIP = 15
XR_RING_TIP = 20
XR_LITTLE_TIP = 25

# Tip index per finger, in BrainCo finger order (thumb, index, middle, ring, pinky)
XR_TIPS = {
    "thumb": XR_THUMB_TIP,
    "index": XR_INDEX_TIP,
    "middle": XR_MIDDLE_TIP,
    "ring": XR_RING_TIP,
    "pinky": XR_LITTLE_TIP,
}

# --- BrainCo Revo2 command contract ---
# 6 motors, normalized position command in [0,1], 0 = open, 1 = closed.
MOTOR_ORDER = ["thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky"]
FINGERS = ["index", "middle", "ring", "pinky"]  # the four 1-DOF fingers

# MJCF actuator name per motor per hand (patched models; actuators are named
# after the joint they drive).
ACTUATOR_NAME = {
    "left": {
        "thumb_flex": "left_thumb_proximal_joint",
        "thumb_rot": "left_thumb_metacarpal_joint",
        "index": "left_index_proximal_joint",
        "middle": "left_middle_proximal_joint",
        "ring": "left_ring_proximal_joint",
        "pinky": "left_pinky_proximal_joint",
    },
    "right": {
        "thumb_flex": "right_thumb_proximal_joint",
        "thumb_rot": "right_thumb_metacarpal_joint",
        "index": "right_index_proximal_joint",
        "middle": "right_middle_proximal_joint",
        "ring": "right_ring_proximal_joint",
        "pinky": "right_pinky_proximal_joint",
    },
}

# Body whose "touch" pad geom (or tip body) is the fingertip keypoint.
DISTAL_BODY = {
    "left": {f: f"left_{f}_distal_link" for f in ["thumb", "index", "middle", "ring", "pinky"]},
    "right": {f: f"right_{f}_distal_link" for f in ["thumb", "index", "middle", "ring", "pinky"]},
}

# URDF joint velocity limits (rad/s) -> normalized cmd rate limits (1/s),
# dividing by the driven joint's range (rad).
CMD_RATE_LIMIT = {
    "thumb_flex": 2.5303 / 1.03,
    "thumb_rot": 2.6175 / 1.57,
    "index": 2.2685 / 1.41,
    "middle": 2.2685 / 1.41,
    "ring": 2.2685 / 1.41,
    "pinky": 2.2685 / 1.41,
}
