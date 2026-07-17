import pickle
from pathlib import Path

import numpy as np
import pinocchio as pin

# casadi + the pinocchio-casadi bindings are ONLY needed for solve_ik (the IPOPT
# arm IK). ego2g1 uses its own mink IK and only wants solve_tau (gravity comp, pure
# pinocchio.rnea), and the pip/cmeel pinocchio wheel ships no casadi bindings. So
# make these optional: absent -> solve_ik/visualisation unavailable, solve_tau fine.
try:
    import casadi
    from pinocchio import casadi as cpin
except ImportError:
    casadi = None
    cpin = None
try:
    import meshcat.geometry as mg
    from pinocchio.visualize import MeshcatVisualizer
except ImportError:
    mg = None
    MeshcatVisualizer = None
from scipy.spatial.transform import Rotation as R

from unitree_deploy.utils.weighted_moving_filter import WeightedMovingFilter


def matrix_to_xyzrpy(T):
    """
    将 4x4 齐次变换矩阵转换为位置(x, y, z) + 欧拉角(r, p, y)（弧度）
    """
    assert T.shape == (4, 4)
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([xyz, rpy])


class G1_29_ArmIK:
    def __init__(self, unit_test=False, visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.unit_test = unit_test
        self.visualization = visualization

        urdf_path = Path(__file__).resolve().parent.parent / "assets" / "g1" / "g1_body29_hand14.urdf"
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "g1"
        self.robot = pin.RobotWrapper.BuildFromURDF(str(urdf_path), str(asset_path))

        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.data = self.reduced_robot.model.createData()

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)

        # Without the pinocchio-casadi bindings we cannot build the IPOPT IK, but the
        # reduced model above is all solve_tau (gravity comp) needs — so finish here.
        if cpin is None:
            self.opti = None
            self.init_data = np.zeros(self.reduced_robot.model.nq)
            self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
            self.vis = None
            return

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        opts = {
            # CasADi-level options
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
            "print_time": False,  # print or not
            # IPOPT solver options
            "ipopt.sb": "yes",  # disable Ipopt's license message
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
            # 'ipopt.hessian_approximation': 'limited-memory',
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (  # noqa: N806
                np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
                .astype(np.float32)
                .T
            )
            FRAME_AXIS_COLORS = (  # noqa: N806
                np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]])
                .astype(np.float32)
                .T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    # Save both robot.model and reduced_robot.model
    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    # Load both robot.model and reduced_robot.model
    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()

        return robot, reduced_robot

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)  # for visualization
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            v = (
                current_lr_arm_motor_dq * 0.0
                if current_lr_arm_motor_dq is not None
                else (sol_q - self.init_data) * 0.0
            )

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            if self.visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            v = (
                current_lr_arm_motor_dq * 0.0
                if current_lr_arm_motor_dq is not None
                else (sol_q - self.init_data) * 0.0
            )

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            print(
                f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}"
            )
            if self.visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def solve_tau(self, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        try:
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                current_lr_arm_motor_q,
                np.zeros(14),
                np.zeros(self.reduced_robot.model.nv),
            )
            return sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            return np.zeros(self.reduced_robot.model.nv)

    def solve_fk(self, q_full):
        assert q_full.shape == (self.reduced_robot.model.nq,)

        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_full)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        ee_pose_l = self.reduced_robot.data.oMf[self.reduced_robot.model.getFrameId("L_ee")].homogeneous
        ee_pose_r = self.reduced_robot.data.oMf[self.reduced_robot.model.getFrameId("R_ee")].homogeneous

        ee_xyzrpy_l = matrix_to_xyzrpy(ee_pose_l)
        ee_xyzrpy_r = matrix_to_xyzrpy(ee_pose_r)

        return ee_xyzrpy_l, ee_xyzrpy_r

    def solve_fk_matrix(self, q_full):
        assert q_full.shape == (self.reduced_robot.model.nq,)

        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_full)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)

        ee_pose_l = self.reduced_robot.data.oMf[self.reduced_robot.model.getFrameId("L_ee")].homogeneous
        ee_pose_r = self.reduced_robot.data.oMf[self.reduced_robot.model.getFrameId("R_ee")].homogeneous

        return ee_pose_l, ee_pose_r
