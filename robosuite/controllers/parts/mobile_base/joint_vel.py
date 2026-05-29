import numpy as np

from robosuite.controllers.parts.mobile_base.mobile_base_controller import MobileBaseController
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}


class MobileBaseJointVelocityController(MobileBaseController):
    """
    Controller for controlling robot arm via impedance control. Allows position control of the robot's joints.

    NOTE: Control input actions assumed to be taken relative to the current joint positions. A given action to this
    controller is assumed to be of the form: (dpos_j0, dpos_j1, ... , dpos_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the joint pos error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of num_joints * 3. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be num_joints * 2.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        qpos_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the magnitude
            of a calculated goal joint position will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint position to
            the goal joint position during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        kp=50,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        qpos_limits=None,
        interpolator=None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        super().__init__(
            sim,
            joint_indexes,
            actuator_range,
            naming_prefix=kwargs.get("naming_prefix", None),
        )

        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits
        self.position_limits = np.array(qpos_limits) if qpos_limits is not None else qpos_limits

        # kp kd
        self.kp = self.nums2array(kp, self.control_dim)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], self.control_dim)
        self.kp_max = self.nums2array(kp_limits[1], self.control_dim)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], self.control_dim)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], self.control_dim)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, (
            "Error: Tried to instantiate OSC controller for unsupported "
            "impedance mode! Inputted impedance mode: {}, Supported modes: {}".format(
                impedance_mode, IMPEDANCE_MODES
            )
        )

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim *= 3
        elif self.impedance_mode == "variable_kp":
            self.control_dim *= 2

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize
        self.goal_qvel = None
        self.init_pos = None
        self.init_ori = None

    def set_goal(self, action, set_qpos=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_qpos (Iterable): If set, overrides @action and sets the desired absolute joint position goal state

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary
        jnt_dim = len(self.qpos_index)
        if self.impedance_mode == "variable":
            damping_ratio, kp, delta = (
                action[:jnt_dim],
                action[jnt_dim : 2 * jnt_dim],
                action[2 * jnt_dim :],
            )
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = (
                2
                * np.sqrt(self.kp)
                * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
            )
        elif self.impedance_mode == "variable_kp":
            kp, delta = action[:jnt_dim], action[jnt_dim:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        else:  # This is case "fixed"
            delta = action

        # Check to make sure delta is size self.joint_dim
        assert (
            len(delta) == jnt_dim
        ), "Delta qpos must be equal to the robot's joint dimension space!"

        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

        curr_pos, curr_ori = self.get_base_pose()

        # transform the action relative to initial base orientation
        init_theta = T.mat2euler(self.init_ori)[2]  # np.arctan2(self.init_pos[1], self.init_pos[0])
        curr_theta = T.mat2euler(curr_ori)[2]  # np.arctan2(curr_pos[1], curr_pos[0])
        theta = curr_theta - init_theta

        base_action = action
        # input raw base action is delta relative to current pose of base
        # controller expects deltas relative to initial pose of base at start of episode
        # transform deltas from current base pose coordinates to initial base pose coordinates
        x, y = base_action[0:2]

        # do the reverse of theta rotation
        base_action[0] = x * np.cos(theta) - y * np.sin(theta)
        base_action[1] = x * np.sin(theta) + y * np.cos(theta)

        self.goal_qvel = base_action
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qvel)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_qvel is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        desired_qvel = None

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                desired_qvel = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_qvel = np.array(self.goal_qvel)

        self.vels = desired_qvel

        ctrl_range = np.stack([self.actuator_min, self.actuator_max], axis=-1)
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.vels = bias + weight * self.vels

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.vels

    def reset_goal(self):
        """
        Resets joint position goal to be current position
        """
        self.goal_qvel = self.joint_vel

        self.init_pos, self.init_ori = self.get_base_pose()

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qvel)

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return "JOINT_VELOCITY"


class MobileBaseJointVelocityAndPositionController(MobileBaseJointVelocityController):
    """
    Mixed controller for controlling mobile base via both velocity and position control.
    Allows specifying which joints use velocity control and which use position control.

    For velocity joints: Uses the same logic as the parent class with bias/weight scaling
    For position joints: Passes through absolute position commands directly

    Additional Args:
        velocity_indices (list): List of joint indices that should use velocity control
        position_indices (list): List of joint indices that should use position control

    All other args are the same as MobileBaseJointVelocityController.
    """

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        velocity_indices=None,
        position_indices=None,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        kp=50,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        qpos_limits=None,
        interpolator=None,
        **kwargs,
    ):
        super().__init__(
            sim=sim,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            input_max=input_max,
            input_min=input_min,
            output_max=output_max,
            output_min=output_min,
            kp=kp,
            damping_ratio=damping_ratio,
            impedance_mode=impedance_mode,
            kp_limits=kp_limits,
            damping_ratio_limits=damping_ratio_limits,
            policy_freq=policy_freq,
            qpos_limits=qpos_limits,
            interpolator=interpolator,
            **kwargs,
        )

        # Set default indices if not provided
        jnt_dim = len(joint_indexes["joints"])
        if velocity_indices is None and position_indices is None:
            # Default: first 3 are velocity (forward, side, yaw), rest are position
            self.velocity_indices = list(range(min(3, jnt_dim)))
            self.position_indices = list(range(3, jnt_dim))
        else:
            self.velocity_indices = velocity_indices if velocity_indices is not None else []
            self.position_indices = position_indices if position_indices is not None else []

        # Validate indices
        all_indices = set(self.velocity_indices + self.position_indices)
        expected_indices = set(range(jnt_dim))
        assert all_indices == expected_indices, (
            f"velocity_indices + position_indices must cover all joint indices. "
            f"Expected: {expected_indices}, Got: {all_indices}"
        )

        # Store for later use
        self.jnt_dim = jnt_dim

    def run_controller(self):
        """
        Calculates the commands required to reach the desired setpoint.
        For velocity joints: applies bias/weight scaling like parent class
        For position joints: passes through absolute position commands directly

        Returns:
             np.array: Command values (mixed velocity and position)
        """
        # Make sure goal has been set
        if self.goal_qvel is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        desired_commands = None

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                desired_commands = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_commands = np.array(self.goal_qvel)

        # Initialize output commands
        output_commands = np.zeros_like(desired_commands)

        # Handle velocity indices with bias/weight scaling (original logic)
        if self.velocity_indices:
            vel_indices = np.array(self.velocity_indices)
            vel_commands = desired_commands[vel_indices]

            # Apply bias/weight scaling for velocity actuators
            ctrl_range = np.stack(
                [self.actuator_min[vel_indices], self.actuator_max[vel_indices]], axis=-1
            )
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            vel_commands = bias + weight * vel_commands

            output_commands[vel_indices] = vel_commands

        # Handle position indices (direct passthrough)
        if self.position_indices:
            pos_indices = np.array(self.position_indices)
            pos_commands = desired_commands[pos_indices]

            # For position control, pass through the absolute position directly
            # Optionally clamp to actuator limits if needed
            pos_commands = np.clip(
                pos_commands, self.actuator_min[pos_indices], self.actuator_max[pos_indices]
            )

            output_commands[pos_indices] = pos_commands

        self.vels = output_commands

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.vels

    @property
    def name(self):
        return "JOINT_VELOCITY_AND_POSITION"
