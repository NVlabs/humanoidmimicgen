# Robot Integration Guide

This guide explains how to integrate a new robot into the Robocasa framework.

## MJCF File Requirements
Location: 
- [`robocasa/models/assets/robots/`](robocasa/models/assets/robots)
- [`robocasa/models/assets/grippers/`](robocasa/models/assets/grippers)


### MJCF Structure Requirements

#### For robot main body
- The MJCF file must have a single root element containing all other elements (`<worldbody>`, `<asset>`, etc.)
- The robot's kinematic tree must start from a `<body name="base">` element that contains a freejoint and is placed directly under `<worldbody>`. No other freejoints should be present.
- Joint definitions for the right arm must precede those of the left arm in the XML structure. This is required by the `robosuite` library, which sets up references to the joint indices from the right arm first.
- End effector attachments (grippers) are supported on robot bodies named `<body name="right_eef">`
- Site definition of `right_center` and `left_center` are required for the arm part.
  e.g. `<site name="right_center" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 1 0 1" type="sphere" group="1" />`

#### For gripper
- It is better to align the orientation of wrist with the current ones such as Inspire hand.
- Add the eef site to the gripper model file. Note that the orientation of the eef site should follow the convention of the following figure.
  ![eef orientation](../../../../docs/images/eef_ori.png)

- Add force/torque sensors to the end effector site of the gripper
  e.g. `<site name="ft_frame" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 1 0 1" type="sphere" group="1" />` to the gripper model file

### Base Mounting Configuration
When operating in fixed-base mode, the body connected to the base joint will be anchored to the world frame. So choose this mounting point carefully, typically at the robot's pelvis/torso, to ensure proper fixed-base behavior.

## Controller Configuration
Location: [`robocasa/examples/third_party_controllers/`](robocasa/examples/third_party_controllers/)


### Configuration Requirements
1. Implement whole-body control configuration (excluding leg components for now)
2. Configure and tune `ik_posture_weight`
   - Remember to add "robot0_" prefix to all parameters


## Robot Registry Updates
Location: [`robocasa/models/robots/manipulators/`](robocasa/models/robots/manipulators/)

### Required Function Updates
Take a look at the existing robot class definitions for examples.
Remember to update the following functions with new robot's specifications:
1. `update_joints()`
   - Add joint names in correct order
   - Maintain right-before-left convention

2. `update_actuators()`
   - Add actuator names
   - Ensure order matches joint configuration


### Troubleshooting
- If the robot is not moving
  - Check the motor's ctrlrange. Make sure they are set to a range that allows for the robot to move.
- If the robot is moving weirdly
  - Check the mapping of joint and actuator names.
  - Check if the right part comes before the left part in the XML structure. If not, adjust them in the `update_joints()` and `update_actuators()` functions.
