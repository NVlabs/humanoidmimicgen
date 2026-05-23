def get_body_ik_solver_settings_type(robot: str):
    robot2body_ik_solver_settings_type = {
        "G1FixedLowerBody": "sim_optimized",
        "G1FixedBase": "sim_optimized",
        "G1FloatingBody": "sim_optimized",
        "G1FloatingBodyWithVertical": "sim_optimized",
        "G1ArmsOnly": "sim_optimized",
        "G1ArmsOnlyFloating": "sim_optimized",
        "G1": "default",
        "GR1ArmsAndWaistFourierHands": "default",
        "GR1FixedLowerBody": "default",
        "GR1ArmsOnlyFourierHands": "default",
        "GR1ArmsOnly": "default",
    }
    return robot2body_ik_solver_settings_type[robot]


def update_robosuite_controller_configs(
    robot: str,
    wbc_version: str = None,
):
    """
    Update the robosuite controller configs based on the robot type and wbc version.
    """
    body_ik_solver_settings_type = get_body_ik_solver_settings_type(robot)
    if robot.startswith("G1"):
        if wbc_version == "homie":
            robosuite_controller_configs = (
                "robocasa/examples/third_party_controller/default_mink_ik_g1_homie.json"
            )
        elif wbc_version == "homie_v2":
            robosuite_controller_configs = (
                "robocasa/examples/third_party_controller/default_mink_ik_g1_homie_v2.json"
            )
        elif wbc_version == "homie_v2_grav_comp":
            robosuite_controller_configs = (
                "robocasa/examples/third_party_controller/default_mink_ik_g1_homie_v2_grav_comp.json"
            )
        elif wbc_version == "homie_v2_grav_comp_tuned":
            robosuite_controller_configs = (
                "robocasa/examples/third_party_controller/default_mink_ik_g1_homie_v2_grav_comp_tuned.json"
            )
        elif wbc_version == "homie_v2_grav_comp_tuned_legs":
            robosuite_controller_configs = (
                "robocasa/examples/third_party_controller/default_mink_ik_g1_homie_v2_grav_comp_tuned_legs.json"
            )
        else:
            if body_ik_solver_settings_type == "default":
                robosuite_controller_configs = (
                    "robocasa/examples/third_party_controller/default_mink_ik_g1_wbc.json"
                )
            elif body_ik_solver_settings_type == "sim_optimized":
                robosuite_controller_configs = (
                    "robocasa/examples/third_party_controller/"
                    "default_mink_ik_g1_wbc_sim_optimized.json"
                )
            else:
                raise ValueError(
                    f"Invalid body_ik_solver_settings_type: {body_ik_solver_settings_type}"
                )
    elif robot.startswith("GR1"):
        return "robocasa/examples/third_party_controller/default_mink_ik_gr1_smallkd.json"
    else:
        raise ValueError(f"Invalid robot: {robot}")
    return robosuite_controller_configs
