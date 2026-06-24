from importlib.resources import files


_CONTROLLER_ROOT = files("humanoidmimicgen.locomanipulation.controllers")
SUPPORTED_WBC_VERSION = "homie_v2_grav_comp_tuned"
PLAYBACK_CONTROLLER_CONFIG = "default_mink_ik_g1_homie_v2_grav_comp_tuned.json"


def update_robosuite_controller_configs(robot: str, wbc_version: str):
    if not robot.startswith("G1"):
        raise ValueError(f"Unsupported robot for local WBC playback: {robot}")
    if wbc_version != SUPPORTED_WBC_VERSION:
        raise ValueError(f"Unsupported WBC version for local replay: {wbc_version}")
    return str(_CONTROLLER_ROOT.joinpath(PLAYBACK_CONTROLLER_CONFIG))
