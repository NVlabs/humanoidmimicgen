from importlib.resources import files


_CONTROLLER_ROOT = files("humanoidmimicgen.locomanipulation").joinpath(
    "examples/third_party_controller"
)

PLAYBACK_CONTROLLER_CONFIGS = {
    "homie_v2": "default_mink_ik_g1_homie_v2.json",
    "homie_v2_grav_comp": "default_mink_ik_g1_homie_v2_grav_comp.json",
    "homie_v2_grav_comp_tuned": "default_mink_ik_g1_homie_v2_grav_comp_tuned.json",
    "homie_v2_grav_comp_tuned_legs": "default_mink_ik_g1_homie_v2_grav_comp_tuned_legs.json",
}


def update_robosuite_controller_configs(robot: str, wbc_version: str):
    if not robot.startswith("G1"):
        raise ValueError(f"Unsupported robot for local WBC playback: {robot}")
    try:
        return str(_CONTROLLER_ROOT.joinpath(PLAYBACK_CONTROLLER_CONFIGS[wbc_version]))
    except KeyError as exc:
        raise ValueError(f"Unsupported WBC version for local replay: {wbc_version}") from exc
