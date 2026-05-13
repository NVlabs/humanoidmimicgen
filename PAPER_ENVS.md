# Paper Environment Inventory

This file records the loco-manipulation environment selection used for the
planned HumanoidMimicGen open source release. The retained source payload is
based on the internal LMG scan scripts, with the strongest paper signal
coming from:

- `lmg_scan_feb_4.py`
- `lmg_scan_apr_16_standingeasy_far_drillref.py`
- `lmg_scan_apr_20_pickdrill_achieved.py`
- `lmg_scan_mildnav_pickdrill.py`

## Primary Scan Environments

The primary scan-backed environment names are:

- `LMDrillLiftBi`
- `LMNavDrillLiftBi`
- `LMDrillPnP90Bi`
- `LMNavDrillPnP90Bi`
- `LMDrillLiftObstacleBi`
- `LMNavDrillLiftObstacleBi`
- `LMPickDrillFromHolderHigh`
- `LMPickDrillFromHolderStandingEasyFar`
- `LMNavPickDrillFromHolderStandingEasyFar`
- `LMMildNavPickDrillFromHolder`

## Retained Source Files

The retained environment files are:

- [robocasa/environments/locomanipulation/base.py](robocasa/environments/locomanipulation/base.py)
- [robocasa/environments/locomanipulation/locomanip.py](robocasa/environments/locomanipulation/locomanip.py)
- [robocasa/environments/locomanipulation/locomanip_basic.py](robocasa/environments/locomanipulation/locomanip_basic.py)
- [robocasa/environments/locomanipulation/locomanip_d1_variants.py](robocasa/environments/locomanipulation/locomanip_d1_variants.py)
- [robocasa/environments/locomanipulation/locomanip_pnp.py](robocasa/environments/locomanipulation/locomanip_pnp.py)
- [robocasa/environments/locomanipulation/locomanip_push.py](robocasa/environments/locomanipulation/locomanip_push.py)
- [robocasa/environments/locomanipulation/locomanip_simple.py](robocasa/environments/locomanipulation/locomanip_simple.py)

Several retained source files define additional ancestor and sibling classes.
They are kept because the primary scan environments inherit shared scene,
object, randomization, and D1-variant logic from those modules.

## Minimal Support Files

The retained support files are limited to package registration, MJCF object
loading, scene arenas, camera constants, object placement, scene configs,
scene success criteria, transforms, and visual material randomization.
