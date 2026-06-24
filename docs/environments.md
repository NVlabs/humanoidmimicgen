# Retained Environments

HumanoidMimicGen retains the humanoid loco-manipulation environments needed by
the public simulation and WBC replay paths. Importing
`humanoidmimicgen.locomanipulation` registers these tasks with RoboSuite.

The source modules live in
[humanoidmimicgen/locomanipulation/envs](../humanoidmimicgen/locomanipulation/envs).
The environment base subclasses RoboSuite's manipulation environment and uses
local scene, object, and success-criteria helpers retained for these tasks.

## Environment Names

The retained environment names are defined in
[`humanoidmimicgen/locomanipulation/envs/base.py`](../humanoidmimicgen/locomanipulation/envs/base.py):

- `LMDrillLift`
- `LMDrillLiftBi`
- `LMDrillPnP90`
- `LMDrillPnP90Bi`
- `LMDrillLiftObstacleBi`
- `LMDrillLiftObstacleDT`
- `LMDrillLiftObstacleDTBi`
- `LMDrillLiftObstacle`
- `LMDrillPnPCloser`
- `LMPickDrillFromHolder`
- `LMPickDrillFromHolderHigh`
- `LMPickDrillFromHolderStanding`
- `LMPickDrillFromHolderStandingEasyFar`
- `LMPushButton`
- `LMBoxLift`
- `LMBoxLiftStatic`
- `LMBoxLiftFloor`
- `LMBoxTableToCartStaticDT`
- `LMBoxTableToShelfStaticIndustrial`
- `LMBoxTableToShelfStaticIndustrialStartBack`
- `LMBottleLiftLowShelf`
- `LMTargetPnPBottleStatic`
- `LMPushShelfForward`

## Benchmark Task Set

The WBC benchmark wrapper uses the retained task families for:

- box lift from floor
- push button
- box lift
- push shelf forward
- drill lift
- drill pick-and-place
- box table-to-shelf
- pick drill from holder
- obstacle-aware drill lift

See [wbc_goal_replay.md](wbc_goal_replay.md) for the batch replay command and
output format.
