# Retained Environments

HumanoidMimicGen retains the humanoid loco-manipulation environments needed by
the public simulation and WBC replay paths. Importing
`humanoidmimicgen.locomanipulation` registers these tasks with RoboSuite.

The source modules live in
[humanoidmimicgen/locomanipulation/envs](../humanoidmimicgen/locomanipulation/envs).
The environment base subclasses RoboSuite's manipulation environment and uses
local scene, object, and success-criteria helpers retained for these tasks.

## Environment Names

The retained paper environment names are defined in
[`humanoidmimicgen/locomanipulation/envs/base.py`](../humanoidmimicgen/locomanipulation/envs/base.py):

- `LMBoxLiftFloor`
- `LMPushButton`
- `LMBoxLift`
- `LMPushShelfForward`
- `LMDrillLift`
- `LMDrillPnP90`
- `LMBoxTableToShelfStaticIndustrial`
- `LMPickDrillFromHolderStandingEasyFar`
- `LMDrillLiftObstacle`

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
