# Retained Environments

HumanoidMimicGen retains the humanoid loco-manipulation environments needed by
the public simulation and WBC replay paths. Importing `robocasa` registers these
tasks with RoboSuite.

The source modules live in
[robocasa/environments/locomanipulation](../robocasa/environments/locomanipulation)
because they build on RoboCasa scene and object helpers while registering as
RoboSuite environments.

## Environment Names

The retained environment names are defined in
[`robocasa/environments/locomanipulation/base.py`](../robocasa/environments/locomanipulation/base.py):

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
