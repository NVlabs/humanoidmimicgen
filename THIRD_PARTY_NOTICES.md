# Third-Party Notices

This document records third-party code, assets, and dependency notices for
the planned HumanoidMimicGen open source distribution.

## Distributed Third-Party Source

### RoboCasa

- Paths:
  - [robocasa](robocasa)
  - RoboCasa-derived support code in [robocasa/environments/locomanipulation](robocasa/environments/locomanipulation)
- Source project: RoboCasa
- License: MIT
- License text: [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt)
- Copyright notice: Copyright (c) 2024 The RoboCasa Team

The `robocasa` package in this repository is derived from RoboCasa source
code and includes NVIDIA-authored humanoid loco-manipulation environment
extensions. Third-party RoboCasa source retains its original copyright and
MIT license notice. NVIDIA-authored source files use Apache-2.0 notices.
Files containing both RoboCasa-derived code and NVIDIA modifications carry
file-level SPDX notices for both components.

### RoboCasa Assets

- Paths:
  - [robocasa/models/assets](robocasa/models/assets)
- Source project: RoboCasa asset tree
- License: MIT, matching the RoboCasa source distribution notice
- License text: [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt)
- Copyright notice: Copyright (c) 2024 The RoboCasa Team

The repository includes the RoboCasa assets vendored from the compatible GR00T
RoboCasa checkout for the retained loco-manipulation environments and
WBC-goal replay rendering.

### RoboSuite

- Paths:
  - [robosuite](robosuite)
- Source project: RoboSuite
- License: MIT
- License text: [LICENSES/ROBOSUITE-MIT.txt](LICENSES/ROBOSUITE-MIT.txt)
- Copyright notice: Copyright (c) 2018-2022, Stanford University

The `robosuite` package in this repository is vendored from the compatible
RoboSuite checkout used by the WBC replay path.

## Runtime Dependencies

The Python package metadata declares direct runtime dependencies used by the
distributed project code. This dependency inventory must be verified against
the exact versions selected for any public release.

| Component | Declared use | License to verify before release |
| --- | --- | --- |
| mujoco | MuJoCo simulation bindings | Apache-2.0 |
| numpy | Numeric arrays | BSD-3-Clause |
| robosuite-models | G1 and other robosuite robot model registrations | MIT; see [LICENSES/ROBOSUITE-MODELS-MIT.txt](LICENSES/ROBOSUITE-MODELS-MIT.txt) |

The verified local import environment used `robosuite==1.5.2`,
`robosuite-models==1.0.0`, `mujoco==3.3.7`, and `numpy==1.26.4`.
The vendored RoboSuite tree corresponds to the local `robosuite==1.5.2`
runtime. RoboSuite also installs transitive runtime packages such as
`numba`, `scipy`, `mink`, `qpsolvers`, `Pillow`, `opencv-python`, `pynput`,
`termcolor`, `pytest`, and `tqdm`. If a release artifact vendors or
redistributes a Python environment rather than only declaring package
dependencies, include the resolved transitive dependency license texts in
that artifact's license bundle.

The optional `visualization` extra declares `matplotlib` for the
`robocasa.utils.camera_utils.visualize_2d_projection` helper. Verify the
exact resolved Matplotlib version and license if that optional extra is
included in a distributed package, container, or release artifact.

## Datasets, Model Weights, and Generated Outputs

Datasets, model weights, generated outputs, and unrelated large RoboCasa
asset payloads are not included in this repository snapshot.

If any of those materials are added before release, preserve their copyright,
attribution, and license notices in place, and update this file with the
component name, repository path, license identifier, license text location,
and any required NOTICE or modification statements.

## Project License

NVIDIA-authored project code intended for open source distribution is
licensed under the Apache License, Version 2.0. The full Apache 2.0 text is
distributed in [LICENSE](LICENSE), and project-level notices are distributed
in [NOTICE](NOTICE).

## Release Review

Before any public distribution, complete the NVIDIA IP review process for
the exact release payload. If code, datasets, model weights, robot assets,
generated outputs, or dependencies are added later, this notice file must be
updated before distribution.
