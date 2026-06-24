# Third-Party Notices

This document records third-party code, assets, and dependency notices for
the HumanoidMimicGen repository snapshot.

## Distributed Third-Party Source

### RoboCasa-Derived Source

- Paths:
  - [humanoidmimicgen/locomanipulation](humanoidmimicgen/locomanipulation)
- Source project: RoboCasa
- License: MIT
- License text: [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt)
- Copyright notice: Copyright (c) 2024 The RoboCasa Team

The retained loco-manipulation support code is derived from RoboCasa source
code and includes NVIDIA-authored humanoid environment extensions. Third-party
RoboCasa source retains its original copyright and
MIT license notice. NVIDIA-authored source files use Apache-2.0 notices.
Files containing both RoboCasa-derived code and NVIDIA modifications carry
file-level SPDX notices for both components.

### RoboCasa-Derived Assets

- Paths:
  - [humanoidmimicgen/locomanipulation/models/assets](humanoidmimicgen/locomanipulation/models/assets)
- Source project: RoboCasa asset tree
- License: MIT, matching the RoboCasa source distribution notice
- License text: [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt)
- Copyright notice: Copyright (c) 2024 The RoboCasa Team

The repository includes assets retained for the humanoid loco-manipulation
environments and
WBC-goal replay rendering.

## Runtime Dependencies

The Python package metadata declares direct runtime dependencies used by the
distributed project code. This dependency inventory must be verified against
the exact versions selected for any public release.

| Component | Declared use | License to verify before release |
| --- | --- | --- |
| mujoco | MuJoCo simulation bindings | Apache-2.0 |
| numpy | Numeric arrays | BSD-3-Clause |
| robosuite | Core MuJoCo environment runtime | MIT; see [LICENSES/ROBOSUITE-MIT.txt](LICENSES/ROBOSUITE-MIT.txt) |
| robosuite-models | G1 and other robosuite robot model registrations | MIT; see [LICENSES/ROBOSUITE-MODELS-MIT.txt](LICENSES/ROBOSUITE-MODELS-MIT.txt) |

The verified local import environment used `robosuite==1.5.2`,
`robosuite-models==1.0.0`, `mujoco==3.3.7`, and `numpy==1.26.4`.
RoboSuite also installs transitive runtime packages such as
`numba`, `scipy`, `mink`, `qpsolvers`, `Pillow`, `opencv-python`, `pynput`,
`termcolor`, `pytest`, and `tqdm`. If a release artifact vendors or
redistributes a Python environment rather than only declaring package
dependencies, include the resolved transitive dependency license texts in
that artifact's license bundle.

The optional `visualization` extra declares `matplotlib` for the
`humanoidmimicgen.locomanipulation.utils.camera_utils.visualize_2d_projection`
helper. Verify the
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
the exact repository contents. If code, datasets, model weights, robot assets,
generated outputs, or dependencies are added later, this notice file must be
updated before distribution.
