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

The verified local import environment used `robosuite==1.5.1`,
`robosuite-models==1.0.0`, `mujoco==3.2.6`, and `numpy==1.26.4`, matching
the versions declared in `pyproject.toml`.
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

## Datasets, Model Weights, Robot Assets, and Generated Outputs

Training datasets and generated outputs are not included in this repository
snapshot. The distributed package does include G1 robot descriptions and
meshes, plus two lower-body policy weights:

- `stand.onnx` is byte-identical to
  [`GR00T-WholeBodyControl-Balance.onnx`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx)
  (`sha256:f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18`).
- `walk.onnx` is byte-identical to
  [`GR00T-WholeBodyControl-Walk.onnx`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx)
  (`sha256:7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407`).

Both files are model weights licensed under the
[NVIDIA Open Model License](LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt), not under
the Apache-2.0 source-code license. Required attribution: "Licensed by NVIDIA
Corporation under the NVIDIA Open Model License".

Before public distribution, the exact provenance, copyright owner, and
distribution authorization for the robot descriptions and meshes must still be
recorded in the IP review. Add any required third-party attribution and license
text to this document and `LICENSES/`. This remains a release gate; do not infer
a license for binary or model assets from the Apache-2.0 project license.

## Project License

NVIDIA-authored project code intended for open source distribution is
licensed under the Apache License, Version 2.0. The full Apache 2.0 text is
distributed in [LICENSE](LICENSE), and project-level notices are distributed
in [NOTICE](NOTICE). The two bundled ONNX model weights are separately licensed
under the [NVIDIA Open Model License](LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt).

## Release Review

Before any public distribution, complete the
[NVIDIA IP Review Process](https://nvidia.atlassian.net/wiki/display/OSS/IP+Review+Process)
for the exact repository contents. If code, datasets, model weights, robot
assets, generated outputs, or dependencies are added later, this notice file
must be updated before distribution.
