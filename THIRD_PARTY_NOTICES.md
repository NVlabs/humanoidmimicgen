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
Files containing both RoboCasa-derived code and NVIDIA modifications preserve
the upstream MIT notice first and carry a stacked NVIDIA Apache-2.0 header.

### RoboCasa-Derived Assets

- Paths:
  - [humanoidmimicgen/locomanipulation/models/assets](humanoidmimicgen/locomanipulation/models/assets)
- Source project: RoboCasa asset tree
- License: MIT, matching the RoboCasa source distribution notice
- License text: [LICENSES/ROBOCASA-MIT.txt](LICENSES/ROBOCASA-MIT.txt)
- Copyright notice: Copyright (c) 2024 The RoboCasa Team

The repository includes assets retained for the humanoid loco-manipulation
environments and
dataset replay rendering.

### Unitree G1 Robot Description and Meshes

- Path:
  - [humanoidmimicgen/wbc/robot_model/model_data/g1](humanoidmimicgen/wbc/robot_model/model_data/g1)
- Source project: [Unitree Robotics `unitree_ros`](https://github.com/unitreerobotics/unitree_ros/tree/f3772ce54c56ef2d34c6aee8100bc768896c7d19/robots/g1_description)
- License: BSD-3-Clause
- License text: [LICENSES/UNITREE-BSD-3-CLAUSE.txt](LICENSES/UNITREE-BSD-3-CLAUSE.txt)
- Copyright notice: Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")

All 50 distributed STL meshes are byte-identical to files in the cited
Unitree G1 description snapshot. The distributed URDF is derived from the
Unitree G1 description and contains project-specific modifications; it retains
the Unitree attribution and adds a stacked NVIDIA Apache-2.0 modification
header.

### README Media

- Paths:
  - [docs/assets/readme_teaser.gif](docs/assets/readme_teaser.gif)
  - [docs/assets/benchmark_tasks](docs/assets/benchmark_tasks)
- Source project: [HumanoidMimicGen project website](https://humanoidmimicgen.github.io/),
  revision `f50f8653aca2aafaa44ebac52b358a12c4824488`
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

`readme_teaser.gif` is a compressed animation from `resources/teaser.mp4` with
the project name added to its final frames.
The benchmark task cards are resized copies of the images under
`resources/benchmark-tasks/`, except `pick-drill-from-holder.jpg`. That card is
composed from two Isaac Sim 5.1 RayTracedLighting renders of a project source
demonstration, showing the upright approach and grasp.

## Runtime Dependencies

The Python package metadata declares direct runtime dependencies used by the
distributed project code. This dependency inventory must be verified against
the exact versions selected for any public release.

| Component | Declared use | License and source |
| --- | --- | --- |
| mujoco | MuJoCo simulation bindings | [Apache-2.0](https://pypi.org/project/mujoco/3.2.6/) |
| numpy | Numeric arrays | [BSD-3-Clause](https://pypi.org/project/numpy/1.26.4/) |
| robosuite | Core MuJoCo environment runtime | MIT; see [LICENSES/ROBOSUITE-MIT.txt](LICENSES/ROBOSUITE-MIT.txt) |
| robosuite-models | G1 and other robosuite robot model registrations | MIT; see [LICENSES/ROBOSUITE-MODELS-MIT.txt](LICENSES/ROBOSUITE-MODELS-MIT.txt) |
| datasets | LeRobot dataset loading (`wbc-replay`) | [Apache-2.0](https://pypi.org/project/datasets/3.6.0/) |
| gymnasium | Environment interfaces (`wbc-replay`) | [MIT](https://pypi.org/project/gymnasium/0.29.1/) |
| lerobot | Dataset schemas (`wbc-replay`) | [MIT](https://pypi.org/project/lerobot/0.1.0/) |
| loguru | Runtime logging (`wbc-replay`) | [MIT](https://pypi.org/project/loguru/) |
| onnxruntime | WBC policy inference (`wbc-replay`) | [MIT](https://pypi.org/project/onnxruntime/1.22.1/) |
| opencv-python | Image processing (`wbc-replay`) | [Apache-2.0](https://pypi.org/project/opencv-python/4.11.0.86/) |
| pin | Pinocchio Python distribution (`wbc-replay`) | [BSD-3-Clause](https://pypi.org/project/pin/) |
| PyYAML | Configuration parsing (`wbc-replay`) | [MIT](https://pypi.org/project/PyYAML/6.0.3/) |
| scipy | Scientific computing (`wbc-replay`) | [BSD-3-Clause](https://pypi.org/project/scipy/1.15.3/) |
| torch | Tensor execution (`wbc-replay`) | [BSD-3-Clause plus bundled third-party notices](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE) |
| tqdm | Progress reporting (`wbc-replay`) | [MPL-2.0 AND MIT](https://pypi.org/project/tqdm/4.67.1/) |
| matplotlib | Optional visualization | [Matplotlib License](https://pypi.org/project/matplotlib/) |

The verified local import environment used `robosuite==1.5.1`,
`robosuite-models==1.0.0`, `mujoco==3.2.6`, and `numpy==1.26.4`, matching
the versions declared in `pyproject.toml`.
RoboSuite also installs transitive runtime packages such as
`numba`, `scipy`, `mink`, `qpsolvers`, `Pillow`, `opencv-python`, `pynput`,
`termcolor`, `pytest`, and `tqdm`. If a release artifact vendors or
redistributes a Python environment rather than only declaring package
dependencies, include the resolved transitive dependency license texts in
that artifact's license bundle.

For unpinned or ranged dependencies, preserve the selected distribution's
license and bundled third-party notices in any binary, container, or vendored
release artifact.

## Datasets, Model Weights, Robot Assets, and Generated Outputs

Training datasets and generated outputs are not included in this repository
snapshot. The distributed package includes G1 robot descriptions and meshes.
It does not include lower-body policy weights. The optional downloader fetches
the following pinned files directly from GR00T Whole-Body Control for local WBC
replay:

- `stand.onnx` is byte-identical to
  [`GR00T-WholeBodyControl-Balance.onnx`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx)
  (`sha256:f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18`).
- `walk.onnx` is byte-identical to
  [`GR00T-WholeBodyControl-Walk.onnx`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx)
  (`sha256:7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407`).

These separately downloaded files are model weights licensed under the
[NVIDIA Open Model License](LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt), not under
the Apache-2.0 source-code license. Required attribution: "Licensed by NVIDIA
Corporation under the NVIDIA Open Model License".

The distributed robot descriptions and meshes are the Unitree-derived assets
documented above and are distributed under the preserved BSD-3-Clause terms.

## Project License

NVIDIA-authored project code intended for open source distribution is
licensed under the Apache License, Version 2.0. The full Apache 2.0 text is
distributed in [LICENSE](LICENSE), and project-level notices are distributed
in [NOTICE](NOTICE). Optional ONNX model weights downloaded separately for WBC
replay are not part of this distribution and are licensed under the [NVIDIA
Open Model License](LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt).

## Release Review

Before any public distribution, complete the
[NVIDIA IP Review Process](https://nvidia.atlassian.net/wiki/display/OSS/IP+Review+Process)
for the exact repository contents. If code, datasets, model weights, robot
assets, generated outputs, or dependencies are added later, this notice file
must be updated before distribution.
