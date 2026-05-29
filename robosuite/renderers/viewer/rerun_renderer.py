"""Rerun renderer class."""

import numpy as np


class RerunViewer:
    def __init__(self, sim, render_camera=None):
        raise NotImplementedError(
            "RerunViewer is not included in the local HumanoidMimicGen runtime. "
            "Use the default offscreen renderer for WBC-goal replay."
        )

        self.sim = sim
        self.rr_viz = RerunViz(
            image_keys=[],
            tensor_keys=[],
            app_name="offscrren_renderer",
            window_size=5.0,
        )
        if render_camera is None:
            render_camera = self.sim.model.camera_id2name(0)
        self.set_camera(camera_name=render_camera)
        self.height = 480
        self.width = 640

    def set_camera(self, camera_id=None, camera_name=None, width=None, height=None):
        """
        Set the camera view to the specified camera ID.

        Args:
            camera_id (int or list): id(s) of the camera to set the current viewer to
            camera_name (str or list or None): name(s) of the camera to set the current viewer to
        """

        # enforce exactly one arg
        assert (camera_id is not None) or (camera_name is not None)
        assert (camera_id is None) or (camera_name is None)

        # set width and height
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

        if camera_id is not None:
            if isinstance(camera_id, int):
                camera_id = [camera_id]
            self.camera_names = [self.sim.model.camera_id2name(cam_id) for cam_id in camera_id]
        else:
            if isinstance(camera_name, str):
                camera_name = [camera_name]
            self.camera_names = list(camera_name)

        self.rr_viz.set_rerun_keys(image_keys=self.camera_names, tensor_keys=[])

    def render(self):
        im_dict = {
            cam_name: self.sim.render(camera_name=cam_name, height=self.height, width=self.width)
            for cam_name in self.camera_names
        }

        im_dict = {cam_name: np.flip(im, axis=0) for cam_name, im in im_dict.items()}

        self.rr_viz.plot_images(im_dict)

    def close(self):
        """
        Any cleanup to close renderer.
        """

        # NOTE: assume that @sim will get cleaned up outside the renderer - just delete the reference
        self.sim = None

        # close window
        self.rr_viz.close()

    def reset(self):
        pass

    def update(self):
        pass
