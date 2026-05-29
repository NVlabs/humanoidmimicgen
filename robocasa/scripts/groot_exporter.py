"""Compatibility stub for exporter hooks not shipped with HumanoidMimicGen."""


class GrootExporter:
    """Placeholder for the external dataset exporter.

    HumanoidMimicGen's released playback path consumes existing LeRobot datasets
    and renders videos. Exporting new datasets to external schemas is not part
    of this local runtime.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "External dataset export is not included in the local HumanoidMimicGen runtime."
        )
