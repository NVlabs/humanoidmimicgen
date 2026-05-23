
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List
import numpy as np
import mujoco



def randomize_colors_and_lights(
    env: Any,
    *,
    seed: Optional[int] = None,
    geom_include: Optional[Iterable[str]] = None,   # names to include; None = all
    geom_exclude: Optional[Iterable[str]] = None,   # names to exclude
    color_jitter: float = 0.5,                     # 0..1; delta applied around current color
    alpha_min: float = 0.4,                         # keep objects visible
    light_pos_jitter: float = 0.2,                  # fraction of current norm
    light_rgb_jitter: float = 0.25                  # 0..1
) -> None:
    """
    Randomize geom colors (RGBA) and light params in-place.
    Safe for MuJoCo 2.x/3.x via `mujoco` Python bindings and robosuite envs.
    """
    rng = np.random.default_rng(seed)
    sim = getattr(env, "sim", None)
    if sim is None:
        print(f"WARNING: No sim found in env {env}. Cannot randomize appearance.")
        return
    model = sim.model

    # --- Geom color jitter (RGBA) ---
    # Works whether colors set via geom.rgba or material->mat_rgba (MuJoCo uses matid=-1 for per-geom RGBA)
    name2id = {model.geom_id2name(i): i for i in range(model.ngeom)}  # id->name may return None for unnamed
    include = set(geom_include) if geom_include is not None else None
    exclude = set(geom_exclude) if geom_exclude is not None else set()

    for gid in range(model.ngeom):
        name = model.geom_id2name(gid)
        if include is not None and (name not in include):
            continue
        if name in exclude:
            continue

        # If geom uses per-geom RGBA (matid == -1), jitter geom_rgba; otherwise jitter material color
        matid = model.geom_matid[gid]
        if matid == -1:
            rgba = model.geom_rgba[gid].copy()
            if rgba.size:  # defensive
                base = rgba[:3]
                delta = rng.uniform(-color_jitter, color_jitter, size=3)
                new_rgb = np.clip(base + delta, 0.0, 1.0)
                new_a = max(alpha_min, rgba[3])
                model.geom_rgba[gid] = np.array([*new_rgb, new_a], dtype=np.float32)
        else:
            rgba = model.mat_rgba[matid].copy()
            if rgba.size:
                base = rgba[:3]
                delta = rng.uniform(-color_jitter, color_jitter, size=3)
                new_rgb = np.clip(base + delta, 0.0, 1.0)
                new_a = max(alpha_min, rgba[3])
                model.mat_rgba[matid] = np.array([*new_rgb, new_a], dtype=np.float32)

    # --- Light jitter (position + RGB) ---
    nlight = model.nlight if hasattr(model, "nlight") else 0
    for lid in range(nlight):
        # Position jitter: move by a fraction of current norm in a random dir
        if hasattr(model, "light_pos"):
            pos = model.light_pos[lid].copy()
            norm = np.linalg.norm(pos) + 1e-6
            step = light_pos_jitter * norm * rng.normal(size=3)
            model.light_pos[lid] = pos + step

        # Ambient/diffuse/specular jitter in RGB
        for attr in ("light_ambient", "light_diffuse", "light_specular"):
            if hasattr(model, attr):
                rgb = getattr(model, attr)[lid].copy()
                delta = rng.uniform(-light_rgb_jitter, light_rgb_jitter, size=3)
                setattr(model, attr, np.array(getattr(model, attr), copy=True))
                getattr(model, attr)[lid] = np.clip(rgb + delta, 0.0, 1.0)


def randomize_textures(
    env: Any,
    *,
    seed: Optional[int] = None,
    brightness: float = 0.20,
    contrast: float = 0.20,
    noise: float = 0.05,
    include_skybox: bool = False,
    restore: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    sim = getattr(env, "sim", None)
    if sim is None:
        print(f"WARNING: No sim found in env {env}. Cannot randomize textures.")
        return
    model = sim.model

    rng = rng or np.random.default_rng()

    # Handle both pre-3.2 (tex_rgb) and 3.2+ (tex_data)
    buf_name = "tex_data" if hasattr(model, "tex_data") else "tex_rgb"
    tex_buf = getattr(model, buf_name)

    for tid in range(model.ntex):
        H = model.tex_height[tid]
        W = model.tex_width[tid]
        C = model.tex_nchannel[tid]
        start = model.tex_adr[tid]
        end = start + H * W * C
        # Write new bytes
        tex_buf[start:end] = rng.integers(0, 256, size=H * W * C, dtype=np.uint8)
       # If we have an OpenGL context, upload this texture so rendering matches CPU memory
        if env.sim._render_context_offscreen is not None:
            mujoco.mjr_uploadTexture(model._model, env.sim._render_context_offscreen.con, tid)
    return

def randomize_appearances(
    env: Any,
    *,
    seed: Optional[int] = None,
    # Color/light parameters
    geom_include: Optional[Iterable[str]] = None,
    geom_exclude: Optional[Iterable[str]] = None,
    color_jitter: float = 0.25,
    alpha_min: float = 0.4,
    light_pos_jitter: float = 0.2,
    light_rgb_jitter: float = 0.25,
    # Texture parameters
    brightness: float = 0.20,
    contrast: float = 0.20,
    noise: float = 0.05,
    include_skybox: bool = False,
    restore: bool = False,
) -> None:
    """
    Comprehensive appearance randomization that applies both color/lighting and texture changes.

    This function combines randomize_colors_and_lights() and randomize_textures() for complete
    visual domain randomization.

    Args:
        env: The environment containing the MuJoCo simulation
        seed: Random seed for reproducibility

        # Color/light parameters:
        geom_include: Geometry names to include (None = all)
        geom_exclude: Geometry names to exclude
        color_jitter: Color variation amount (0..1)
        alpha_min: Minimum alpha to keep objects visible
        light_pos_jitter: Light position jitter as fraction of current norm
        light_rgb_jitter: Light color variation (0..1)

        # Texture parameters:
        brightness: Brightness variation (±scale on [0..255])
        contrast: Contrast variation (±scale around mean 127.5)
        noise: Gaussian noise as fraction of 255
        include_skybox: Whether to randomize skybox textures
        restore: If True, restore original textures instead of randomizing
    """
    # Use the same seed for both functions to ensure reproducible results
    rng = np.random.default_rng(seed)

    # Generate separate seeds for each function to avoid correlation
    color_seed = rng.integers(0, 2**31) if seed is not None else None
    texture_seed = rng.integers(0, 2**31) if seed is not None else None

    # Apply color and lighting randomization
    randomize_colors_and_lights(
        env,
        seed=color_seed,
        geom_include=geom_include,
        geom_exclude=geom_exclude,
        color_jitter=color_jitter,
        alpha_min=alpha_min,
        light_pos_jitter=light_pos_jitter,
        light_rgb_jitter=light_rgb_jitter,
    )

    # Apply texture randomization
    # TODO: unclear if this actually changes the textures
    # randomize_textures(
    #     env,
    #     seed=texture_seed,
    #     brightness=brightness,
    #     contrast=contrast,
    #     noise=noise,
    #     include_skybox=include_skybox,
    #     restore=restore,
    # )
