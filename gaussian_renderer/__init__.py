import math

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.gaussian_model import GaussianModel
from utils.sg_utils import eval_sg
from utils.sh_utils import RGB2SH, eval_sh


def _estimate_projected_radius(pc: GaussianModel, viewpoint_camera, tanfovx: float, tanfovy: float):
    ones = torch.ones((pc.get_xyz.shape[0], 1), device=pc.get_xyz.device, dtype=pc.get_xyz.dtype)
    homo_xyz = torch.cat([pc.get_xyz, ones], dim=1)
    view_xyz = torch.matmul(homo_xyz, viewpoint_camera.world_view_transform)
    depth = torch.clamp(torch.abs(view_xyz[:, 2]), min=1e-4)

    fx = 0.5 * float(viewpoint_camera.image_width) / tanfovx
    fy = 0.5 * float(viewpoint_camera.image_height) / tanfovy
    focal = 0.5 * (fx + fy)
    world_radius = pc.get_scaling.max(dim=1).values
    return focal * world_radius / depth


def _eval_adaptive_low_sh(
    pc: GaussianModel,
    viewdirs: torch.Tensor,
    projected_radius: torch.Tensor,
    max_degree: int,
    small_th: float,
    medium_th: float,
):
    max_degree = int(max(0, min(2, max_degree)))
    if max_degree == 0:
        return pc.get_diffuse

    sh0 = RGB2SH(pc.get_diffuse)
    sh_coeff = torch.cat([sh0.unsqueeze(-1), pc.get_low_sh], dim=-1)
    viewdirs = F.normalize(viewdirs, dim=-1)

    if max_degree == 1:
        deg_ids = torch.where(projected_radius < small_th, 0, 1)
    else:
        deg_ids = torch.where(
            projected_radius < small_th,
            torch.zeros_like(projected_radius, dtype=torch.long),
            torch.where(projected_radius < medium_th, torch.ones_like(projected_radius, dtype=torch.long), 2),
        )

    low_color = torch.zeros_like(pc.get_diffuse)
    for deg in range(max_degree + 1):
        mask = deg_ids == deg
        if mask.any():
            low_color[mask] = eval_sh(deg, sh_coeff[mask], viewdirs[mask])
    return low_color


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    sg_weight: float = 1.0,
    use_adaptive_low_sh: bool = False,
    adaptive_sh_max_degree: int = 2,
    sh_small_radius_threshold: float = 1.5,
    sh_medium_radius_threshold: float = 6.0,
):
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if override_color is None:
        viewdirs = viewpoint_camera.camera_center.unsqueeze(0) - means3D
        if use_adaptive_low_sh:
            projected_radius = _estimate_projected_radius(pc, viewpoint_camera, tanfovx, tanfovy)
            base_color = _eval_adaptive_low_sh(
                pc=pc,
                viewdirs=viewdirs,
                projected_radius=projected_radius,
                max_degree=adaptive_sh_max_degree,
                small_th=sh_small_radius_threshold,
                medium_th=sh_medium_radius_threshold,
            )
        else:
            base_color = pc.get_diffuse

        sg_term = eval_sg(
            viewdirs,
            pc.get_sg_axis,
            pc.get_sg_sharpness,
            pc.get_sg_amplitude,
        )
        colors_precomp = base_color + float(max(0.0, min(1.0, sg_weight))) * sg_term
        colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)
    else:
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "colors_precomp": colors_precomp,
    }
