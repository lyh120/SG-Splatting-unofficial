import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sg_utils import inv_softplus
from utils.system_utils import mkdir_p


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return strip_symmetric(actual_covariance)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        sh_degree: int = 0,
        num_sg: int = 3,
        sg_init_sharpness: float = 8.0,
        diffuse_bias: float = 0.5,
        sg_axis_mode: str = "orthogonal",
        adaptive_sh_max_degree: int = 2,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.sg_axis_mode = sg_axis_mode if sg_axis_mode in {"free", "orthogonal"} else "orthogonal"
        self.num_sg = min(num_sg, 6) if self.sg_axis_mode == "orthogonal" else num_sg
        self.sg_init_sharpness = sg_init_sharpness
        self.diffuse_bias = diffuse_bias
        self.adaptive_sh_max_degree = adaptive_sh_max_degree

        self._xyz = torch.empty(0)
        self._diffuse = torch.empty(0)
        self._sg_axis = torch.empty(0)
        self._sg_basis_rotation = torch.empty(0)
        self._sg_sharpness = torch.empty(0)
        self._sg_amplitude = torch.empty(0)
        self._low_sh = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._diffuse,
            self._sg_axis,
            self._sg_basis_rotation,
            self._sg_sharpness,
            self._sg_amplitude,
            self._low_sh,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.num_sg,
            self.sg_init_sharpness,
            self.diffuse_bias,
            self.sg_axis_mode,
            self.adaptive_sh_max_degree,
        )

    def restore(self, model_args, training_args):
        base_fields = list(model_args)
        if len(base_fields) < 17:
            raise RuntimeError("Checkpoint format is too old and unsupported.")

        if len(base_fields) >= 21:
            self.active_sh_degree = base_fields[0]
            self._xyz = base_fields[1]
            self._diffuse = base_fields[2]
            self._sg_axis = base_fields[3]
            self._sg_basis_rotation = base_fields[4]
            self._sg_sharpness = base_fields[5]
            self._sg_amplitude = base_fields[6]
            self._low_sh = base_fields[7]
            self._scaling = base_fields[8]
            self._rotation = base_fields[9]
            self._opacity = base_fields[10]
            self.max_radii2D = base_fields[11]
            xyz_gradient_accum = base_fields[12]
            denom = base_fields[13]
            opt_dict = base_fields[14]
            self.spatial_lr_scale = base_fields[15]
            self.num_sg = base_fields[16]
            self.sg_init_sharpness = base_fields[17]
            self.diffuse_bias = base_fields[18]
            self.sg_axis_mode = base_fields[19]
            self.adaptive_sh_max_degree = base_fields[20]
        else:
            self.active_sh_degree = base_fields[0]
            self._xyz = base_fields[1]
            self._diffuse = base_fields[2]
            self._sg_axis = base_fields[3]
            self._sg_sharpness = base_fields[4]
            self._sg_amplitude = base_fields[5]
            self._scaling = base_fields[6]
            self._rotation = base_fields[7]
            self._opacity = base_fields[8]
            self.max_radii2D = base_fields[9]
            xyz_gradient_accum = base_fields[10]
            denom = base_fields[11]
            opt_dict = base_fields[12]
            self.spatial_lr_scale = base_fields[13]
            self.num_sg = base_fields[14]
            self.sg_init_sharpness = base_fields[15]
            self.diffuse_bias = base_fields[16]
            self.sg_axis_mode = "free"
            self.adaptive_sh_max_degree = 0
            rots = torch.zeros((self._xyz.shape[0], 4), device="cuda")
            rots[:, 0] = 1.0
            self._sg_basis_rotation = nn.Parameter(rots.requires_grad_(True))
            self._low_sh = nn.Parameter(
                torch.zeros((self._xyz.shape[0], 3, 8), device="cuda", dtype=self._xyz.dtype).requires_grad_(True)
            )
        if self.sg_axis_mode == "orthogonal":
            self.num_sg = min(self.num_sg, 6)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_diffuse(self):
        return torch.sigmoid(self._diffuse)

    @property
    def get_sg_axis(self):
        if self.sg_axis_mode == "orthogonal":
            canonical_axis = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                device=self._xyz.device,
                dtype=self._xyz.dtype,
            )
            canonical_axis = canonical_axis.unsqueeze(0).expand(self._xyz.shape[0], -1, -1)
            basis_rotation = build_rotation(F.normalize(self._sg_basis_rotation, dim=-1))
            rotated_axis = torch.matmul(canonical_axis, basis_rotation.transpose(1, 2))
            return F.normalize(rotated_axis[:, : self.num_sg, :], dim=-1)
        return F.normalize(self._sg_axis.view(-1, self.num_sg, 3), dim=-1)

    @property
    def get_sg_sharpness(self):
        return F.softplus(self._sg_sharpness).view(-1, self.num_sg, 1)

    @property
    def get_sg_amplitude(self):
        return F.softplus(self._sg_amplitude).view(-1, self.num_sg, 3)

    @property
    def get_low_sh(self):
        return self._low_sh

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def sg_regularization(self):
        return self.get_sg_amplitude.mean() + 0.1 * self.get_sg_sharpness.mean() + 0.01 * self.get_low_sh.abs().mean()

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda().clamp(1e-4, 1.0 - 1e-4)

        print("Number of points at initialisation :", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        diffuse = inverse_sigmoid((fused_color * self.diffuse_bias).clamp(1e-4, 1.0 - 1e-4))
        sg_axis = torch.randn((fused_point_cloud.shape[0], self.num_sg, 3), device="cuda")
        sg_basis_rotation = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        sg_basis_rotation[:, 0] = 1.0
        sg_sharpness = inv_softplus(
            torch.full((fused_point_cloud.shape[0], self.num_sg, 1), self.sg_init_sharpness, device="cuda")
        )
        sg_amplitude = inv_softplus(torch.full((fused_point_cloud.shape[0], self.num_sg, 3), 1e-3, device="cuda"))
        low_sh = torch.zeros((fused_point_cloud.shape[0], 3, 8), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._diffuse = nn.Parameter(diffuse.requires_grad_(True))
        self._sg_axis = nn.Parameter(sg_axis.reshape(fused_point_cloud.shape[0], -1).requires_grad_(True))
        self._sg_basis_rotation = nn.Parameter(sg_basis_rotation.requires_grad_(True))
        self._sg_sharpness = nn.Parameter(sg_sharpness.reshape(fused_point_cloud.shape[0], -1).requires_grad_(True))
        self._sg_amplitude = nn.Parameter(sg_amplitude.reshape(fused_point_cloud.shape[0], -1).requires_grad_(True))
        self._low_sh = nn.Parameter(low_sh.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._diffuse], "lr": training_args.diffuse_lr, "name": "diffuse"},
            {"params": [self._sg_axis], "lr": training_args.sg_axis_lr, "name": "sg_axis"},
            {"params": [self._sg_basis_rotation], "lr": training_args.sg_basis_lr, "name": "sg_basis_rotation"},
            {"params": [self._sg_sharpness], "lr": training_args.sg_sharpness_lr, "name": "sg_sharpness"},
            {"params": [self._sg_amplitude], "lr": training_args.sg_color_lr, "name": "sg_amplitude"},
            {"params": [self._low_sh], "lr": training_args.sh_lr, "name": "low_sh"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
        return None

    def construct_list_of_attributes(self):
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for idx in range(self._diffuse.shape[1]):
            attributes.append(f"diffuse_{idx}")
        for idx in range(self._sg_axis.shape[1]):
            attributes.append(f"sg_axis_{idx}")
        for idx in range(self._sg_basis_rotation.shape[1]):
            attributes.append(f"sg_basis_rot_{idx}")
        for idx in range(self._sg_sharpness.shape[1]):
            attributes.append(f"sg_sharpness_{idx}")
        for idx in range(self._sg_amplitude.shape[1]):
            attributes.append(f"sg_amplitude_{idx}")
        for idx in range(self._low_sh.reshape(self._low_sh.shape[0], -1).shape[1]):
            attributes.append(f"low_sh_{idx}")
        attributes.append("opacity")
        for idx in range(self._scaling.shape[1]):
            attributes.append(f"scale_{idx}")
        for idx in range(self._rotation.shape[1]):
            attributes.append(f"rot_{idx}")
        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        diffuse = self._diffuse.detach().cpu().numpy()
        sg_axis = self._sg_axis.detach().cpu().numpy()
        sg_basis_rotation = self._sg_basis_rotation.detach().cpu().numpy()
        sg_sharpness = self._sg_sharpness.detach().cpu().numpy()
        sg_amplitude = self._sg_amplitude.detach().cpu().numpy()
        low_sh = self._low_sh.detach().cpu().numpy().reshape(self._low_sh.shape[0], -1)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                diffuse,
                sg_axis,
                sg_basis_rotation,
                sg_sharpness,
                sg_amplitude,
                low_sh,
                opacities,
                scale,
                rotation,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        diffuse_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("diffuse_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        sg_axis_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_axis_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        sg_sharpness_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_sharpness_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        sg_basis_rotation_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_basis_rot_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        sg_amplitude_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("sg_amplitude_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        low_sh_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("low_sh_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")],
            key=lambda x: int(x.split("_")[-1]),
        )

        diffuse = np.stack([np.asarray(plydata.elements[0][name]) for name in diffuse_names], axis=1)
        sg_axis = np.stack([np.asarray(plydata.elements[0][name]) for name in sg_axis_names], axis=1)
        sg_sharpness = np.stack([np.asarray(plydata.elements[0][name]) for name in sg_sharpness_names], axis=1)
        if sg_basis_rotation_names:
            sg_basis_rotation = np.stack([np.asarray(plydata.elements[0][name]) for name in sg_basis_rotation_names], axis=1)
        else:
            sg_basis_rotation = np.zeros((xyz.shape[0], 4), dtype=np.float32)
            sg_basis_rotation[:, 0] = 1.0
        sg_amplitude = np.stack([np.asarray(plydata.elements[0][name]) for name in sg_amplitude_names], axis=1)
        if low_sh_names:
            low_sh = np.stack([np.asarray(plydata.elements[0][name]) for name in low_sh_names], axis=1)
            low_sh = low_sh.reshape(xyz.shape[0], 3, 8)
        else:
            low_sh = np.zeros((xyz.shape[0], 3, 8), dtype=np.float32)
        scales = np.stack([np.asarray(plydata.elements[0][name]) for name in scale_names], axis=1)
        rots = np.stack([np.asarray(plydata.elements[0][name]) for name in rot_names], axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._diffuse = nn.Parameter(torch.tensor(diffuse, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sg_axis = nn.Parameter(torch.tensor(sg_axis, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sg_basis_rotation = nn.Parameter(
            torch.tensor(sg_basis_rotation, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sg_sharpness = nn.Parameter(
            torch.tensor(sg_sharpness, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sg_amplitude = nn.Parameter(
            torch.tensor(sg_amplitude, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._low_sh = nn.Parameter(torch.tensor(low_sh, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._diffuse = optimizable_tensors["diffuse"]
        self._sg_axis = optimizable_tensors["sg_axis"]
        self._sg_basis_rotation = optimizable_tensors["sg_basis_rotation"]
        self._sg_sharpness = optimizable_tensors["sg_sharpness"]
        self._sg_amplitude = optimizable_tensors["sg_amplitude"]
        self._low_sh = optimizable_tensors["low_sh"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_diffuse,
        new_sg_axis,
        new_sg_basis_rotation,
        new_sg_sharpness,
        new_sg_amplitude,
        new_low_sh,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        new_tensors = {
            "xyz": new_xyz,
            "diffuse": new_diffuse,
            "sg_axis": new_sg_axis,
            "sg_basis_rotation": new_sg_basis_rotation,
            "sg_sharpness": new_sg_sharpness,
            "sg_amplitude": new_sg_amplitude,
            "low_sh": new_low_sh,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(new_tensors)
        self._xyz = optimizable_tensors["xyz"]
        self._diffuse = optimizable_tensors["diffuse"]
        self._sg_axis = optimizable_tensors["sg_axis"]
        self._sg_basis_rotation = optimizable_tensors["sg_basis_rotation"]
        self._sg_sharpness = optimizable_tensors["sg_sharpness"]
        self._sg_amplitude = optimizable_tensors["sg_amplitude"]
        self._low_sh = optimizable_tensors["low_sh"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_diffuse = self._diffuse[selected_pts_mask].repeat(N, 1)
        new_sg_axis = self._sg_axis[selected_pts_mask].repeat(N, 1)
        new_sg_basis_rotation = self._sg_basis_rotation[selected_pts_mask].repeat(N, 1)
        new_sg_sharpness = self._sg_sharpness[selected_pts_mask].repeat(N, 1)
        new_sg_amplitude = self._sg_amplitude[selected_pts_mask].repeat(N, 1)
        new_low_sh = self._low_sh[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_diffuse,
            new_sg_axis,
            new_sg_basis_rotation,
            new_sg_sharpness,
            new_sg_amplitude,
            new_low_sh,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_diffuse = self._diffuse[selected_pts_mask]
        new_sg_axis = self._sg_axis[selected_pts_mask]
        new_sg_basis_rotation = self._sg_basis_rotation[selected_pts_mask]
        new_sg_sharpness = self._sg_sharpness[selected_pts_mask]
        new_sg_amplitude = self._sg_amplitude[selected_pts_mask]
        new_low_sh = self._low_sh[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_diffuse,
            new_sg_axis,
            new_sg_basis_rotation,
            new_sg_sharpness,
            new_sg_amplitude,
            new_low_sh,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
