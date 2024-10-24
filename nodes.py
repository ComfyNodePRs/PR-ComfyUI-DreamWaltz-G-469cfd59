import os
import os.path as osp
from pathlib import Path
from .configs import (
    TrainConfig,
    NeRFConfig,
    RenderConfig,
    GuideConfig,
    DataConfig,
    PromptConfig,
    OptimConfig,
    LogConfig,
)
from .core.trainer import Trainer


class RenderConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gs_type": ("STRING", {"default": RenderConfig.gs_type}),
                "deform_type": ("STRING", {"default": RenderConfig.deform_type}),
                "deform_with_shape": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_with_shape},
                ),
                "deform_rotation_mode": (
                    "STRING",
                    {"default": RenderConfig.deform_rotation_mode},
                ),
                "lbs_lr": ("FLOAT", {"default": RenderConfig.lbs_lr}),
                "betas_lr": ("FLOAT", {"default": RenderConfig.betas_lr}),
                "deform_learn_v_template": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_learn_v_template},
                ),
                "deform_learn_shapedirs": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_learn_shapedirs},
                ),
                "deform_learn_posedirs": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_learn_posedirs},
                ),
                "deform_learn_expr_dirs": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_learn_expr_dirs},
                ),
                "deform_learn_lbs_weights": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_learn_lbs_weights},
                ),
                "deform_learn_J_regressor": (
                    "BOOLEAN",
                    {"default": RenderConfig.deform_learn_J_regressor},
                ),
                "always_animate": ("BOOLEAN", {"default": RenderConfig.always_animate}),
                "lbs_weight_smooth": (
                    "BOOLEAN",
                    {"default": RenderConfig.lbs_weight_smooth},
                ),
                "lbs_weight_smooth_K": (
                    "INT",
                    {
                        "default": (
                            RenderConfig.lbs_weight_smooth_K
                            if RenderConfig.lbs_weight_smooth_K is not None
                            else 30
                        )
                    },
                ),
                "lbs_weight_smooth_N": (
                    "INT",
                    {
                        "default": (
                            RenderConfig.lbs_weight_smooth_N
                            if RenderConfig.lbs_weight_smooth_N is not None
                            else 5000
                        )
                    },
                ),
                "use_joint_shape_offsets": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_joint_shape_offsets},
                ),
                "use_vertex_shape_offsets": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_vertex_shape_offsets},
                ),
                "use_vertex_pose_offsets": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_vertex_pose_offsets},
                ),
                "use_non_rigid_offsets": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_non_rigid_offsets},
                ),
                "use_non_rigid_scales": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_non_rigid_scales},
                ),
                "use_non_rigid_rotations": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_non_rigid_rotations},
                ),
                "non_rigid_scale_mode": (
                    "STRING",
                    {"default": RenderConfig.non_rigid_scale_mode},
                ),
                "non_rigid_rotation_mode": (
                    "STRING",
                    {"default": RenderConfig.non_rigid_rotation_mode},
                ),
                "sh_levels": ("INT", {"default": RenderConfig.sh_levels}),
                "spatial_scale": (
                    "FLOAT",
                    {
                        "default": (
                            RenderConfig.spatial_scale
                            if RenderConfig.spatial_scale is not None
                            else 1.0
                        )
                    },
                ),
                "init_opacity": ("FLOAT", {"default": RenderConfig.init_opacity}),
                "init_offset": ("FLOAT", {"default": RenderConfig.init_offset}),
                "init_scale": ("FLOAT", {"default": RenderConfig.init_scale}),
                "init_scale_radius_rate": (
                    "FLOAT",
                    {"default": RenderConfig.init_scale_radius_rate},
                ),
                "max_scale": ("FLOAT", {"default": RenderConfig.max_scale}),
                "bg_color_r": ("FLOAT", {"default": RenderConfig.bg_color[0]}),
                "bg_color_g": ("FLOAT", {"default": RenderConfig.bg_color[1]}),
                "bg_color_b": ("FLOAT", {"default": RenderConfig.bg_color[2]}),
                "use_mlp_background": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_mlp_background},
                ),
                "use_video_background": (
                    "STRING",
                    {
                        "default": (
                            RenderConfig.use_video_background
                            if RenderConfig.use_video_background is not None
                            else ""
                        )
                    },
                ),
                "use_gs_background": (
                    "STRING",
                    {
                        "default": (
                            RenderConfig.use_gs_background
                            if RenderConfig.use_gs_background is not None
                            else ""
                        )
                    },
                ),
                "gaussian_color_init": (
                    "STRING",
                    {"default": RenderConfig.gaussian_color_init},
                ),
                "gaussian_point_init": (
                    "STRING",
                    {"default": RenderConfig.gaussian_point_init},
                ),
                "gaussian_scale_init": (
                    "STRING",
                    {"default": RenderConfig.gaussian_scale_init},
                ),
                "n_gaussians": ("INT", {"default": RenderConfig.n_gaussians}),
                "n_gaussians_per_vertex": (
                    "INT",
                    {"default": RenderConfig.n_gaussians_per_vertex},
                ),
                "n_gaussians_per_triangle": (
                    "INT",
                    {"default": RenderConfig.n_gaussians_per_triangle},
                ),
                "position_lr_init": (
                    "FLOAT",
                    {"default": RenderConfig.position_lr_init},
                ),
                "position_lr_final": (
                    "FLOAT",
                    {"default": RenderConfig.position_lr_final},
                ),
                "feature_lr": ("FLOAT", {"default": RenderConfig.feature_lr}),
                "opacity_lr": ("FLOAT", {"default": RenderConfig.opacity_lr}),
                "scaling_lr": ("FLOAT", {"default": RenderConfig.scaling_lr}),
                "rotation_lr": ("FLOAT", {"default": RenderConfig.rotation_lr}),
                "use_densifier": ("BOOLEAN", {"default": RenderConfig.use_densifier}),
                "densify_from_iter": (
                    "INT",
                    {
                        "default": (
                            RenderConfig.densify_from_iter
                            if RenderConfig.densify_from_iter is not None
                            else 0
                        )
                    },
                ),
                "densify_until_iter": (
                    "INT",
                    {
                        "default": (
                            RenderConfig.densify_until_iter
                            if RenderConfig.densify_until_iter is not None
                            else 0
                        )
                    },
                ),
                "densify_grad_threshold": (
                    "FLOAT",
                    {"default": RenderConfig.densify_grad_threshold},
                ),
                "densify_disable_clone": (
                    "BOOLEAN",
                    {"default": RenderConfig.densify_disable_clone},
                ),
                "densify_disable_split": (
                    "BOOLEAN",
                    {"default": RenderConfig.densify_disable_split},
                ),
                "densify_disable_prune": (
                    "BOOLEAN",
                    {"default": RenderConfig.densify_disable_prune},
                ),
                "densify_disable_reset": (
                    "BOOLEAN",
                    {"default": RenderConfig.densify_disable_reset},
                ),
                "enable_grad_prune": (
                    "BOOLEAN",
                    {"default": RenderConfig.enable_grad_prune},
                ),
                "from_nerf": (
                    "STRING",
                    {
                        "default": (
                            RenderConfig.from_nerf
                            if RenderConfig.from_nerf is not None
                            else ""
                        )
                    },
                ),
                "nerf_resolution": ("INT", {"default": RenderConfig.nerf_resolution}),
                "nerf_exclusion_bboxes": (
                    "STRING",
                    {
                        "default": (
                            RenderConfig.nerf_exclusion_bboxes
                            if RenderConfig.nerf_exclusion_bboxes is not None
                            else ""
                        )
                    },
                ),
                "reset_nerf": ("BOOLEAN", {"default": RenderConfig.reset_nerf}),
                "use_nerf_opacities": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_opacities},
                ),
                "use_nerf_scales_and_quaternions": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_scales_and_quaternions},
                ),
                "use_nerf_scales": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_scales},
                ),
                "use_nerf_quaternions": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_quaternions},
                ),
                "use_nerf_encoded_position": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_encoded_position},
                ),
                "use_deform_scales_and_quaternions": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_deform_scales_and_quaternions},
                ),
                "use_nerf_mesh_opacities": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_mesh_opacities},
                ),
                "use_nerf_mesh_scales_and_quaternions": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_nerf_mesh_scales_and_quaternions},
                ),
                "prune_points_close_to_mesh": (
                    "BOOLEAN",
                    {"default": RenderConfig.prune_points_close_to_mesh},
                ),
                "prune_dists_close_to_mesh": (
                    "FLOAT",
                    {
                        "default": (
                            RenderConfig.prune_dists_close_to_mesh
                            if RenderConfig.prune_dists_close_to_mesh is not None
                            else 0.01
                        )
                    },
                ),
                "learn_positions": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_positions},
                ),
                "learn_scales": ("BOOLEAN", {"default": RenderConfig.learn_scales}),
                "learn_quaternions": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_quaternions},
                ),
                "learn_lbs_weights": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_lbs_weights},
                ),
                "learn_hand_betas": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_hand_betas},
                ),
                "learn_face_betas": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_face_betas},
                ),
                "learn_mesh_bary_coords": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_mesh_bary_coords},
                ),
                "learn_mesh_vertex_coords": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_mesh_vertex_coords},
                ),
                "learn_mesh_scales": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_mesh_scales},
                ),
                "learn_mesh_quaternions": (
                    "BOOLEAN",
                    {"default": RenderConfig.learn_mesh_quaternions},
                ),
                "lambda_outfit_offset": (
                    "FLOAT",
                    {"default": RenderConfig.lambda_outfit_offset},
                ),
                "lambda_outfit_scale": (
                    "FLOAT",
                    {"default": RenderConfig.lambda_outfit_scale},
                ),
                "render_mesh_binding_3d_gaussians_only": (
                    "BOOLEAN",
                    {"default": RenderConfig.render_mesh_binding_3d_gaussians_only},
                ),
                "render_unconstrained_3d_gaussians_only": (
                    "BOOLEAN",
                    {"default": RenderConfig.render_unconstrained_3d_gaussians_only},
                ),
                "use_zero_scales": (
                    "BOOLEAN",
                    {"default": RenderConfig.use_zero_scales},
                ),
                "use_constant_color_r": (
                    "FLOAT",
                    {
                        "default": (
                            RenderConfig.use_constant_colors[0]
                            if RenderConfig.use_constant_colors is not None
                            else 0.0
                        )
                    },
                ),
                "use_constant_color_g": (
                    "FLOAT",
                    {
                        "default": (
                            RenderConfig.use_constant_colors[1]
                            if RenderConfig.use_constant_colors is not None
                            else 0.0
                        )
                    },
                ),
                "use_constant_color_b": (
                    "FLOAT",
                    {
                        "default": (
                            RenderConfig.use_constant_colors[2]
                            if RenderConfig.use_constant_colors is not None
                            else 0.0
                        )
                    },
                ),
                "use_constant_opacities": (
                    "FLOAT",
                    {
                        "default": (
                            RenderConfig.use_constant_opacities
                            if RenderConfig.use_constant_opacities is not None
                            else 0.0
                        )
                    },
                ),
                "avatar_scale": (
                    "STRING",
                    {
                        "default": (
                            RenderConfig.avatar_scale
                            if RenderConfig.avatar_scale is not None
                            else ""
                        )
                    },
                ),
                "avatar_transl": (
                    "STRING",
                    {
                        "default": (
                            RenderConfig.avatar_transl
                            if RenderConfig.avatar_transl is not None
                            else ""
                        )
                    },
                ),
                "use_fixed_n_gaussians": (
                    "INT",
                    {
                        "default": (
                            RenderConfig.use_fixed_n_gaussians
                            if RenderConfig.use_fixed_n_gaussians is not None
                            else 100000
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("RENDERCFG",)
    RETURN_NAMES = ("render_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        gs_type,
        deform_type,
        deform_with_shape,
        deform_rotation_mode,
        lbs_lr,
        betas_lr,
        deform_learn_v_template,
        deform_learn_shapedirs,
        deform_learn_posedirs,
        deform_learn_expr_dirs,
        deform_learn_lbs_weights,
        deform_learn_J_regressor,
        always_animate,
        lbs_weight_smooth,
        lbs_weight_smooth_K,
        lbs_weight_smooth_N,
        use_joint_shape_offsets,
        use_vertex_shape_offsets,
        use_vertex_pose_offsets,
        use_non_rigid_offsets,
        use_non_rigid_scales,
        use_non_rigid_rotations,
        non_rigid_scale_mode,
        non_rigid_rotation_mode,
        sh_levels,
        spatial_scale,
        init_opacity,
        init_offset,
        init_scale,
        init_scale_radius_rate,
        max_scale,
        bg_color_r,
        bg_color_g,
        bg_color_b,
        use_mlp_background,
        use_video_background,
        use_gs_background,
        gaussian_color_init,
        gaussian_point_init,
        gaussian_scale_init,
        n_gaussians,
        n_gaussians_per_vertex,
        n_gaussians_per_triangle,
        position_lr_init,
        position_lr_final,
        feature_lr,
        opacity_lr,
        scaling_lr,
        rotation_lr,
        use_densifier,
        densify_from_iter,
        densify_until_iter,
        densify_grad_threshold,
        densify_disable_clone,
        densify_disable_split,
        densify_disable_prune,
        densify_disable_reset,
        enable_grad_prune,
        from_nerf,
        nerf_resolution,
        nerf_exclusion_bboxes,
        reset_nerf,
        use_nerf_opacities,
        use_nerf_scales_and_quaternions,
        use_nerf_scales,
        use_nerf_quaternions,
        use_nerf_encoded_position,
        use_deform_scales_and_quaternions,
        use_nerf_mesh_opacities,
        use_nerf_mesh_scales_and_quaternions,
        prune_points_close_to_mesh,
        prune_dists_close_to_mesh,
        learn_positions,
        learn_scales,
        learn_quaternions,
        learn_lbs_weights,
        learn_hand_betas,
        learn_face_betas,
        learn_mesh_bary_coords,
        learn_mesh_vertex_coords,
        learn_mesh_scales,
        learn_mesh_quaternions,
        lambda_outfit_offset,
        lambda_outfit_scale,
        render_mesh_binding_3d_gaussians_only,
        render_unconstrained_3d_gaussians_only,
        use_zero_scales,
        use_constant_color_r,
        use_constant_color_g,
        use_constant_color_b,
        use_constant_opacities,
        avatar_scale,
        avatar_transl,
        use_fixed_n_gaussians,
    ):
        # Combine bg_color and use_constant_colors into tuples
        bg_color = (bg_color_r, bg_color_g, bg_color_b)
        use_constant_colors = (
            use_constant_color_r,
            use_constant_color_g,
            use_constant_color_b,
        )

        # Create an instance of RenderConfig with the inputs provided by the user, or default values if not modified
        config = RenderConfig(
            gs_type=gs_type,
            deform_type=deform_type,
            deform_with_shape=deform_with_shape,
            deform_rotation_mode=deform_rotation_mode,
            lbs_lr=lbs_lr,
            betas_lr=betas_lr,
            deform_learn_v_template=deform_learn_v_template,
            deform_learn_shapedirs=deform_learn_shapedirs,
            deform_learn_posedirs=deform_learn_posedirs,
            deform_learn_expr_dirs=deform_learn_expr_dirs,
            deform_learn_lbs_weights=deform_learn_lbs_weights,
            deform_learn_J_regressor=deform_learn_J_regressor,
            always_animate=always_animate,
            lbs_weight_smooth=lbs_weight_smooth,
            lbs_weight_smooth_K=lbs_weight_smooth_K,
            lbs_weight_smooth_N=lbs_weight_smooth_N,
            use_joint_shape_offsets=use_joint_shape_offsets,
            use_vertex_shape_offsets=use_vertex_shape_offsets,
            use_vertex_pose_offsets=use_vertex_pose_offsets,
            use_non_rigid_offsets=use_non_rigid_offsets,
            use_non_rigid_scales=use_non_rigid_scales,
            use_non_rigid_rotations=use_non_rigid_rotations,
            non_rigid_scale_mode=non_rigid_scale_mode,
            non_rigid_rotation_mode=non_rigid_rotation_mode,
            sh_levels=sh_levels,
            spatial_scale=spatial_scale,
            init_opacity=init_opacity,
            init_offset=init_offset,
            init_scale=init_scale,
            init_scale_radius_rate=init_scale_radius_rate,
            max_scale=max_scale,
            bg_color=bg_color,
            use_mlp_background=use_mlp_background,
            use_video_background=use_video_background,
            use_gs_background=use_gs_background,
            gaussian_color_init=gaussian_color_init,
            gaussian_point_init=gaussian_point_init,
            gaussian_scale_init=gaussian_scale_init,
            n_gaussians=n_gaussians,
            n_gaussians_per_vertex=n_gaussians_per_vertex,
            n_gaussians_per_triangle=n_gaussians_per_triangle,
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
            use_densifier=use_densifier,
            densify_from_iter=densify_from_iter,
            densify_until_iter=densify_until_iter,
            densify_grad_threshold=densify_grad_threshold,
            densify_disable_clone=densify_disable_clone,
            densify_disable_split=densify_disable_split,
            densify_disable_prune=densify_disable_prune,
            densify_disable_reset=densify_disable_reset,
            enable_grad_prune=enable_grad_prune,
            from_nerf=from_nerf,
            nerf_resolution=nerf_resolution,
            nerf_exclusion_bboxes=nerf_exclusion_bboxes,
            reset_nerf=reset_nerf,
            use_nerf_opacities=use_nerf_opacities,
            use_nerf_scales_and_quaternions=use_nerf_scales_and_quaternions,
            use_nerf_scales=use_nerf_scales,
            use_nerf_quaternions=use_nerf_quaternions,
            use_nerf_encoded_position=use_nerf_encoded_position,
            use_deform_scales_and_quaternions=use_deform_scales_and_quaternions,
            use_nerf_mesh_opacities=use_nerf_mesh_opacities,
            use_nerf_mesh_scales_and_quaternions=use_nerf_mesh_scales_and_quaternions,
            prune_points_close_to_mesh=prune_points_close_to_mesh,
            prune_dists_close_to_mesh=prune_dists_close_to_mesh,
            learn_positions=learn_positions,
            learn_scales=learn_scales,
            learn_quaternions=learn_quaternions,
            learn_lbs_weights=learn_lbs_weights,
            learn_hand_betas=learn_hand_betas,
            learn_face_betas=learn_face_betas,
            learn_mesh_bary_coords=learn_mesh_bary_coords,
            learn_mesh_vertex_coords=learn_mesh_vertex_coords,
            learn_mesh_scales=learn_mesh_scales,
            learn_mesh_quaternions=learn_mesh_quaternions,
            lambda_outfit_offset=lambda_outfit_offset,
            lambda_outfit_scale=lambda_outfit_scale,
            render_mesh_binding_3d_gaussians_only=render_mesh_binding_3d_gaussians_only,
            render_unconstrained_3d_gaussians_only=render_unconstrained_3d_gaussians_only,
            use_zero_scales=use_zero_scales,
            use_constant_colors=use_constant_colors,
            use_constant_opacities=use_constant_opacities,
            avatar_scale=avatar_scale,
            avatar_transl=avatar_transl,
            use_fixed_n_gaussians=use_fixed_n_gaussians,
        )
        return (config,)


class NeRFConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "desired_resolution": (
                    "INT",
                    {"default": NeRFConfig.desired_resolution},
                ),
                "num_levels": ("INT", {"default": NeRFConfig.num_levels}),
                "level_dim": ("INT", {"default": NeRFConfig.level_dim}),
                "base_resolution": ("INT", {"default": NeRFConfig.base_resolution}),
                "density_activation": (
                    "STRING",
                    {"default": NeRFConfig.density_activation},
                ),
                "cuda_ray": ("BOOLEAN", {"default": NeRFConfig.cuda_ray}),
                "grid_size": ("INT", {"default": NeRFConfig.grid_size}),
                "max_steps": ("INT", {"default": NeRFConfig.max_steps}),
                "num_steps": ("INT", {"default": NeRFConfig.num_steps}),
                "upsample_steps": ("INT", {"default": NeRFConfig.upsample_steps}),
                "update_extra_interval": (
                    "INT",
                    {"default": NeRFConfig.update_extra_interval},
                ),
                "max_ray_batch": ("INT", {"default": NeRFConfig.max_ray_batch}),
                "density_thresh": ("FLOAT", {"default": NeRFConfig.density_thresh}),
                "bound": ("FLOAT", {"default": NeRFConfig.bound}),
                "dt_gamma": ("FLOAT", {"default": NeRFConfig.dt_gamma}),
                "min_near": ("FLOAT", {"default": NeRFConfig.min_near}),
                "backbone": ("STRING", {"default": NeRFConfig.backbone}),
                "nerf_type": ("STRING", {"default": NeRFConfig.nerf_type}),
                "structure": ("STRING", {"default": NeRFConfig.structure}),
                "density_prior": ("STRING", {"default": NeRFConfig.density_prior}),
                "bg_mode": ("STRING", {"default": NeRFConfig.bg_mode}),
                "bg_radius": ("FLOAT", {"default": NeRFConfig.bg_radius}),
                "rand_bg_prob": (
                    "FLOAT",
                    {
                        "default": (
                            NeRFConfig.rand_bg_prob
                            if NeRFConfig.rand_bg_prob is not None
                            else 0.0
                        )
                    },
                ),
                "bg_suppress": ("BOOLEAN", {"default": NeRFConfig.bg_suppress}),
                "bg_suppress_dist": ("FLOAT", {"default": NeRFConfig.bg_suppress_dist}),
                "detach_bg_weights_sum": (
                    "BOOLEAN",
                    {"default": NeRFConfig.detach_bg_weights_sum},
                ),
                "dmtet": ("BOOLEAN", {"default": NeRFConfig.dmtet}),
                "dmtet_reso_scale": ("INT", {"default": NeRFConfig.dmtet_reso_scale}),
                "lock_geo": ("BOOLEAN", {"default": NeRFConfig.lock_geo}),
                "tet_grid_size": ("INT", {"default": NeRFConfig.tet_grid_size}),
                "lambda_normal": ("FLOAT", {"default": NeRFConfig.lambda_normal}),
                "lambda_2d_normal_smooth": (
                    "FLOAT",
                    {"default": NeRFConfig.lambda_2d_normal_smooth},
                ),
                "lambda_3d_normal_smooth": (
                    "FLOAT",
                    {"default": NeRFConfig.lambda_3d_normal_smooth},
                ),
                "lambda_mesh_normal": (
                    "FLOAT",
                    {"default": NeRFConfig.lambda_mesh_normal},
                ),
                "lambda_mesh_laplacian": (
                    "FLOAT",
                    {"default": NeRFConfig.lambda_mesh_laplacian},
                ),
                "optimizer": ("STRING", {"default": NeRFConfig.optimizer}),
                "lr": ("FLOAT", {"default": NeRFConfig.lr}),
                "bg_lr": ("FLOAT", {"default": NeRFConfig.bg_lr}),
                "start_shading_iter": (
                    "INT",
                    {
                        "default": (
                            NeRFConfig.start_shading_iter
                            if NeRFConfig.start_shading_iter is not None
                            else 0
                        )
                    },
                ),
                "lr_policy": ("STRING", {"default": NeRFConfig.lr_policy}),
                "lambda_opacity": ("FLOAT", {"default": NeRFConfig.lambda_opacity}),
                "lambda_entropy": ("FLOAT", {"default": NeRFConfig.lambda_entropy}),
                "lambda_emptiness": ("FLOAT", {"default": NeRFConfig.lambda_emptiness}),
                "sparsity_multiplier": (
                    "FLOAT",
                    {"default": NeRFConfig.sparsity_multiplier},
                ),
                "sparsity_step": ("FLOAT", {"default": NeRFConfig.sparsity_step}),
                "lambda_shape": ("FLOAT", {"default": NeRFConfig.lambda_shape}),
            },
        }

    RETURN_TYPES = ("NERFCFG",)
    RETURN_NAMES = ("nerf_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        desired_resolution,
        num_levels,
        level_dim,
        base_resolution,
        density_activation,
        cuda_ray,
        grid_size,
        max_steps,
        num_steps,
        upsample_steps,
        update_extra_interval,
        max_ray_batch,
        density_thresh,
        bound,
        dt_gamma,
        min_near,
        backbone,
        nerf_type,
        structure,
        density_prior,
        bg_mode,
        bg_radius,
        rand_bg_prob,
        bg_suppress,
        bg_suppress_dist,
        detach_bg_weights_sum,
        dmtet,
        dmtet_reso_scale,
        lock_geo,
        tet_grid_size,
        lambda_normal,
        lambda_2d_normal_smooth,
        lambda_3d_normal_smooth,
        lambda_mesh_normal,
        lambda_mesh_laplacian,
        optimizer,
        lr,
        bg_lr,
        start_shading_iter,
        lr_policy,
        lambda_opacity,
        lambda_entropy,
        lambda_emptiness,
        sparsity_multiplier,
        sparsity_step,
        lambda_shape,
    ):
        # Create an instance of NeRFConfig with the inputs provided by the user, or default values if not modified
        config = NeRFConfig(
            desired_resolution=desired_resolution,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            density_activation=density_activation,
            cuda_ray=cuda_ray,
            grid_size=grid_size,
            max_steps=max_steps,
            num_steps=num_steps,
            upsample_steps=upsample_steps,
            update_extra_interval=update_extra_interval,
            max_ray_batch=max_ray_batch,
            density_thresh=density_thresh,
            bound=bound,
            dt_gamma=dt_gamma,
            min_near=min_near,
            backbone=backbone,
            nerf_type=nerf_type,
            structure=structure,
            density_prior=density_prior,
            bg_mode=bg_mode,
            bg_radius=bg_radius,
            rand_bg_prob=rand_bg_prob,
            bg_suppress=bg_suppress,
            bg_suppress_dist=bg_suppress_dist,
            detach_bg_weights_sum=detach_bg_weights_sum,
            dmtet=dmtet,
            dmtet_reso_scale=dmtet_reso_scale,
            lock_geo=lock_geo,
            tet_grid_size=tet_grid_size,
            lambda_normal=lambda_normal,
            lambda_2d_normal_smooth=lambda_2d_normal_smooth,
            lambda_3d_normal_smooth=lambda_3d_normal_smooth,
            lambda_mesh_normal=lambda_mesh_normal,
            lambda_mesh_laplacian=lambda_mesh_laplacian,
            optimizer=optimizer,
            lr=lr,
            bg_lr=bg_lr,
            start_shading_iter=start_shading_iter,
            lr_policy=lr_policy,
            lambda_opacity=lambda_opacity,
            lambda_entropy=lambda_entropy,
            lambda_emptiness=lambda_emptiness,
            sparsity_multiplier=sparsity_multiplier,
            sparsity_step=sparsity_step,
            lambda_shape=lambda_shape,
        )
        return (config,)


class GuideConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Text fields
                "text": ("STRING", {"default": GuideConfig.text}),
                "text_set": (
                    "STRING",
                    {
                        "default": (
                            GuideConfig.text_set
                            if GuideConfig.text_set is not None
                            else ""
                        )
                    },
                ),
                "null_text": ("STRING", {"default": GuideConfig.null_text}),
                "negative_text": ("STRING", {"default": GuideConfig.negative_text}),
                "use_negative_text": (
                    "BOOLEAN",
                    {"default": GuideConfig.use_negative_text},
                ),
                "negative_text_in_SBP": (
                    "STRING",
                    {"default": GuideConfig.negative_text_in_SBP},
                ),
                # General
                "dtype": ("STRING", {"default": GuideConfig.dtype}),
                "grad_viz": ("BOOLEAN", {"default": GuideConfig.grad_viz}),
                # Diffusion Model
                "diffusion": ("STRING", {"default": GuideConfig.diffusion}),
                "diffusion_fp16": ("BOOLEAN", {"default": GuideConfig.diffusion_fp16}),
                # ControlNet
                "use_controlnet": ("BOOLEAN", {"default": GuideConfig.use_controlnet}),
                "controlnet": ("STRING", {"default": GuideConfig.controlnet}),
                "controlnet_fp16": (
                    "BOOLEAN",
                    {"default": GuideConfig.controlnet_fp16},
                ),
                "controlnet_condition": (
                    "STRING",
                    {"default": GuideConfig.controlnet_condition},
                ),
                "controlnet_scale": (
                    "FLOAT",
                    {"default": GuideConfig.controlnet_scale},
                ),
                # Extra Module
                "lora_name": (
                    "STRING",
                    {
                        "default": (
                            GuideConfig.lora_name
                            if GuideConfig.lora_name is not None
                            else ""
                        )
                    },
                ),
                "concept_name": (
                    "STRING",
                    {
                        "default": (
                            GuideConfig.concept_name
                            if GuideConfig.concept_name is not None
                            else ""
                        )
                    },
                ),
                # CFG
                "guidance_scale": ("FLOAT", {"default": GuideConfig.guidance_scale}),
                "guidance_adjust": ("STRING", {"default": GuideConfig.guidance_adjust}),
                # Timestep
                "min_timestep": ("FLOAT", {"default": GuideConfig.min_timestep}),
                "max_timestep": ("FLOAT", {"default": GuideConfig.max_timestep}),
                "time_sampling": ("STRING", {"default": GuideConfig.time_sampling}),
                "time_annealing": ("STRING", {"default": GuideConfig.time_annealing}),
                "time_annealing_window": (
                    "STRING",
                    {"default": GuideConfig.time_annealing_window},
                ),
                # SDS
                "sds_loss_type": ("STRING", {"default": GuideConfig.sds_loss_type}),
                "sds_weight_type": ("STRING", {"default": GuideConfig.sds_weight_type}),
                "input_interpolate": (
                    "BOOLEAN",
                    {"default": GuideConfig.input_interpolate},
                ),
                # Gradients
                "grad_latent_clip": (
                    "BOOLEAN",
                    {"default": GuideConfig.grad_latent_clip},
                ),
                "grad_latent_clip_scale": (
                    "FLOAT",
                    {"default": GuideConfig.grad_latent_clip_scale},
                ),
                "grad_latent_norm": (
                    "BOOLEAN",
                    {"default": GuideConfig.grad_latent_norm},
                ),
                "grad_latent_nan_to_num": (
                    "BOOLEAN",
                    {"default": GuideConfig.grad_latent_nan_to_num},
                ),
                "grad_rgb_clip": ("BOOLEAN", {"default": GuideConfig.grad_rgb_clip}),
                "grad_rgb_clip_mask_guidance": (
                    "BOOLEAN",
                    {"default": GuideConfig.grad_rgb_clip_mask_guidance},
                ),
                "grad_rgb_clip_scale": (
                    "FLOAT",
                    {"default": GuideConfig.grad_rgb_clip_scale},
                ),
                "grad_rgb_norm": ("BOOLEAN", {"default": GuideConfig.grad_rgb_norm}),
                # PGC
                "pgc_clip_rgb": ("FLOAT", {"default": GuideConfig.pgc_clip_rgb}),
                "pgc_suppress_type": (
                    "INT",
                    {"default": GuideConfig.pgc_suppress_type},
                ),
                # SDS Loss
                "lambda_guidance": ("FLOAT", {"default": GuideConfig.lambda_guidance}),
            },
        }

    RETURN_TYPES = ("GUIDECFG",)
    RETURN_NAMES = ("guide_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        text,
        text_set,
        null_text,
        negative_text,
        use_negative_text,
        negative_text_in_SBP,
        dtype,
        grad_viz,
        diffusion,
        diffusion_fp16,
        use_controlnet,
        controlnet,
        controlnet_fp16,
        controlnet_condition,
        controlnet_scale,
        lora_name,
        concept_name,
        guidance_scale,
        guidance_adjust,
        min_timestep,
        max_timestep,
        time_sampling,
        time_annealing,
        time_annealing_window,
        sds_loss_type,
        sds_weight_type,
        input_interpolate,
        grad_latent_clip,
        grad_latent_clip_scale,
        grad_latent_norm,
        grad_latent_nan_to_num,
        grad_rgb_clip,
        grad_rgb_clip_mask_guidance,
        grad_rgb_clip_scale,
        grad_rgb_norm,
        pgc_clip_rgb,
        pgc_suppress_type,
        lambda_guidance,
    ):
        # Create an instance of GuideConfig with the inputs provided by the user, or default values if not modified
        config = GuideConfig(
            text=text,
            text_set=text_set if text_set else None,
            null_text=null_text,
            negative_text=negative_text,
            use_negative_text=use_negative_text,
            negative_text_in_SBP=negative_text_in_SBP,
            dtype=dtype,
            grad_viz=grad_viz,
            diffusion=diffusion,
            diffusion_fp16=diffusion_fp16,
            use_controlnet=use_controlnet,
            controlnet=controlnet,
            controlnet_fp16=controlnet_fp16,
            controlnet_condition=controlnet_condition,
            controlnet_scale=controlnet_scale,
            lora_name=lora_name if lora_name else None,
            concept_name=concept_name if concept_name else None,
            guidance_scale=guidance_scale,
            guidance_adjust=guidance_adjust,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            time_sampling=time_sampling,
            time_annealing=time_annealing,
            time_annealing_window=time_annealing_window,
            sds_loss_type=sds_loss_type,
            sds_weight_type=sds_weight_type,
            input_interpolate=input_interpolate,
            grad_latent_clip=grad_latent_clip,
            grad_latent_clip_scale=grad_latent_clip_scale,
            grad_latent_norm=grad_latent_norm,
            grad_latent_nan_to_num=grad_latent_nan_to_num,
            grad_rgb_clip=grad_rgb_clip,
            grad_rgb_clip_mask_guidance=grad_rgb_clip_mask_guidance,
            grad_rgb_clip_scale=grad_rgb_clip_scale,
            grad_rgb_norm=grad_rgb_norm,
            pgc_clip_rgb=pgc_clip_rgb,
            pgc_suppress_type=pgc_suppress_type,
            lambda_guidance=lambda_guidance,
        )
        return (config,)


class DataConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Training resolution
                "train_w": ("STRING", {"default": str(DataConfig.train_w)}),
                "train_h": ("STRING", {"default": str(DataConfig.train_h)}),
                "grid_milestone": (
                    "STRING",
                    {
                        "default": (
                            DataConfig.grid_milestone
                            if DataConfig.grid_milestone is not None
                            else ""
                        )
                    },
                ),
                "progressive_grid": (
                    "BOOLEAN",
                    {"default": DataConfig.progressive_grid},
                ),
                # Inference resolution
                "eval_w": ("INT", {"default": DataConfig.eval_w}),
                "eval_h": ("INT", {"default": DataConfig.eval_h}),
                "test_w": ("INT", {"default": DataConfig.test_w}),
                "test_h": ("INT", {"default": DataConfig.test_h}),
                # Camera ranges
                "elevation_range": ("STRING", {"default": DataConfig.elevation_range}),
                "azimuth_range": ("STRING", {"default": DataConfig.azimuth_range}),
                # Splitting fovy_range and radius_range
                "fovy_min": ("FLOAT", {"default": DataConfig.fovy_range[0]}),
                "fovy_max": ("FLOAT", {"default": DataConfig.fovy_range[1]}),
                "radius_min": ("FLOAT", {"default": DataConfig.radius_range[0]}),
                "radius_max": ("FLOAT", {"default": DataConfig.radius_range[1]}),
                # Z clipping
                "z_near": ("FLOAT", {"default": DataConfig.z_near}),
                "z_far": ("FLOAT", {"default": DataConfig.z_far}),
                "progressive_radius": (
                    "BOOLEAN",
                    {"default": DataConfig.progressive_radius},
                ),
                "progressive_radius_ranges": (
                    "STRING",
                    {
                        "default": (
                            DataConfig.progressive_radius_ranges
                            if DataConfig.progressive_radius_ranges is not None
                            else ""
                        )
                    },
                ),
                # View batching and jitter
                "batched_view": ("BOOLEAN", {"default": DataConfig.batched_view}),
                "uniform_sphere_rate": (
                    "FLOAT",
                    {"default": DataConfig.uniform_sphere_rate},
                ),
                "jitter_pose": ("BOOLEAN", {"default": DataConfig.jitter_pose}),
                # Splitting vertical_jitter and camera_offset
                "vertical_jitter_min": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.vertical_jitter[0]
                            if DataConfig.vertical_jitter
                            else -0.5
                        )
                    },
                ),
                "vertical_jitter_max": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.vertical_jitter[1]
                            if DataConfig.vertical_jitter
                            else 0.5
                        )
                    },
                ),
                "use_human_vertical_jitter": (
                    "BOOLEAN",
                    {"default": DataConfig.use_human_vertical_jitter},
                ),
                "camera_offset_x": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.camera_offset[0]
                            if DataConfig.camera_offset
                            else 0.0
                        )
                    },
                ),
                "camera_offset_y": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.camera_offset[1]
                            if DataConfig.camera_offset
                            else 0.0
                        )
                    },
                ),
                "camera_offset_z": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.camera_offset[2]
                            if DataConfig.camera_offset
                            else 0.0
                        )
                    },
                ),
                # Eval options
                "eval_size": ("INT", {"default": DataConfig.eval_size}),
                "full_eval_size": ("INT", {"default": DataConfig.full_eval_size}),
                "eval_azimuth": ("FLOAT", {"default": DataConfig.eval_azimuth}),
                "eval_elevation": ("FLOAT", {"default": DataConfig.eval_elevation}),
                "eval_radius": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.eval_radius if DataConfig.eval_radius else 2.4
                        )
                    },
                ),
                "eval_radius_rate": ("FLOAT", {"default": DataConfig.eval_radius_rate}),
                "eval_save_video": ("BOOLEAN", {"default": DataConfig.eval_save_video}),
                "eval_save_image": ("BOOLEAN", {"default": DataConfig.eval_save_image}),
                "eval_video_fps": ("INT", {"default": DataConfig.eval_video_fps}),
                "eval_fix_animation": (
                    "BOOLEAN",
                    {"default": DataConfig.eval_fix_animation},
                ),
                "eval_camera_track": (
                    "STRING",
                    {"default": DataConfig.eval_camera_track},
                ),
                "eval_camera_offset_x": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.eval_camera_offset[0]
                            if DataConfig.eval_camera_offset
                            else 0.0
                        )
                    },
                ),
                "eval_camera_offset_y": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.eval_camera_offset[1]
                            if DataConfig.eval_camera_offset
                            else 0.0
                        )
                    },
                ),
                "eval_camera_offset_z": (
                    "FLOAT",
                    {
                        "default": (
                            DataConfig.eval_camera_offset[2]
                            if DataConfig.eval_camera_offset
                            else 0.0
                        )
                    },
                ),
                "eval_bg_mode": (
                    "STRING",
                    {
                        "default": (
                            DataConfig.eval_bg_mode if DataConfig.eval_bg_mode else ""
                        )
                    },
                ),
                "eval_body_part": (
                    "STRING",
                    {
                        "default": (
                            DataConfig.eval_body_part
                            if DataConfig.eval_body_part
                            else ""
                        )
                    },
                ),
                # Camera focus
                "body_prob": ("FLOAT", {"default": DataConfig.body_prob}),
                "head_prob": ("FLOAT", {"default": DataConfig.head_prob}),
                "face_prob": ("FLOAT", {"default": DataConfig.face_prob}),
                "hand_prob": ("FLOAT", {"default": DataConfig.hand_prob}),
                "arm_prob": ("FLOAT", {"default": DataConfig.arm_prob}),
                "foot_prob": ("FLOAT", {"default": DataConfig.foot_prob}),
                # Head camera settings
                "head_azimuth_range": (
                    "STRING",
                    {"default": DataConfig.head_azimuth_range},
                ),
                "head_elevation_range": (
                    "STRING",
                    {"default": DataConfig.head_elevation_range},
                ),
                "head_radius_min": (
                    "FLOAT",
                    {"default": DataConfig.head_radius_range[0]},
                ),
                "head_radius_max": (
                    "FLOAT",
                    {"default": DataConfig.head_radius_range[1]},
                ),
                # Face camera settings
                "face_azimuth_range": (
                    "STRING",
                    {"default": DataConfig.face_azimuth_range},
                ),
                "face_elevation_range": (
                    "STRING",
                    {"default": DataConfig.face_elevation_range},
                ),
                "face_radius_min": (
                    "FLOAT",
                    {"default": DataConfig.face_radius_range[0]},
                ),
                "face_radius_max": (
                    "FLOAT",
                    {"default": DataConfig.face_radius_range[1]},
                ),
                # Hand camera settings
                "hand_left_azimuth_range": (
                    "STRING",
                    {"default": DataConfig.hand_left_azimuth_range},
                ),
                "hand_right_azimuth_range": (
                    "STRING",
                    {"default": DataConfig.hand_right_azimuth_range},
                ),
                "hand_elevation_range": (
                    "STRING",
                    {"default": DataConfig.hand_elevation_range},
                ),
                "hand_radius_min": (
                    "FLOAT",
                    {"default": DataConfig.hand_radius_range[0]},
                ),
                "hand_radius_max": (
                    "FLOAT",
                    {"default": DataConfig.hand_radius_range[1]},
                ),
                # Foot camera settings
                "foot_left_azimuth_range": (
                    "STRING",
                    {"default": DataConfig.foot_left_azimuth_range},
                ),
                "foot_right_azimuth_range": (
                    "STRING",
                    {"default": DataConfig.foot_right_azimuth_range},
                ),
                "foot_elevation_range": (
                    "STRING",
                    {"default": DataConfig.foot_elevation_range},
                ),
                "foot_radius_min": (
                    "FLOAT",
                    {"default": DataConfig.foot_radius_range[0]},
                ),
                "foot_radius_max": (
                    "FLOAT",
                    {"default": DataConfig.foot_radius_range[1]},
                ),
                # Additional data
                "cameras": (
                    "STRING",
                    {
                        "default": (
                            DataConfig.cameras if DataConfig.cameras is not None else ""
                        )
                    },
                ),
                "random_pose_iter": ("INT", {"default": DataConfig.random_pose_iter}),
                # Objaverse
                "objaverse_id": ("STRING", {"default": DataConfig.objaverse_id}),
            },
        }

    RETURN_TYPES = ("DATACFG",)
    RETURN_NAMES = ("data_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        train_w,
        train_h,
        grid_milestone,
        progressive_grid,
        eval_w,
        eval_h,
        test_w,
        test_h,
        elevation_range,
        azimuth_range,
        fovy_min,
        fovy_max,
        radius_min,
        radius_max,
        z_near,
        z_far,
        progressive_radius,
        progressive_radius_ranges,
        batched_view,
        uniform_sphere_rate,
        jitter_pose,
        vertical_jitter_min,
        vertical_jitter_max,
        use_human_vertical_jitter,
        camera_offset_x,
        camera_offset_y,
        camera_offset_z,
        eval_size,
        full_eval_size,
        eval_azimuth,
        eval_elevation,
        eval_radius,
        eval_radius_rate,
        eval_save_video,
        eval_save_image,
        eval_video_fps,
        eval_fix_animation,
        eval_camera_track,
        eval_camera_offset_x,
        eval_camera_offset_y,
        eval_camera_offset_z,
        eval_bg_mode,
        eval_body_part,
        body_prob,
        head_prob,
        face_prob,
        hand_prob,
        arm_prob,
        foot_prob,
        head_azimuth_range,
        head_elevation_range,
        head_radius_min,
        head_radius_max,
        face_azimuth_range,
        face_elevation_range,
        face_radius_min,
        face_radius_max,
        hand_left_azimuth_range,
        hand_right_azimuth_range,
        hand_elevation_range,
        hand_radius_min,
        hand_radius_max,
        foot_left_azimuth_range,
        foot_right_azimuth_range,
        foot_elevation_range,
        foot_radius_min,
        foot_radius_max,
        cameras,
        random_pose_iter,
        objaverse_id,
    ):
        # Combine split ranges and offsets into tuples
        fovy_range = (fovy_min, fovy_max)
        radius_range = (radius_min, radius_max)
        vertical_jitter = (vertical_jitter_min, vertical_jitter_max)
        camera_offset = (camera_offset_x, camera_offset_y, camera_offset_z)
        eval_camera_offset = (
            eval_camera_offset_x,
            eval_camera_offset_y,
            eval_camera_offset_z,
        )

        # Create an instance of DataConfig with the inputs provided by the user, or default values if not modified
        config = DataConfig(
            train_w=train_w,
            train_h=train_h,
            grid_milestone=grid_milestone if grid_milestone else None,
            progressive_grid=progressive_grid,
            eval_w=eval_w,
            eval_h=eval_h,
            test_w=test_w,
            test_h=test_h,
            elevation_range=elevation_range,
            azimuth_range=azimuth_range,
            fovy_range=fovy_range,
            radius_range=radius_range,
            z_near=z_near,
            z_far=z_far,
            progressive_radius=progressive_radius,
            progressive_radius_ranges=(
                progressive_radius_ranges if progressive_radius_ranges else None
            ),
            batched_view=batched_view,
            uniform_sphere_rate=uniform_sphere_rate,
            jitter_pose=jitter_pose,
            vertical_jitter=vertical_jitter,
            use_human_vertical_jitter=use_human_vertical_jitter,
            camera_offset=camera_offset if camera_offset != (0.0, 0.0, 0.0) else None,
            eval_size=eval_size,
            full_eval_size=full_eval_size,
            eval_azimuth=eval_azimuth,
            eval_elevation=eval_elevation,
            eval_radius=eval_radius if eval_radius else None,
            eval_radius_rate=eval_radius_rate,
            eval_save_video=eval_save_video,
            eval_save_image=eval_save_image,
            eval_video_fps=eval_video_fps,
            eval_fix_animation=eval_fix_animation,
            eval_camera_track=eval_camera_track,
            eval_camera_offset=(
                eval_camera_offset if eval_camera_offset != (0.0, 0.0, 0.0) else None
            ),
            eval_bg_mode=eval_bg_mode if eval_bg_mode else None,
            eval_body_part=eval_body_part if eval_body_part else None,
            body_prob=body_prob,
            head_prob=head_prob,
            face_prob=face_prob,
            hand_prob=hand_prob,
            arm_prob=arm_prob,
            foot_prob=foot_prob,
            head_azimuth_range=head_azimuth_range,
            head_elevation_range=head_elevation_range,
            head_radius_range=(head_radius_min, head_radius_max),
            face_azimuth_range=face_azimuth_range,
            face_elevation_range=face_elevation_range,
            face_radius_range=(face_radius_min, face_radius_max),
            hand_left_azimuth_range=hand_left_azimuth_range,
            hand_right_azimuth_range=hand_right_azimuth_range,
            hand_elevation_range=hand_elevation_range,
            hand_radius_range=(hand_radius_min, hand_radius_max),
            foot_left_azimuth_range=foot_left_azimuth_range,
            foot_right_azimuth_range=foot_right_azimuth_range,
            foot_elevation_range=foot_elevation_range,
            foot_radius_range=(foot_radius_min, foot_radius_max),
            cameras=cameras if cameras else None,
            random_pose_iter=random_pose_iter,
            objaverse_id=objaverse_id,
        )
        return (config,)


class PromptConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # View-dependent Text Augmentation
                "text_augmentation": (
                    "BOOLEAN",
                    {"default": PromptConfig.text_augmentation},
                ),
                "text_augmentation_mode": (
                    "STRING",
                    {"default": PromptConfig.text_augmentation_mode},
                ),
                "angle_front": ("FLOAT", {"default": PromptConfig.angle_front}),
                "angle_overhead": ("FLOAT", {"default": PromptConfig.angle_overhead}),
                # SMPL Model
                "flat_hand_mean": ("BOOLEAN", {"default": PromptConfig.flat_hand_mean}),
                # SMPL Prompt
                "smpl_type": ("STRING", {"default": PromptConfig.smpl_type}),
                "smpl_gender": ("STRING", {"default": PromptConfig.smpl_gender}),
                "smpl_age": ("STRING", {"default": PromptConfig.smpl_age}),
                "use_smplx_2020_neutral": (
                    "BOOLEAN",
                    {"default": PromptConfig.use_smplx_2020_neutral},
                ),
                "num_person": (
                    "INT",
                    {
                        "default": (
                            PromptConfig.num_person
                            if PromptConfig.num_person is not None
                            else 0
                        )
                    },
                ),
                "scene": ("STRING", {"default": PromptConfig.scene}),
                "canonical_pose": ("STRING", {"default": PromptConfig.canonical_pose}),
                "canonical_mixup_prob": (
                    "FLOAT",
                    {"default": PromptConfig.canonical_mixup_prob},
                ),
                # SMPL Sequence
                "frame_interval": (
                    "INT",
                    {
                        "default": (
                            PromptConfig.frame_interval
                            if PromptConfig.frame_interval is not None
                            else 0
                        )
                    },
                ),
                # SMPL Shape
                "canonical_betas": (
                    "STRING",
                    {
                        "default": (
                            PromptConfig.canonical_betas
                            if PromptConfig.canonical_betas is not None
                            else ""
                        )
                    },
                ),
                "observed_betas": (
                    "STRING",
                    {
                        "default": (
                            PromptConfig.observed_betas
                            if PromptConfig.observed_betas is not None
                            else ""
                        )
                    },
                ),
                "pop_betas": ("BOOLEAN", {"default": PromptConfig.pop_betas}),
                "max_beta_iteration": (
                    "INT",
                    {"default": PromptConfig.max_beta_iteration},
                ),
                # NeRF Depth
                "nerf_depth": ("BOOLEAN", {"default": PromptConfig.nerf_depth}),
                "nerf_depth_step": ("FLOAT", {"default": PromptConfig.nerf_depth_step}),
                # Others
                "centralize_pelvis": (
                    "BOOLEAN",
                    {"default": PromptConfig.centralize_pelvis},
                ),
                "pop_transl": ("BOOLEAN", {"default": PromptConfig.pop_transl}),
                "normalize_transl": (
                    "BOOLEAN",
                    {"default": PromptConfig.normalize_transl},
                ),
                "pop_global_orient": (
                    "BOOLEAN",
                    {"default": PromptConfig.pop_global_orient},
                ),
                # Object
                "num_object": ("INT", {"default": PromptConfig.num_object}),
                # Skeleton Map
                "use_occlusion_culling": (
                    "BOOLEAN",
                    {"default": PromptConfig.use_occlusion_culling},
                ),
                "draw_body_keypoints": (
                    "BOOLEAN",
                    {"default": PromptConfig.draw_body_keypoints},
                ),
                "draw_hand_keypoints": (
                    "BOOLEAN",
                    {"default": PromptConfig.draw_hand_keypoints},
                ),
                "draw_face_landmarks": (
                    "BOOLEAN",
                    {"default": PromptConfig.draw_face_landmarks},
                ),
                "ignore_body_self_occlusion": (
                    "BOOLEAN",
                    {"default": PromptConfig.ignore_body_self_occlusion},
                ),
                "openpose_left_right_flip": (
                    "BOOLEAN",
                    {"default": PromptConfig.openpose_left_right_flip},
                ),
                "adaptive_hand_dist_thres": (
                    "FLOAT",
                    {
                        "default": (
                            PromptConfig.adaptive_hand_dist_thres
                            if PromptConfig.adaptive_hand_dist_thres is not None
                            else 0.0
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("PROMPTCFG",)
    RETURN_NAMES = ("prompt_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        text_augmentation,
        text_augmentation_mode,
        angle_front,
        angle_overhead,
        flat_hand_mean,
        smpl_type,
        smpl_gender,
        smpl_age,
        use_smplx_2020_neutral,
        num_person,
        scene,
        canonical_pose,
        canonical_mixup_prob,
        frame_interval,
        canonical_betas,
        observed_betas,
        pop_betas,
        max_beta_iteration,
        nerf_depth,
        nerf_depth_step,
        centralize_pelvis,
        pop_transl,
        normalize_transl,
        pop_global_orient,
        num_object,
        use_occlusion_culling,
        draw_body_keypoints,
        draw_hand_keypoints,
        draw_face_landmarks,
        ignore_body_self_occlusion,
        openpose_left_right_flip,
        adaptive_hand_dist_thres,
    ):
        # Create an instance of PromptConfig with the inputs provided by the user, or default values if not modified
        config = PromptConfig(
            text_augmentation=text_augmentation,
            text_augmentation_mode=text_augmentation_mode,
            angle_front=angle_front,
            angle_overhead=angle_overhead,
            flat_hand_mean=flat_hand_mean,
            smpl_type=smpl_type,
            smpl_gender=smpl_gender,
            smpl_age=smpl_age,
            use_smplx_2020_neutral=use_smplx_2020_neutral,
            num_person=num_person if num_person else None,
            scene=scene,
            canonical_pose=canonical_pose,
            canonical_mixup_prob=canonical_mixup_prob,
            frame_interval=frame_interval if frame_interval else None,
            canonical_betas=canonical_betas if canonical_betas else None,
            observed_betas=observed_betas if observed_betas else None,
            pop_betas=pop_betas,
            max_beta_iteration=max_beta_iteration,
            nerf_depth=nerf_depth,
            nerf_depth_step=nerf_depth_step,
            centralize_pelvis=centralize_pelvis,
            pop_transl=pop_transl,
            normalize_transl=normalize_transl,
            pop_global_orient=pop_global_orient,
            num_object=num_object,
            use_occlusion_culling=use_occlusion_culling,
            draw_body_keypoints=draw_body_keypoints,
            draw_hand_keypoints=draw_hand_keypoints,
            draw_face_landmarks=draw_face_landmarks,
            ignore_body_self_occlusion=ignore_body_self_occlusion,
            openpose_left_right_flip=openpose_left_right_flip,
            adaptive_hand_dist_thres=(
                adaptive_hand_dist_thres if adaptive_hand_dist_thres else None
            ),
        )
        return (config,)


class OptimConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Batch size
                "batch_size": ("INT", {"default": OptimConfig.batch_size}),
                # Seed for experiment
                "seed": ("INT", {"default": OptimConfig.seed}),
                # Total iterations
                "iters": ("INT", {"default": OptimConfig.iters}),
                # Use mixed precision (fp16)
                "fp16": ("BOOLEAN", {"default": OptimConfig.fp16}),
                # Resume from checkpoint
                "resume": ("BOOLEAN", {"default": OptimConfig.resume}),
                # Load existing model (optional checkpoint)
                "ckpt": (
                    "STRING",
                    {
                        "default": (
                            OptimConfig.ckpt if OptimConfig.ckpt is not None else ""
                        )
                    },
                ),
                "ckpt_extra": (
                    "STRING",
                    {
                        "default": (
                            OptimConfig.ckpt_extra
                            if OptimConfig.ckpt_extra is not None
                            else ""
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("OPTIMCFG",)
    RETURN_NAMES = ("optim_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        batch_size,
        seed,
        iters,
        fp16,
        resume,
        ckpt,
        ckpt_extra,
    ):
        # Create an instance of OptimConfig with the inputs provided by the user or default values if not modified
        config = OptimConfig(
            batch_size=batch_size,
            seed=seed,
            iters=iters,
            fp16=fp16,
            resume=resume,
            ckpt=ckpt if ckpt else None,
            ckpt_extra=ckpt_extra if ckpt_extra else None,
        )
        return (config,)


class LogConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Experiment name
                "exp_name": ("STRING", {"default": LogConfig.exp_name}),
                # Experiment output directory
                "exp_root": ("STRING", {"default": str(LogConfig.exp_root)}),
                # Save interval
                "save_interval": ("INT", {"default": LogConfig.save_interval}),
                # Snapshot interval
                "snapshot_interval": ("INT", {"default": LogConfig.snapshot_interval}),
                # Evaluate interval
                "evaluate_interval": ("INT", {"default": LogConfig.evaluate_interval}),
                # Run only test
                "eval_only": ("BOOLEAN", {"default": LogConfig.eval_only}),
                # Optional evaluation directory name
                "eval_dirname": (
                    "STRING",
                    {
                        "default": (
                            LogConfig.eval_dirname
                            if LogConfig.eval_dirname is not None
                            else ""
                        )
                    },
                ),
                # Resume pretrain
                "resume_pretrain": ("BOOLEAN", {"default": LogConfig.resume_pretrain}),
                # Run only pretrain
                "pretrain_only": ("BOOLEAN", {"default": LogConfig.pretrain_only}),
                # NV strain only
                "nvstrain_only": ("BOOLEAN", {"default": LogConfig.nvstrain_only}),
                # Any train only
                "anytrain_only": ("BOOLEAN", {"default": LogConfig.anytrain_only}),
                # Nerf to GS
                "nerf2gs": ("BOOLEAN", {"default": LogConfig.nerf2gs}),
                # Number of past checkpoints to keep
                "max_keep_ckpts": ("INT", {"default": LogConfig.max_keep_ckpts}),
                # Skip RGB
                "skip_rgb": ("BOOLEAN", {"default": LogConfig.skip_rgb}),
                # Debug mode
                "debug": ("BOOLEAN", {"default": LogConfig.debug}),
                # Check mode
                "check": ("BOOLEAN", {"default": LogConfig.check}),
                # Check SD
                "check_sd": ("BOOLEAN", {"default": LogConfig.check_sd}),
            }
        }

    RETURN_TYPES = ("LOGCFG",)
    RETURN_NAMES = ("log_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        exp_name,
        exp_root,
        save_interval,
        snapshot_interval,
        evaluate_interval,
        eval_only,
        eval_dirname,
        resume_pretrain,
        pretrain_only,
        nvstrain_only,
        anytrain_only,
        nerf2gs,
        max_keep_ckpts,
        skip_rgb,
        debug,
        check,
        check_sd,
    ):
        # Create an instance of LogConfig with the inputs provided by the user, or default values if not modified
        config = LogConfig(
            exp_name=exp_name,
            exp_root=Path(exp_root),
            save_interval=save_interval,
            snapshot_interval=snapshot_interval,
            evaluate_interval=evaluate_interval,
            eval_only=eval_only,
            eval_dirname=eval_dirname if eval_dirname else None,
            resume_pretrain=resume_pretrain,
            pretrain_only=pretrain_only,
            nvstrain_only=nvstrain_only,
            anytrain_only=anytrain_only,
            nerf2gs=nerf2gs,
            max_keep_ckpts=max_keep_ckpts,
            skip_rgb=skip_rgb,
            debug=debug,
            check=check,
            check_sd=check_sd,
        )
        return (config,)


class TrainConfigNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # LogConfig node
                "log_config": ("LOGCFG",),
                # RenderConfig node
                "render_config": ("RENDERCFG",),
                # NeRFConfig node
                "nerf_config": ("NERFCFG",),
                # DataConfig node
                "data_config": ("DATACFG",),
                # OptimConfig node
                "optim_config": ("OPTIMCFG",),
                # GuideConfig node
                "guide_config": ("GUIDECFG",),
                # PromptConfig node
                "prompt_config": ("PROMPTCFG",),
                # Device selection
                "device": ("STRING", {"default": TrainConfig.device}),
                # Optional character
                "character": (
                    "STRING",
                    {
                        "default": (
                            TrainConfig.character
                            if TrainConfig.character is not None
                            else ""
                        )
                    },
                ),
                # Sigma guidance
                "use_sigma_guidance": (
                    "BOOLEAN",
                    {"default": TrainConfig.use_sigma_guidance},
                ),
                "use_sigma_hand_guidance": (
                    "BOOLEAN",
                    {"default": TrainConfig.use_sigma_hand_guidance},
                ),
                "use_sigma_face_guidance": (
                    "BOOLEAN",
                    {"default": TrainConfig.use_sigma_face_guidance},
                ),
                "sigma_loss_type": ("STRING", {"default": TrainConfig.sigma_loss_type}),
                "sigma_prob": ("FLOAT", {"default": TrainConfig.sigma_prob}),
                "sigma_num_points": ("INT", {"default": TrainConfig.sigma_num_points}),
                # Sigma guidance parameters
                "sigma_surface_thickness": (
                    "FLOAT",
                    {"default": TrainConfig.sigma_surface_thickness},
                ),
                "sigma_guidance_peak": (
                    "FLOAT",
                    {"default": TrainConfig.sigma_guidance_peak},
                ),
                "sigma_noise_range": (
                    "FLOAT",
                    {"default": TrainConfig.sigma_noise_range},
                ),
                "sigma_guidance_delta": (
                    "FLOAT",
                    {"default": TrainConfig.sigma_guidance_delta},
                ),
                # Lambda sigma values
                "lambda_sigma_sigma": (
                    "FLOAT",
                    {"default": TrainConfig.lambda_sigma_sigma},
                ),
                "lambda_sigma_albedo": (
                    "FLOAT",
                    {"default": TrainConfig.lambda_sigma_albedo},
                ),
                "lambda_sigma_normal": (
                    "FLOAT",
                    {"default": TrainConfig.lambda_sigma_normal},
                ),
                # Predefined body parts
                "predefined_body_parts": (
                    "STRING",
                    {"default": TrainConfig.predefined_body_parts},
                ),
                # Stage
                "stage": ("STRING", {"default": TrainConfig.stage}),
            }
        }

    RETURN_TYPES = ("TRAINCFG",)
    RETURN_NAMES = ("train_config",)

    FUNCTION = "run"

    CATEGORY = "DreamWaltzG"

    def run(
        self,
        log_config,
        render_config,
        nerf_config,
        data_config,
        optim_config,
        guide_config,
        prompt_config,
        device,
        character,
        use_sigma_guidance,
        use_sigma_hand_guidance,
        use_sigma_face_guidance,
        sigma_loss_type,
        sigma_prob,
        sigma_num_points,
        sigma_surface_thickness,
        sigma_guidance_peak,
        sigma_noise_range,
        sigma_guidance_delta,
        lambda_sigma_sigma,
        lambda_sigma_albedo,
        lambda_sigma_normal,
        predefined_body_parts,
        stage,
    ):
        # Create an instance of TrainConfig with the inputs provided by the user or default values if not modified
        config = TrainConfig(
            log=log_config,
            render=render_config,
            nerf=nerf_config,
            data=data_config,
            optim=optim_config,
            guide=guide_config,
            prompt=prompt_config,
            device=device,
            character=character if character else None,
            use_sigma_guidance=use_sigma_guidance,
            use_sigma_hand_guidance=use_sigma_hand_guidance,
            use_sigma_face_guidance=use_sigma_face_guidance,
            sigma_loss_type=sigma_loss_type,
            sigma_prob=sigma_prob,
            sigma_num_points=sigma_num_points,
            sigma_surface_thickness=sigma_surface_thickness,
            sigma_guidance_peak=sigma_guidance_peak,
            sigma_noise_range=sigma_noise_range,
            sigma_guidance_delta=sigma_guidance_delta,
            lambda_sigma_sigma=lambda_sigma_sigma,
            lambda_sigma_albedo=lambda_sigma_albedo,
            lambda_sigma_normal=lambda_sigma_normal,
            predefined_body_parts=predefined_body_parts,
            stage=stage,
        )
        return (config,)


class DreamWaltzGStageOneTrainer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "train_config": ("TRAINCFG", {}),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )  # The output will now be the full path to the checkpoints folder
    RETURN_NAMES = ("checkpoints_folder_path",)

    FUNCTION = "init_training"

    CATEGORY = "DreamWaltzG"

    def init_training(
        self,
        train_config,
    ):
        # Fixed last checkpoint for stage one
        last_ckpt = "external/human_templates/instant-ngp/adult_neutral/"

        # Construct the experiment directory with fixed suffix
        exp_root = train_config.log.exp_root
        exp_name = f"{exp_root}/nerf,64>256,10k"

        # Set the stage 1 default parameters
        train_config.log.exp_name = exp_name
        train_config.optim.ckpt = last_ckpt
        train_config.prompt.predefined_body_parts = "hands"
        train_config.stage = "nerf"  # Set the stage to 'nerf'

        # Setting parameters based on the defaults for stage 1
        train_config.nerf.bg_mode = "gray"  # Background mode set to 'gray'
        train_config.optim.fp16 = True  # Mixed precision enabled (FP16)
        train_config.optim.iters = 10000  # Total iterations for stage 1
        train_config.prompt.scene = "canonical"  # Set prompt scene to 'canonical'

        # Training data resolution and progressive grid setup
        train_config.data.train_w = "64,128,256"  # Progressive width resolution
        train_config.data.train_h = "64,128,256"  # Progressive height resolution
        train_config.data.progressive_grid = True  # Enable progressive grid

        # Sigma guidance for optimization
        train_config.use_sigma_guidance = True  # Enable sigma guidance

        # Initialize the trainer with the modified train_config
        trainer = Trainer(train_config)

        # Run the appropriate training phase based on the configuration
        if train_config.log.eval_only:
            trainer.full_eval()
        elif train_config.log.pretrain_only:
            trainer.pretrain()
        elif train_config.log.nerf2gs:
            trainer.pretrain_nerf2gs()
        else:
            trainer.train()

        # Define the checkpoints folder path (assuming checkpoints are saved within the exp_name folder)
        checkpoints_folder_path = os.path.join(exp_name, "checkpoints")

        # Ensure the directory exists
        if not os.path.exists(checkpoints_folder_path):
            os.makedirs(checkpoints_folder_path)

        # print train_config for debugging
        print(train_config)

        # Return the checkpoints folder path (to be passed to stage 2)
        return checkpoints_folder_path


# Update the mappings for the node
NODE_CLASS_MAPPINGS = {
    "DreamWaltzGStageOneTrainer": DreamWaltzGStageOneTrainer,
    "NeRFConfigNode": NeRFConfigNode,
    "RenderConfigNode": RenderConfigNode,
    "GuideConfigNode": GuideConfigNode,
    "DataConfigNode": DataConfigNode,
    "PromptConfigNode": PromptConfigNode,
    "OptimConfigNode": OptimConfigNode,
    "LogConfigNode": LogConfigNode,
    "TrainConfigNode": TrainConfigNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamWaltzGStageOneTrainer": "DreamWaltzG Stage One Trainer",
    "NeRFConfigNode": "NeRF Configuration",
    "RenderConfigNode": "Render Configuration",
    "GuideConfigNode": "Guide Configuration",
    "DataConfigNode": "Data Configuration",
    "PromptConfigNode": "Prompt Configuration",
    "OptimConfigNode": "Optimization Configuration",
    "LogConfigNode": "Logging Configuration",
    "TrainConfigNode": "Training Configuration",
}
