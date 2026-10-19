"""
Interactive 3D visualization and real-time simulation viewer for SimplexGate-3D.

Controls:
  Left Mouse Drag : Orbit 3D camera (yaw / pitch)
  Mouse Wheel     : Zoom camera in / out (or W / S keys)
  SPACE           : Pause / resume time integration
  C               : Trigger sub-grid collapse evaluation wave
  N               : Inject rotating 3D gas nebula
  B               : Inject massive relativistic core merger
  G               : Toggle 3D gravitational potential mesh
  R               : Re-initialize 3D volume
  ESC             : Quit
"""

import os
import math
import time
from typing import Tuple, Dict, List
import torch
import numpy as np
import pygame

from core_spacetime_3d import Spacetime3DConfig, RelativisticSpacetime3DEngine, get_device
from subgrid_sph_solver import SubGridSPHConfig, Batched3DSPHCollapseSolver
from train_and_benchmark_3d import (
    SciMLBenchmarkConfig,
    UncertaintyGatedSimplexEnsemble,
    UnconstrainedMLPSurrogate,
)
from hybrid_universe_3d import Coupled3DHybridUniverse


class Live3DUniverseSandboxApp:
    def __init__(self, width: int = 1360, height: int = 840):
        pygame.init()
        pygame.display.set_caption("SimplexGate-3D | Interactive AI-Powered Relativistic Universe Sandbox")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_hud = pygame.font.SysFont("Consolas", 14)
        self.font_small = pygame.font.SysFont("Consolas", 12)

        self.device = get_device()
        self.spacetime_cfg = Spacetime3DConfig(
            num_particles=14000,
            grid_size=64,
            box_length=100.0,
            macro_dt=0.030,
            jeans_overdensity=5.2,
            collapse_search_radius=4.2,
        )
        self.sph_cfg = SubGridSPHConfig(t_gamow_henon=0.06)
        self.bench_cfg = SciMLBenchmarkConfig()

        self.sph_solver = Batched3DSPHCollapseSolver(self.sph_cfg, self.device)
        self.ensemble = UncertaintyGatedSimplexEnsemble(self.bench_cfg, self.device)
        self.mlp_unconstrained = UnconstrainedMLPSurrogate(self.bench_cfg.hidden_dim).to(self.device)

        if not os.path.exists(self.bench_cfg.weights_save_path):
            raise FileNotFoundError(
                f"Missing '{self.bench_cfg.weights_save_path}'. Run `train_and_benchmark_3d.py` first!"
            )
        ckpt = torch.load(self.bench_cfg.weights_save_path, map_location=self.device, weights_only=True)
        self.ensemble.load_state_dict(ckpt["ensemble_state"])
        self.mlp_unconstrained.load_state_dict(ckpt["mlp_state"])
        self.ensemble.eval()

        self.universe = Coupled3DHybridUniverse(
            self.spacetime_cfg, self.sph_solver, self.ensemble, self.mlp_unconstrained, self.device
        )
        self.tau_gate = 0.0020

        # 3D Camera Orbit Parameters
        self.cam_yaw = 0.65
        self.cam_pitch = 0.42
        self.cam_dist = 135.0
        self.fov_scale = 840.0
        self.show_spacetime_grid = True
        self.paused = False

        # Live Telemetry & Event Feed
        self.step_count = 0
        self.initial_mass = 0.0
        self.last_telemetry: Dict[str, float] = {}
        self.ai_fast_forwards = 0
        self.sph_fallbacks = 0
        self.saved_compute_ms = 0.0
        self.recent_events: List[Dict] = []
        self.shockwave_rings: List[Dict] = []

        self.reset_universe(seed=42)

    def reset_universe(self, seed: int = 42):
        self.universe.engine.initialize_cosmic_web_3d(seed=seed)
        self.initial_mass = self.universe.engine.mass.sum().item()
        self.step_count = 0
        self.ai_fast_forwards = 0
        self.sph_fallbacks = 0
        self.saved_compute_ms = 0.0
        self.recent_events.clear()
        self.shockwave_rings.clear()
        self.last_telemetry = self.step_stabilized_spacetime()

    @torch.no_grad()
    def step_stabilized_spacetime(self) -> Dict[str, float]:
        """
        Advances the 3D Relativistic Spacetime Engine with a Sub-Luminal Lorentz
        Velocity Limiter (|v| <= 0.35 * c) and Softened Compact Remnant Coupling
        so multi-thousand-step sandbox runs never suffer 1PN v^2 -> Inf blowups.
        """
        eng = self.universe.engine
        dt = eng.cfg.macro_dt
        v_max = 0.32 * eng.cfg.c_light  # ~14.4 code velocity units (sub-luminal 1PN domain)

        def get_safe_accel():
            rho = eng.deposit_mass_cic_3d()
            phi, gx, gy, gz = eng.solve_spacetime_metric_3d(rho)
            flat_idx, weights = eng._compute_cic_indices_and_weights(eng.pos)
            idx_flat = flat_idx.reshape(-1)

            ax_p = (gx.view(-1)[idx_flat].view(-1, 8) * weights).sum(dim=1)
            ay_p = (gy.view(-1)[idx_flat].view(-1, 8) * weights).sum(dim=1)
            az_p = (gz.view(-1)[idx_flat].view(-1, 8) * weights).sum(dim=1)
            a_newton = torch.stack([ax_p, ay_p, az_p], dim=-1)

            # Bound maximum grid acceleration near ultra-massive Black Hole voxels
            a_mag = torch.norm(a_newton, dim=-1, keepdim=True).clamp(min=1e-6)
            a_newton = a_newton * torch.clamp(45.0 / a_mag, max=1.0)

            phi_p = (phi.view(-1)[idx_flat].view(-1, 8) * weights).sum(dim=1)
            c2 = eng.cfg.c_light ** 2
            v_sq = (eng.vel ** 2).sum(dim=-1)

            metric_lapse = (1.0 + (2.0 * phi_p) / c2 - v_sq / c2).clamp(min=0.08, max=1.0)
            eng.proper_time_rate = torch.sqrt(metric_lapse)

            factor_1pn = (1.0 + v_sq / c2 + (4.0 * phi_p) / c2).clamp(min=0.6, max=1.8)
            a_dot_v = (a_newton * eng.vel).sum(dim=-1, keepdim=True)
            drag_1pn = ((4.0 / c2) * a_dot_v * eng.vel).clamp(min=-15.0, max=15.0)
            a_rel = a_newton * factor_1pn.unsqueeze(-1) - drag_1pn

            # Enforce exact zero net force across all particles
            net_f = (eng.mass.unsqueeze(-1) * a_rel).sum(dim=0)
            a_rel = a_rel - net_f / eng.mass.sum()
            return a_rel, rho, phi

        # 1. Symplectic Half-Kick
        acc_0, _, _ = get_safe_accel()
        eng.vel = eng.vel + 0.5 * dt * acc_0

        # Apply Lorentz sub-luminal speed limiter while preserving zero net momentum
        v_norm = torch.norm(eng.vel, dim=-1, keepdim=True).clamp(min=1e-6)
        eng.vel = eng.vel * torch.clamp(v_max / v_norm, max=1.0)
        eng.vel -= (eng.mass.unsqueeze(-1) * eng.vel).sum(dim=0) / eng.mass.sum()

        # 2. Full Drift on 3D Periodic Torus
        eng.pos = (eng.pos + dt * eng.vel) % eng.L

        # 3. Second Half-Kick
        acc_1, self.rho_3d, self.phi_3d = get_safe_accel()
        eng.vel = eng.vel + 0.5 * dt * acc_1

        v_norm = torch.norm(eng.vel, dim=-1, keepdim=True).clamp(min=1e-6)
        eng.vel = eng.vel * torch.clamp(v_max / v_norm, max=1.0)
        eng.vel -= (eng.mass.unsqueeze(-1) * eng.vel).sum(dim=0) / eng.mass.sum()

        # Sanitize any numerical edge cases
        eng.pos = torch.nan_to_num(eng.pos, nan=50.0) % eng.L
        eng.vel = torch.nan_to_num(eng.vel, nan=0.0)

        momentum_vec = (eng.mass.unsqueeze(-1) * eng.vel).sum(dim=0)
        return {
            "net_momentum_norm": torch.norm(momentum_vec).item(),
            "min_time_dilation": eng.proper_time_rate.min().item(),
        }

    @torch.no_grad()
    def spawn_nebula_or_merger(self, massive_bh_progenitor: bool = False):
        eng = self.universe.engine
        gas_indices = torch.nonzero(eng.particle_type == 0, as_tuple=False).squeeze(-1)
        if gas_indices.numel() < 600:
            return

        # Select diffuse background gas particles with healthy mass
        perm = torch.randperm(gas_indices.numel(), device=self.device)[:450]
        chosen = gas_indices[perm]

        L = eng.L
        center = torch.empty(3, device=self.device).uniform_(0.20 * L, 0.80 * L)
        radius_scale = 1.6 if massive_bh_progenitor else 2.8
        rel_p = torch.randn((450, 3), device=self.device) * radius_scale
        eng.pos[chosen] = (center.unsqueeze(0) + rel_p) % L

        spin_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        omega = 0.04 if massive_bh_progenitor else 0.20
        v_rot = torch.linalg.cross(spin_axis.expand_as(rel_p) * omega, rel_p, dim=-1)
        v_infall = -0.45 * rel_p / (torch.norm(rel_p, dim=-1, keepdim=True) + 0.3) if massive_bh_progenitor else 0.0
        eng.vel[chosen] = v_rot + v_infall + torch.randn((450, 3), device=self.device) * 0.18
        eng.vel -= (eng.mass.unsqueeze(-1) * eng.vel).sum(dim=0) / eng.mass.sum()

        if massive_bh_progenitor:
            self.trigger_collapse_wave(force_bh=True)

    @torch.no_grad()
    def trigger_collapse_wave(self, force_bh: bool = False):
        eng = self.universe.engine
        # Cap total compact remnants at 85 so the universe always retains rich cosmic gas nebulae
        n_compact = int((eng.particle_type > 0).sum().item())
        if n_compact > 85 and not force_bh:
            return

        clouds = eng.extract_3d_collapsing_cloud_tensors(self.rho_3d, max_clouds=4)
        if len(clouds) == 0:
            return

        # Filter out clouds whose center is already right on top of an existing Star or Black Hole
        compact_pos = eng.pos[eng.particle_type > 0]
        valid_clouds = []
        for c in clouds:
            if compact_pos.shape[0] > 0:
                dists = torch.norm(eng.minimum_image_displacement_3d(compact_pos, c["center_3d"].unsqueeze(0)), dim=-1)
                if dists.min().item() < 3.0 and not force_bh:
                    continue
            if c["z_vector"][0].item() < 250.0 and not force_bh:
                continue
            valid_clouds.append(c)

        if len(valid_clouds) == 0:
            return

        z_raw = torch.stack([c["z_vector"] for c in valid_clouds], dim=0)
        z_phys = self.universe.map_macro_cloud_to_physical_regime(z_raw)
        if force_bh:
            z_phys[0, 0] = 2650.0
            z_phys[0, 1] = 0.12
            z_phys[0, 2] = 0.025
            z_phys[0, 4] = 0.95
            z_phys[0, 5] = 0.88

        t0 = time.perf_counter()
        ens_out = self.ensemble.forward_with_uncertainty(z_phys)
        y_out = ens_out["y_mean"].clone()
        unc = ens_out["uncertainty"]
        ood_mask = unc > self.tau_gate

        n_fb = int(ood_mask.sum().item())
        n_ai = int((~ood_mask).sum().item())
        if n_fb > 0:
            y_out[ood_mask] = self.sph_solver.solve_3d_collapse_batch(z_phys[ood_mask])["y_target"]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.ai_fast_forwards += n_ai
        self.sph_fallbacks += n_fb
        self.saved_compute_ms += n_ai * 770.0

        for i, c_info in enumerate(valid_clouds):
            # Cap outward shock velocity before injecting into live sandbox for long-term orbital stability
            y_inject = y_out[i].clone()
            y_inject[4] = y_inject[4].clamp(max=8.0)
            self.universe.inject_3d_remnants_conserving_momentum(c_info, y_inject)

            u_val = float(unc[i].item())
            is_ood = bool(ood_mask[i].item())
            a_star = float(y_out[i, 0].item() * 100.0)
            a_bh = float(y_out[i, 1].item() * 100.0)
            c_pos = c_info["center_3d"].detach().cpu().tolist()

            remnant_type = "BLACK HOLE" if a_bh >= 25.0 else ("STAR" if a_star >= 15.0 else "TIDAL SHEAR")
            self.shockwave_rings.append(
                {
                    "pos": (float(c_pos[0]), float(c_pos[1]), float(c_pos[2])),
                    "radius": 0.8,
                    "max_radius": 10.5 if remnant_type == "BLACK HOLE" else 6.5,
                    "color": (255, 50, 120) if remnant_type == "BLACK HOLE" else (255, 215, 80),
                }
            )
            self.recent_events.insert(
                0,
                {
                    "step": self.step_count,
                    "type": remnant_type,
                    "u_val": u_val,
                    "routed_sph": is_ood,
                    "star_pct": a_star,
                    "bh_pct": a_bh,
                    "ms": elapsed_ms / max(len(valid_clouds), 1),
                },
            )
        self.recent_events = self.recent_events[:9]

    def project_3d_to_2d(
        self, pts_3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts_safe = np.nan_to_num(pts_3d, nan=50.0, posinf=50.0, neginf=50.0)
        L_half = 0.5 * self.spacetime_cfg.box_length
        p = pts_safe - L_half

        cy, sy = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        cp, sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)

        x1 = p[:, 0] * cy - p[:, 1] * sy
        y1 = p[:, 0] * sy + p[:, 1] * cy
        z1 = p[:, 2]

        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp

        depth = self.cam_dist + y2
        valid = depth > 5.0
        scale = self.fov_scale / np.maximum(depth, 5.0)

        cx_screen = (self.width - 360) // 2
        cy_screen = self.height // 2
        sx = np.clip(cx_screen + x1 * scale, -2000, 4000).astype(np.int32)
        sy_scr = np.clip(cy_screen - z2 * scale, -2000, 4000).astype(np.int32)
        return sx, sy_scr, depth, valid

    def render_warped_spacetime_grid(self):
        phi_2d = np.nan_to_num(self.phi_3d.min(dim=2).values.detach().cpu().numpy(), nan=0.0)
        step_g = 4
        idx_g = np.arange(0, self.spacetime_cfg.grid_size, step_g)
        G_n = len(idx_g)
        L = self.spacetime_cfg.box_length

        xs, ys = np.meshgrid(idx_g * (L / 64.0), idx_g * (L / 64.0), indexing="ij")
        phi_sub = phi_2d[np.ix_(idx_g, idx_g)]
        zs = np.clip(8.0 + phi_sub * 0.022, -28.0, 10.0)

        pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=-1)
        sx, sy, _, valid = self.project_3d_to_2d(pts)
        sx = sx.reshape(G_n, G_n)
        sy = sy.reshape(G_n, G_n)
        valid = valid.reshape(G_n, G_n)

        for i in range(G_n):
            for j in range(G_n):
                if not valid[i, j]:
                    continue
                well_intensity = int(np.clip(-phi_sub[i, j] * 0.18, 0, 180))
                col = (35 + well_intensity, 30, 85 + well_intensity // 2)
                if i + 1 < G_n and valid[i + 1, j]:
                    pygame.draw.line(self.screen, col, (sx[i, j], sy[i, j]), (sx[i + 1, j], sy[i + 1, j]), 1)
                if j + 1 < G_n and valid[i, j + 1]:
                    pygame.draw.line(self.screen, col, (sx[i, j], sy[i, j]), (sx[i, j + 1], sy[i, j + 1]), 1)

    def render_frame(self, fps: float):
        self.screen.fill((7, 9, 15))

        # 1. Draw 3D Bounding Box Wireframe
        L = self.spacetime_cfg.box_length
        corners = np.array(
            [[x, y, z] for x in (0, L) for y in (0, L) for z in (0, L)], dtype=np.float32
        )
        cx, cy, _, c_val = self.project_3d_to_2d(corners)
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for a, b in edges:
            if c_val[a] and c_val[b]:
                pygame.draw.line(self.screen, (32, 42, 65), (cx[a], cy[a]), (cx[b], cy[b]), 1)

        # 2. Draw Warped Spacetime Metric Grid
        if self.show_spacetime_grid:
            self.render_warped_spacetime_grid()

        # 3. Project & Render 3D Gas, Ignited Stars, and Relativistic Black Holes
        eng = self.universe.engine
        pos_np = eng.pos.detach().cpu().numpy()
        ptype_np = eng.particle_type.detach().cpu().numpy()
        trate_np = np.nan_to_num(eng.proper_time_rate.detach().cpu().numpy(), nan=1.0)

        sx, sy, depth, valid = self.project_3d_to_2d(pos_np)
        viewport_w = self.width - 360
        in_screen = valid & (sx >= 1) & (sx < viewport_w - 2) & (sy >= 1) & (sy < self.height - 2)

        # Draw Diffuse 3D Cosmic Gas Particles as 2x2 glowing plasma points
        gas_mask = in_screen & (ptype_np == 0)
        gas_x = sx[gas_mask]
        gas_y = sy[gas_mask]
        gas_tr = trate_np[gas_mask]

        pix_arr = pygame.surfarray.pixels3d(self.screen)
        r_col = np.clip((1.0 - gas_tr) * 340.0 + 55.0, 55, 255).astype(np.uint8)
        g_col = np.clip(gas_tr * 195.0, 65, 220).astype(np.uint8)
        b_col = np.full_like(r_col, 245, dtype=np.uint8)

        # 2x2 pixel splat so cosmic web filaments glow brightly at 1080p/1440p
        for dx_p, dy_p in ((0, 0), (1, 0), (0, 1), (1, 1)):
            pix_arr[gas_x + dx_p, gas_y + dy_p, 0] = r_col
            pix_arr[gas_x + dx_p, gas_y + dy_p, 1] = g_col
            pix_arr[gas_x + dx_p, gas_y + dy_p, 2] = b_col
        del pix_arr

        # Draw Expanding Supernova / Stellar Wind Shockwave Rings
        active_rings = []
        for ring in self.shockwave_rings:
            ring_pos_arr = np.array([ring["pos"]], dtype=np.float32)
            rx, ry, r_depth, r_val = self.project_3d_to_2d(ring_pos_arr)
            if r_val[0] and r_depth[0] > 5.0:
                scr_rad = int(ring["radius"] * self.fov_scale / r_depth[0])
                if 2 < scr_rad < 300 and 0 <= rx[0] < viewport_w and 0 <= ry[0] < self.height:
                    pygame.draw.circle(self.screen, ring["color"], (int(rx[0]), int(ry[0])), scr_rad, 1)
            ring["radius"] += 0.20
            if ring["radius"] <= ring["max_radius"]:
                active_rings.append(ring)
        self.shockwave_rings = active_rings

        # Draw Ignited Stars (Glowing Gold Cores with Outer Corona)
        star_indices = np.where(in_screen & (ptype_np == 1))[0]
        for idx in star_indices:
            px, py = int(sx[idx]), int(sy[idx])
            rad = max(3, int(1.15 * self.fov_scale / depth[idx]))
            pygame.draw.circle(self.screen, (180, 110, 20), (px, py), rad + 3)
            pygame.draw.circle(self.screen, (255, 215, 60), (px, py), rad)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), max(1, rad // 2))

        # Draw Relativistic Black Holes (Black Event Horizon + Glowing Cyan/Crimson Photon Ring)
        bh_indices = np.where(in_screen & (ptype_np == 2))[0]
        for idx in bh_indices:
            px, py = int(sx[idx]), int(sy[idx])
            rad = max(6, int(1.95 * self.fov_scale / depth[idx]))
            pygame.draw.circle(self.screen, (0, 230, 255), (px, py), rad + 5, 2)
            pygame.draw.circle(self.screen, (255, 20, 100), (px, py), rad + 2, 2)
            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), rad)

        # 4. Render Scientific Telemetry & AI Router HUD
        hud_x = self.width - 350
        pygame.draw.rect(self.screen, (14, 18, 28), (hud_x, 0, 350, self.height))
        pygame.draw.line(self.screen, (55, 75, 110), (hud_x, 0), (hud_x, self.height), 2)

        y_cursor = 18
        title_surf = self.font_title.render("SIMPLEXGATE-3D TELEMETRY", True, (0, 230, 255))
        self.screen.blit(title_surf, (hud_x + 16, y_cursor))
        y_cursor += 28

        n_gas = int((ptype_np == 0).sum())
        n_stars = int((ptype_np == 1).sum())
        n_bhs = int((ptype_np == 2).sum())
        cur_mass = eng.mass.sum().item()
        mass_drift = abs(cur_mass - self.initial_mass) / self.initial_mass * 100.0

        stats_lines = [
            f"Device        : {str(self.device).upper()} ({fps:.0f} FPS)",
            f"Spacetime Step: {self.step_count} ({'PAUSED' if self.paused else 'RUNNING'})",
            f"3D Grid Voxels: 64x64x64 (262,144)",
            f"Cosmic Gas    : {n_gas:,} particles",
            f"Ignited Stars : {n_stars} (Gold Cores)",
            f"Black Holes   : {n_bhs} (Relativistic TOV)",
            f"Min d(tau)/dt : {trate_np.min():.4f} (Time Dilation)",
            f"3D Mass Drift : {mass_drift:.6f}% (Exact)",
            f"Net Momentum  : {self.last_telemetry.get('net_momentum_norm', 0.0):.2e}",
        ]
        for line in stats_lines:
            surf = self.font_hud.render(line, True, (215, 225, 240))
            self.screen.blit(surf, (hud_x + 16, y_cursor))
            y_cursor += 20

        y_cursor += 10
        pygame.draw.line(self.screen, (45, 60, 85), (hud_x + 12, y_cursor), (self.width - 12, y_cursor), 1)
        y_cursor += 12

        router_title = self.font_title.render("EPISTEMIC AI ROUTER GATE", True, (255, 210, 60))
        self.screen.blit(router_title, (hud_x + 16, y_cursor))
        y_cursor += 24

        router_lines = [
            f"Gate Threshold tau : {self.tau_gate:.4f}",
            f"AI Fast-Forwards   : {self.ai_fast_forwards} (0.4 ms/cloud)",
            f"3D SPH Fallbacks   : {self.sph_fallbacks} (OOD Caught)",
            f"Compute Saved      : {self.saved_compute_ms/1000.0:.1f} sec",
        ]
        for line in router_lines:
            surf = self.font_hud.render(line, True, (180, 245, 195))
            self.screen.blit(surf, (hud_x + 16, y_cursor))
            y_cursor += 20

        y_cursor += 8
        log_header = self.font_hud.render("RECENT SUB-GRID COLLAPSE LOG:", True, (140, 180, 255))
        self.screen.blit(log_header, (hud_x + 16, y_cursor))
        y_cursor += 22

        for ev in self.recent_events:
            badge_col = (255, 85, 95) if ev["routed_sph"] else (70, 235, 130)
            route_str = "3D-SPH" if ev["routed_sph"] else "AI-SKIP"
            line1 = f"[{route_str}] {ev['type']} (U={ev['u_val']:.1e})"
            line2 = f"  -> Star:{ev['star_pct']:.0f}% | BH:{ev['bh_pct']:.0f}% | {ev['ms']:.1f}ms"
            s1 = self.font_small.render(line1, True, badge_col)
            s2 = self.font_small.render(line2, True, (190, 200, 215))
            self.screen.blit(s1, (hud_x + 16, y_cursor))
            self.screen.blit(s2, (hud_x + 16, y_cursor + 14))
            y_cursor += 32

        y_ctrl = self.height - 155
        pygame.draw.line(self.screen, (45, 60, 85), (hud_x + 12, y_ctrl - 8), (self.width - 12, y_ctrl - 8), 1)
        controls = [
            "[Mouse Drag] Orbit 3D Camera",
            "[Wheel/W/S]  Zoom In / Out",
            "[SPACE]      Pause / Resume",
            "[C]          Trigger AI Collapse Wave",
            "[N]          Spawn 3D Gas Nebula",
            "[B]          Spawn Black Hole Merger",
            "[G]          Toggle Spacetime Grid",
            "[R]          Reset 3D Universe",
        ]
        for c_line in controls:
            cs = self.font_small.render(c_line, True, (150, 175, 205))
            self.screen.blit(cs, (hud_x + 16, y_ctrl))
            y_ctrl += 18

        pygame.display.flip()

    def run(self):
        dragging = False
        last_mouse = (0, 0)
        running = True

        while running:
            fps = self.clock.get_fps()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_g:
                        self.show_spacetime_grid = not self.show_spacetime_grid
                    elif event.key == pygame.K_c:
                        self.trigger_collapse_wave()
                    elif event.key == pygame.K_n:
                        self.spawn_nebula_or_merger(massive_bh_progenitor=False)
                    elif event.key == pygame.K_b:
                        self.spawn_nebula_or_merger(massive_bh_progenitor=True)
                    elif event.key == pygame.K_r:
                        self.reset_universe(seed=int(time.time()) % 10000)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and event.pos[0] < self.width - 350:
                        dragging = True
                        last_mouse = event.pos
                    elif event.button == 4:
                        self.cam_dist = max(45.0, self.cam_dist - 8.0)
                    elif event.button == 5:
                        self.cam_dist = min(280.0, self.cam_dist + 8.0)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    last_mouse = event.pos
                    self.cam_yaw += dx * 0.007
                    self.cam_pitch = float(np.clip(self.cam_pitch + dy * 0.007, -1.35, 1.35))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.cam_dist = max(45.0, self.cam_dist - 1.8)
            if keys[pygame.K_s]:
                self.cam_dist = min(280.0, self.cam_dist + 1.8)

            if not self.paused:
                self.last_telemetry = self.step_stabilized_spacetime()
                self.step_count += 1
                if self.step_count % 25 == 0:
                    self.trigger_collapse_wave()
            else:
                self.cam_yaw += 0.0015

            self.render_frame(fps)
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    app = Live3DUniverseSandboxApp()
    app.run()