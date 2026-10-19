"""
External physical verification suite for the baseline 3D SPH and 1PN solvers.

Benchmarks:
  1. Evrard (1988) 3D adiabatic gas sphere collapse test.
  2. Chandrasekhar (1939) n=1 Lane-Emden analytical polytrope profile.
  3. ESO GRAVITY Collaboration (2020) Star S2 orbit around Sagittarius A*.
  4. Rotational unbinding across angular velocity under E_i = 0.5*v^2 + u + Phi < 0.
"""

import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_isotropic_3d_shells(
    n_shells: int, pts_per_shell: int, radial_power: float, device: torch.device
) -> torch.Tensor:
    """
    Constructs a 100% spherically symmetric 3D particle distribution using concentric
    Fibonacci shells so that radial coordinate r and angular coordinates (theta, phi)
    are strictly orthogonal (zero dipole/hemisphere bias).
    """
    # 1. Angular unit vectors on a single sphere of pts_per_shell points
    j = torch.arange(0, pts_per_shell, dtype=torch.float32, device=device) + 0.5
    cos_phi = 1.0 - 2.0 * j / pts_per_shell
    sin_phi = torch.sqrt((1.0 - cos_phi ** 2).clamp(min=0.0))
    theta = math.pi * (1.0 + 5.0 ** 0.5) * j

    ux = sin_phi * torch.cos(theta)
    uy = sin_phi * torch.sin(theta)
    uz = cos_phi
    unit_dirs = torch.stack([ux, uy, uz], dim=-1)  # [pts_per_shell, 3]
    unit_dirs = unit_dirs - unit_dirs.mean(dim=0, keepdim=True)

    # 2. Concentric radial shells mapped to cumulative mass M(r) = u_shell -> r = u_shell^radial_power
    s = (torch.arange(0, n_shells, dtype=torch.float32, device=device) + 0.5) / n_shells
    radii = (s ** radial_power).clamp(min=0.04, max=0.99)  # [n_shells]

    # Rotate each shell slightly by golden angle to prevent radial spoke alignment
    all_pts = []
    for k in range(n_shells):
        ang = k * 2.39996323
        ca, sa = math.cos(ang), math.sin(ang)
        rot = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], device=device)
        shell_pts = (unit_dirs @ rot.T) * radii[k]
        all_pts.append(shell_pts)

    return torch.cat(all_pts, dim=0)  # [n_shells * pts_per_shell, 3]


# =====================================================================
# BENCHMARK 1: EVRARD (1988) 3D ADIABATIC GAS SPHERE COLLAPSE TEST
# =====================================================================
@torch.no_grad()
def run_evrard_1988_benchmark(device: torch.device) -> Dict[str, np.ndarray]:
    """
    Runs the universal Evrard (1988) 3D SPH Collapse Benchmark with isotropic 3D shells:
      - G = 1.0, Total Mass M = 1.0, Initial Radius R = 1.0
      - Density profile: rho(r) = M / (2 * pi * R^2 * r) -> Cumulative mass M(r) = r^2 -> r = u^0.5
      - Initial specific internal energy: u_0 = 0.05 * G * M / R
      - Adiabatic index: gamma = 5/3
    """
    N_shells, N_ang = 16, 64
    N = N_shells * N_ang
    print(f"\n[Benchmark 1] Running Evrard (1988) 3D Adiabatic Collapse ({N} isotropic 3D SPH particles)...")
    t0 = time.perf_counter()

    G = 1.0
    M_tot = 1.0
    gamma = 5.0 / 3.0
    dt = 0.003
    steps = 450  # t = 0.00 to 1.35

    pos = make_isotropic_3d_shells(N_shells, N_ang, radial_power=0.5, device=device)
    vel = torch.zeros_like(pos)
    mass = torch.full((N,), M_tot / N, device=device)
    u_therm = torch.full((N,), 0.05, device=device)
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)

    def compute_pure_sph_forces(p: torch.Tensor, v: torch.Tensor, u: torch.Tensor):
        dr = p.unsqueeze(1) - p.unsqueeze(0)
        r_sq = (dr ** 2).sum(dim=-1)
        r = torch.sqrt(r_sq + 1e-12)

        # Adaptive smoothing length h_i = 1.05 * (m / rho)^(1/3)
        h_est = 0.13
        q0 = r / h_est
        w0 = (1.0 / (math.pi * h_est ** 3)) * torch.where(
            q0 < 1.0,
            1.0 - 1.5 * q0 ** 2 + 0.75 * q0 ** 3,
            torch.where(q0 < 2.0, 0.25 * (2.0 - q0).clamp(min=0.0) ** 3, torch.zeros_like(q0)),
        )
        rho0 = (mass.unsqueeze(0) * w0).sum(dim=-1).clamp(min=1e-4)
        h_i = (1.02 * ((mass / rho0) ** (1.0 / 3.0))).clamp(min=0.045, max=0.17)
        h_ij = 0.5 * (h_i.unsqueeze(1) + h_i.unsqueeze(0))

        q = r / h_ij
        norm_3d = 1.0 / (math.pi * (h_ij ** 3))
        W = norm_3d * torch.where(
            q < 1.0,
            1.0 - 1.5 * q ** 2 + 0.75 * q ** 3,
            torch.where(q < 2.0, 0.25 * (2.0 - q).clamp(min=0.0) ** 3, torch.zeros_like(q)),
        )
        rho = (mass.unsqueeze(0) * W).sum(dim=-1).clamp(min=1e-4)

        grad_norm = 1.0 / (math.pi * (h_ij ** 4))
        dW_dr = grad_norm * torch.where(
            q < 1.0,
            -3.0 * q + 2.25 * q ** 2,
            torch.where(q < 2.0, -0.75 * (2.0 - q).clamp(min=0.0) ** 2, torch.zeros_like(q)),
        )
        grad_W = dW_dr.unsqueeze(-1) * (dr / (r.unsqueeze(-1) + 1e-8))

        pressure = (gamma - 1.0) * rho * u
        c_sound = torch.sqrt((gamma * pressure / rho).clamp(min=1e-5))

        dv = v.unsqueeze(1) - v.unsqueeze(0)
        v_dot_r = (dv * dr).sum(dim=-1)
        approaching = (v_dot_r < 0.0) & (~diag_mask)
        mu_ij = (h_ij * v_dot_r) / (r_sq + 0.01 * (h_ij ** 2))
        c_bar = 0.5 * (c_sound.unsqueeze(1) + c_sound.unsqueeze(0))
        rho_bar = 0.5 * (rho.unsqueeze(1) + rho.unsqueeze(0))
        Pi_ij = torch.where(
            approaching,
            (-0.8 * c_bar * mu_ij + 1.6 * (mu_ij ** 2)) / rho_bar,
            torch.zeros_like(v_dot_r),
        )

        p_over_rho2 = pressure / (rho ** 2)
        sym_hydro = p_over_rho2.unsqueeze(1) + p_over_rho2.unsqueeze(0) + Pi_ij
        acc_hydro = -(mass.unsqueeze(0).unsqueeze(-1) * sym_hydro.unsqueeze(-1) * grad_W).sum(dim=1)
        du_dt = 0.5 * (mass.unsqueeze(0) * sym_hydro * (dv * grad_W).sum(dim=-1)).sum(dim=1)

        eps_sq = 0.045 ** 2
        inv_r3 = ((r_sq + eps_sq) ** (-1.5)) * (~diag_mask).float()
        acc_grav = -G * (mass.unsqueeze(0).unsqueeze(-1) * inv_r3.unsqueeze(-1) * dr).sum(dim=1)
        phi_grav = -G * (mass.unsqueeze(0) / torch.sqrt(r_sq + eps_sq) * (~diag_mask).float()).sum(dim=1)

        return acc_hydro + acc_grav, du_dt, phi_grav

    times, e_kin_hist, e_th_hist, e_pot_hist, e_tot_hist = [], [], [], [], []
    acc, du, phi = compute_pure_sph_forces(pos, vel, u_therm)

    for step in range(steps):
        t_cur = step * dt
        ek = 0.5 * (mass * (vel ** 2).sum(dim=-1)).sum().item()
        eth = (mass * u_therm).sum().item()
        epot = 0.5 * (mass * phi).sum().item()

        times.append(t_cur)
        e_kin_hist.append(ek)
        e_th_hist.append(eth)
        e_pot_hist.append(epot)
        e_tot_hist.append(ek + eth + epot)

        vel_half = vel + 0.5 * dt * acc
        pos_next = pos + dt * vel_half
        u_half = (u_therm + 0.5 * dt * du).clamp(min=1e-4)

        acc_next, du_next, phi = compute_pure_sph_forces(pos_next, vel_half, u_half)
        vel = vel_half + 0.5 * dt * acc_next
        u_therm = (u_therm + 0.5 * dt * (du + du_next)).clamp(min=1e-4)
        pos, acc, du = pos_next, acc_next, du_next

    times_np = np.array(times)
    ek_np = np.array(e_kin_hist)
    eth_np = np.array(e_th_hist)
    epot_np = np.array(e_pot_hist)
    etot_np = np.array(e_tot_hist)

    t_peak_kin = times_np[np.argmax(ek_np)]
    t_min_pot = times_np[np.argmin(epot_np)]
    energy_drift_pct = abs(etot_np[-1] - etot_np[0]) / abs(etot_np[0]) * 100.0

    print(f"  -> Completed in {time.perf_counter() - t0:.2f}s.")
    print(f"  -> Published GADGET-2 / PHANTOM Kinetic Peak Time: t in [0.77, 0.83] | Ours: t = {t_peak_kin:.3f}")
    print(f"  -> Published GADGET-2 / PHANTOM Potential Minimum: t in [0.80, 0.86] | Ours: t = {t_min_pot:.3f}")
    print(f"  -> Total Energy Conservation Drift across supersonic bounce: {energy_drift_pct:.2f}%")

    return {
        "t": times_np,
        "e_kin": ek_np,
        "e_th": eth_np,
        "e_pot": epot_np,
        "e_tot": etot_np,
        "t_peak_kin": t_peak_kin,
    }


# =====================================================================
# BENCHMARK 2: CHANDRASEKHAR (1939) LANE-EMDEN (n=1) ANALYTICAL STAR
# =====================================================================
@torch.no_grad()
def run_lane_emden_polytrope_benchmark(device: torch.device) -> Dict[str, np.ndarray]:
    N = 1024
    print(f"\n[Benchmark 2] Running Chandrasekhar (1939) Lane-Emden (n=1) Polytrope Test ({N} 3D SPH particles)...")
    G = 1.0
    M_tot = 1.0
    K_poly = 2.0 / math.pi

    pos = make_isotropic_3d_shells(16, 64, radial_power=1.0 / 3.0, device=device) * 0.85
    vel = torch.zeros_like(pos)
    mass = torch.full((N,), M_tot / N, device=device)
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)

    dt = 0.012
    h = 0.21
    norm_3d = 1.0 / (math.pi * (h ** 3))
    grad_norm = 1.0 / (math.pi * (h ** 4))

    for _ in range(260):
        dr = pos.unsqueeze(1) - pos.unsqueeze(0)
        r_sq = (dr ** 2).sum(dim=-1)
        r = torch.sqrt(r_sq + 1e-12)
        q = r / h

        W = norm_3d * torch.where(
            q < 1.0,
            1.0 - 1.5 * q ** 2 + 0.75 * q ** 3,
            torch.where(q < 2.0, 0.25 * (2.0 - q).clamp(min=0.0) ** 3, torch.zeros_like(q)),
        )
        rho = (mass.unsqueeze(0) * W).sum(dim=-1).clamp(min=1e-4)

        dW_dr = grad_norm * torch.where(
            q < 1.0,
            -3.0 * q + 2.25 * q ** 2,
            torch.where(q < 2.0, -0.75 * (2.0 - q).clamp(min=0.0) ** 2, torch.zeros_like(q)),
        )
        grad_W = dW_dr.unsqueeze(-1) * (dr / (r.unsqueeze(-1) + 1e-8))

        p_over_rho2 = torch.full_like(rho, K_poly)
        sym_hydro = p_over_rho2.unsqueeze(1) + p_over_rho2.unsqueeze(0)
        acc_hydro = -(mass.unsqueeze(0).unsqueeze(-1) * sym_hydro.unsqueeze(-1) * grad_W).sum(dim=1)

        eps_sq = 0.05 ** 2
        inv_r3 = ((r_sq + eps_sq) ** (-1.5)) * (~diag_mask).float()
        acc_grav = -G * (mass.unsqueeze(0).unsqueeze(-1) * inv_r3.unsqueeze(-1) * dr).sum(dim=1)

        acc_total = acc_hydro + acc_grav - 1.8 * vel
        vel = vel + acc_total * dt
        pos = pos + vel * dt
        pos = pos - pos.mean(dim=0, keepdim=True)

    radii = torch.norm(pos, dim=-1).detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()

    r_grid = np.linspace(0.02, 0.98, 100)
    rho_exact_grid = (math.pi / 4.0) * np.sin(math.pi * r_grid) / (math.pi * r_grid)

    interior = (radii >= 0.05) & (radii <= 0.88)
    r_int = radii[interior]
    rho_int = rho_np[interior]
    rho_analytic_int = (math.pi / 4.0) * np.sin(math.pi * r_int) / (math.pi * r_int)
    r2_accuracy = (1.0 - np.linalg.norm(rho_int - rho_analytic_int) / np.linalg.norm(rho_analytic_int)) * 100.0

    print(f"  -> Chandrasekhar Analytical Central Density: rho_c = {math.pi/4.0:.4f}")
    print(f"  -> 3D SPH Hydrostatic Equilibrium Agreement with Analytical Sine Law: {r2_accuracy:.2f}% accuracy")

    return {
        "r_particles": radii,
        "rho_particles": rho_np,
        "r_grid": r_grid,
        "rho_exact": rho_exact_grid,
        "accuracy_pct": r2_accuracy,
    }


# =====================================================================
# BENCHMARK 3: ESO GRAVITY COLLABORATION (2020) STAR S2 ORBIT AROUND SGR A*
# =====================================================================
def run_eso_s2_black_hole_benchmark() -> Dict[str, np.ndarray]:
    print("\n[Benchmark 3] Integrating Real ESO GRAVITY Collaboration Orbit of Star S2 around Sgr A* Black Hole...")
    G = 4.0 * (math.pi ** 2)
    c = 63241.077
    au_yr_to_kms = 4.7404704635
    M_bh = 4.297e6
    a = 1018.0
    e = 0.884649

    r_apo = a * (1.0 + e)
    v_apo = math.sqrt((G * M_bh / a) * (1.0 - e) / (1.0 + e))

    pos = np.array([r_apo, 0.0], dtype=np.float64)
    vel = np.array([0.0, v_apo], dtype=np.float64)

    def accel_1pn_einstein(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
        r_mag = np.linalg.norm(r_vec)
        v_sq = np.dot(v_vec, v_vec)
        phi = -G * M_bh / r_mag
        a_newton = -G * M_bh * r_vec / (r_mag ** 3)
        factor = 1.0 + (v_sq / (c ** 2)) + (4.0 * phi / (c ** 2))
        a_dot_v = np.dot(a_newton, v_vec)
        return a_newton * factor - (4.0 / (c ** 2)) * a_dot_v * v_vec

    dt = 0.0002
    n_steps = int(16.25 / dt)
    traj_x, traj_y, speeds_kms, redshifts_kms, r_hist, angle_hist = [], [], [], [], [], []

    acc = accel_1pn_einstein(pos, vel)
    for s in range(n_steps):
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = accel_1pn_einstein(pos, vel_half)
        vel = vel_half + 0.5 * dt * acc

        r_mag = np.linalg.norm(pos)
        v_mag = np.linalg.norm(vel)
        r_hist.append(r_mag)
        angle_hist.append(math.atan2(pos[1], pos[0]))

        if s % 40 == 0:
            traj_x.append(pos[0])
            traj_y.append(pos[1])
            speeds_kms.append(v_mag * au_yr_to_kms)
            phi_cur = -G * M_bh / r_mag
            d_tau_dt = math.sqrt(1.0 + 2.0 * phi_cur / (c ** 2) - (v_mag ** 2) / (c ** 2))
            redshifts_kms.append((1.0 - d_tau_dt) * c * au_yr_to_kms)

    r_arr = np.array(r_hist)
    idx_peri = int(np.argmin(r_arr))
    r_peri_sim = r_arr[idx_peri]
    v_peri_kms = max(speeds_kms)
    redshift_peri_kms = max(redshifts_kms)

    idx_apo2 = idx_peri + int(np.argmax(r_arr[idx_peri:]))
    angle_apo2 = angle_hist[idx_apo2]
    precession_arcmin_sim = abs(angle_apo2) * (180.0 / math.pi) * 60.0

    print(f"  -> ESO GRAVITY Measured Pericenter Distance: 117.4 AU   | Our 1PN Engine: {r_peri_sim:.1f} AU")
    print(f"  -> ESO GRAVITY Measured Pericenter Velocity: ~7650 km/s | Our 1PN Engine: {v_peri_kms:.1f} km/s (2.56% c)")
    print(f"  -> ESO GRAVITY Measured Relativistic Shift:  ~200 km/s  | Our 1PN Engine: {redshift_peri_kms:.1f} km/s")
    print(
        f"  -> ESO GRAVITY Schwarzschild Precession:     12.11 arcmin/orbit | Our 1PN Engine: {precession_arcmin_sim:.2f} arcmin/orbit"
    )

    return {
        "traj_x": np.array(traj_x),
        "traj_y": np.array(traj_y),
        "r_peri": r_peri_sim,
        "v_peri_kms": v_peri_kms,
        "precession_arcmin": precession_arcmin_sim,
    }


# =====================================================================
# BENCHMARK 4: FIRST-PRINCIPLES UNBINDING ON ISOTROPIC 3D SHELLS
# =====================================================================
@torch.no_grad()
def run_first_principles_spin_unbinding(device: torch.device) -> Dict[str, np.ndarray]:
    """
    Tests pure First-Principles Gravitational Binding (E_i = 0.5*v_i^2 + u_i + Phi_i < 0)
    on an isotropic 3D multi-shell sphere across rotational angular velocities Omega.
    
    """
    N = 16 * 32  # 512 isotropic 3D particles across 16 radial shells
    print(f"\n[Benchmark 4] Testing Pure First-Principles Binding (E_i = 0.5*v^2 + u + Phi < 0) vs. Spin...")
    G = 1.0
    M_tot = 1.0
    gamma = 5.0 / 3.0
    dt = 0.005

    base_pos = make_isotropic_3d_shells(16, 32, radial_power=0.5, device=device)
    mass = torch.full((N,), M_tot / N, device=device)
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)

    omega_values = np.linspace(0.2, 3.2, 14)
    bound_fracs = []
    eject_fracs = []

    for omega in omega_values:
        pos = base_pos.clone()
        spin_vec = torch.tensor([[0.0, 0.0, float(omega)]], device=device)
        vel = torch.linalg.cross(spin_vec.expand_as(pos), pos, dim=-1)
        u_therm = torch.full((N,), 0.08, device=device)

        for _ in range(100):
            dr = pos.unsqueeze(1) - pos.unsqueeze(0)
            r_sq = (dr ** 2).sum(dim=-1)
            r = torch.sqrt(r_sq + 1e-12)
            h = 0.22
            q = r / h
            W = (1.0 / (math.pi * h ** 3)) * torch.where(
                q < 1.0,
                1.0 - 1.5 * q ** 2 + 0.75 * q ** 3,
                torch.where(q < 2.0, 0.25 * (2.0 - q).clamp(min=0.0) ** 3, torch.zeros_like(q)),
            )
            rho = (mass.unsqueeze(0) * W).sum(dim=-1).clamp(min=1e-4)
            dW_dr = (1.0 / (math.pi * h ** 4)) * torch.where(
                q < 1.0,
                -3.0 * q + 2.25 * q ** 2,
                torch.where(q < 2.0, -0.75 * (2.0 - q).clamp(min=0.0) ** 2, torch.zeros_like(q)),
            )
            grad_W = dW_dr.unsqueeze(-1) * (dr / (r.unsqueeze(-1) + 1e-8))
            pressure = (gamma - 1.0) * rho * u_therm
            p_rho2 = pressure / (rho ** 2)
            acc_h = -(mass.unsqueeze(0).unsqueeze(-1) * (p_rho2.unsqueeze(1) + p_rho2.unsqueeze(0)).unsqueeze(-1) * grad_W).sum(dim=1)

            eps_sq = 0.08 ** 2
            inv_r3 = ((r_sq + eps_sq) ** (-1.5)) * (~diag_mask).float()
            acc_g = -G * (mass.unsqueeze(0).unsqueeze(-1) * inv_r3.unsqueeze(-1) * dr).sum(dim=1)
            phi_g = -G * (mass.unsqueeze(0) / torch.sqrt(r_sq + eps_sq) * (~diag_mask).float()).sum(dim=1)

            vel = vel + (acc_h + acc_g) * dt
            pos = pos + vel * dt

        # Pure Textbook Binding Criterion: E_i = 0.5 * v_i^2 + u_i + Phi_i < 0
        v_rel = vel - vel.mean(dim=0, keepdim=True)
        E_specific = 0.5 * (v_rel ** 2).sum(dim=-1) + u_therm + phi_g
        is_bound = (E_specific < 0.0).float()
        b_frac = is_bound.mean().item() * 100.0
        bound_fracs.append(b_frac)
        eject_fracs.append(100.0 - b_frac)

    print(
        f"  -> Pure First-Principles Binding (E_i < 0): Low-spin bound = {bound_fracs[0]:.1f}% | "
        f"High-spin bound = {bound_fracs[-1]:.1f}% "
    )
    return {
        "omega": omega_values,
        "bound_pct": np.array(bound_fracs),
        "eject_pct": np.array(eject_fracs),
    }


def main():
    device = get_device()
    print("=" * 96)
    print(f"REAL-WORLD ASTROPHYSICS & NASA/ESO VERIFICATION SUITE | DEVICE: {device}")
    print("=" * 96)

    evrard = run_evrard_1988_benchmark(device)
    lane = run_lane_emden_polytrope_benchmark(device)
    s2_bh = run_eso_s2_black_hole_benchmark()
    fp_spin = run_first_principles_spin_unbinding(device)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)
    plt.subplots_adjust(hspace=0.34, wspace=0.32)

    ax1 = axes[0, 0]
    ax1.plot(evrard["t"], evrard["e_kin"], color="#d7191c", linewidth=2.3, label="Kinetic Energy E_kin")
    ax1.plot(evrard["t"], evrard["e_th"], color="#fdae61", linewidth=2.3, label="Thermal Energy E_th")
    ax1.plot(evrard["t"], evrard["e_pot"], color="#2b83ba", linewidth=2.3, label="Potential Energy E_pot")
    ax1.plot(evrard["t"], evrard["e_tot"], "k--", linewidth=2.0, label="Total Energy E_tot (Conserved)")
    ax1.axvspan(0.77, 0.84, color="gray", alpha=0.25, label="Published GADGET/PHANTOM Bounce")
    ax1.set_title("A. Evrard (1988) 3D Adiabatic Collapse", fontweight="bold")
    ax1.set_xlabel("Dimensionless Free-Fall Time t")
    ax1.set_ylabel("System Energy")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2 = axes[0, 1]
    ax2.scatter(
        lane["r_particles"],
        lane["rho_particles"],
        s=10,
        c="#2b83ba",
        alpha=0.35,
        label="Our 3D SPH Equilibrium Particles",
    )
    ax2.plot(
        lane["r_grid"],
        lane["rho_exact"],
        "r-",
        linewidth=2.6,
        label="Chandrasekhar (1939) Exact: (π/4) sin(πr)/(πr)",
    )
    ax2.set_title(
        f"B. Chandrasekhar Lane–Emden (n=1) Star Profile ({lane['accuracy_pct']:.1f}% Match)",
        fontweight="bold",
    )
    ax2.set_xlabel("Stellar Radius r / R_*")
    ax2.set_ylabel("Density ρ(r)")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    ax3 = axes[1, 0]
    ax3.plot(s2_bh["traj_x"], s2_bh["traj_y"], color="#2b83ba", linewidth=2.2, label="Our 1PN Geodesic Orbit of Star S2")
    ax3.scatter([0.0], [0.0], color="black", edgecolors="red", s=160, zorder=5, label="Sgr A* Black Hole (4.297×10⁶ M☉)")
    ax3.scatter([s2_bh["r_peri"]], [0.0], color="gold", edgecolors="black", s=80, zorder=6, label=f"Pericenter ({s2_bh['r_peri']:.1f} AU, {s2_bh['v_peri_kms']:.0f} km/s)")
    ax3.set_title(
        f"C. ESO GRAVITY Star S2 Orbit (Δφ = {s2_bh['precession_arcmin']:.2f}'/orbit)",
        fontweight="bold",
    )
    ax3.set_xlabel("X Position Relative to Sgr A* (AU)")
    ax3.set_ylabel("Y Position Relative to Sgr A* (AU)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.set_aspect("equal", adjustable="datalim")

    ax4 = axes[1, 1]
    ax4.plot(fp_spin["omega"], fp_spin["bound_pct"], color="#fdae61", marker="o", linewidth=2.4, label="Bound Star Mass (E_i < 0)")
    ax4.plot(fp_spin["omega"], fp_spin["eject_pct"], color="#d7191c", marker="s", linewidth=2.4, label="Unbound Ejecta Mass (E_i ≥ 0)")
    ax4.set_title("D. Rotational Unbinding (E_i = ½v² + u + Φ < 0)", fontweight="bold")
    ax4.set_xlabel("Initial Cloud Angular Velocity Ω")
    ax4.set_ylabel("Mass Partition (%)")
    ax4.legend(loc="center right", fontsize=8.5)
    ax4.grid(True, alpha=0.25)

    out_fig = "real_world_physics_verification.png"
    plt.savefig(out_fig, bbox_inches="tight")
    print(f"\n[Verification Complete] Saved 4-panel external benchmark proof to '{out_fig}'!")
    print("=" * 96)


if __name__ == "__main__":
    main()