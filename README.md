# SimplexGate-3D

**Uncertainty-Gated Hybrid Solver Routing with Exact Simplicial Conservation for Multi-Scale Astrophysical Simulations**

**Author:** Abhinav Bhupesh Dwivedi (Department of Computer Science, University of York)  
**ORCID:** [0009-0000-8207-0990](https://orcid.org/0009-0000-8207-0990)

---

## Overview

`SimplexGate-3D` is a PyTorch framework investigating hybrid neural-numerical time-stepping in multi-scale self-gravitating systems. It couples a 3D Post-Newtonian Particle-Mesh (PM) macro-solver (`64^3` lattice, Trilinear Cloud-in-Cell assignment) with a sub-grid Henon-normalized 3D Smoothed Particle Hydrodynamics (SPH) solver.

To address conservation drift in soft-penalty Physics-Informed Neural Networks (PINNs) and extrapolation failures under Out-of-Distribution (OOD) shift, `SimplexGate-3D` combines:
1. **Analytic 3-Way Simplicial Projection:** Constrains predicted remnant mass fractions ($\alpha_{\text{star}}, \alpha_{\text{BH}}, \alpha_{\text{eject}}$) to the unit 2-simplex $\Delta^2$, preserving mass balance to 32-bit floating-point precision ($10^{-7}\%$).
2. **Epistemic Uncertainty-Gated Batching:** Evaluates ensemble variance $\mathcal{U}(\mathbf{z})$ over six dimensionless physical invariants ($M_{\mathrm{cloud}}$, $\alpha_{\mathrm{vir}}$, $\lambda_{\mathrm{spin}}$, $A_{\mathrm{triax}}$, $\eta_{\mathrm{div}}$, $C_{\mathrm{rel}}$), resolving low-variance In-Distribution (ID) collapses on the neural branch (`0.4 ms`) while batching high-variance OOD events to the 3D SPH solver on the GPU.

---

## Repository Structure

| File | Description |
|---|---|
| `core_spacetime_3d.py` | 3D Trilinear CIC Particle-Mesh solver, 7-point lattice Green's function, 1PN geodesic KDK integrator, and 6D halo invariant extractor. |
| `subgrid_sph_solver.py` | Batched 3D SPH micro-collapse solver in Henon virial units with Monaghan artificial viscosity and Buchdahl/TOV horizon partitioning. |
| `train_and_benchmark_3d.py` | Generates the 3,000-trajectory 3D SPH dataset (`dataset_3d_sph.pt`), trains baseline MLPs, Soft-PINNs, and the 5-model Simplex ensemble (`ensemble_3d_weights.pt`), and runs CUDA benchmarks. |
| `hybrid_universe_3d.py` | Coupled 40-step 3D multi-scale rollout comparing event-driven 3D SPH, unconstrained MLP, blind simplex surrogate, and `SimplexGate-3D`. |
| `verify_real_world_physics.py` | External solver verification against the Evrard (1988) adiabatic collapse test, Chandrasekhar (1939) $n=1$ Lane-Emden polytrope, and ESO GRAVITY (2020) Star S2 orbit around Sgr A*. |
| `universe_sandbox_live.py` | Interactive 60-FPS 3D visualization and real-time epistemic routing telemetry sandbox. |
| `SimplexGate_3D_Paper.tex` | LaTeX source for the accompanying manuscript (`SimplexGate_3D_Paper.pdf`). |

---

## Reproducing the Benchmarks

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run External Physical Verification
```bash
python verify_real_world_physics.py
python core_spacetime_3d.py
python subgrid_sph_solver.py
```

### 3. Train Models & Run Controlled 3D SPH Ablations
```bash
python train_and_benchmark_3d.py
```

### 4. Run Coupled 3D Universe Evolution & Interactive Sandbox
```bash
python hybrid_universe_3d.py
python universe_sandbox_live.py
```

---

## License
Released under the MIT License. See `LICENSE` for details.
