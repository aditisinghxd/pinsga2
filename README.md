## ⚠️ Publication Status

This repository contains research code associated with an ongoing and/or planned academic publication.

The code is provided **for review and reproducibility purposes only**.  
Reuse, redistribution, or use of this code (or substantial parts of it) in other academic or commercial work **prior to formal publication is not permitted** without explicit permission from the author.


# Preference-based Multi-Objective Optimization of Land-Use Allocation using PI-NSGA-II

This repository contains the full implementation used in the Master’s thesis:

**“Preference-based Multi-Objective Optimization of Land-Use Allocation using PI-NSGA-II”**  
M.Sc. Data Science, Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)

The project implements a **spatially explicit, constrained version of PI-NSGA-II** for land-use allocation on raster maps, integrating **progressively learned stakeholder preferences** directly into the optimization process.

---

## 📌 Research Context

Land-use planning involves strong trade-offs (e.g. agricultural productivity vs. biodiversity).  
While classical multi-objective algorithms (e.g. NSGA-II) generate full Pareto fronts, real-world decision-making often requires **focusing on stakeholder-relevant regions** rather than enumerating all trade-offs.

This work extends PI-NSGA-II to:
- operate directly on spatial land-use raster maps,
- enforce policy-feasible constraints (transition rules, area bounds),
- guide convergence using **interactive or automated decision-maker preferences**.

---

## ⚙️ Key Features

- Spatial land-use optimization on raster maps
- Patch-based genome representation for scalability
- Constraint handling via:
  - land-use transition matrix
  - min–max area constraints
  - repair mutation
- Two-objective optimization:
  - Crop Yield
  - Forest Species Richness
- Comparison of:
  - Standard NSGA-II
  - Preference-guided PI-NSGA-II
- Support for:
  - Automated Decision Maker (reference-point based)
  - Manual (interactive) Decision Maker

---

## 🧠 Methodology Overview

**Workflow**
1. Load land-use raster map
2. Identify static vs. non-static cells
3. Create patch-ID map (patch-based genome)
4. Initialize population using feasible configurations
5. Run NSGA-II or PI-NSGA-II
6. Periodically elicit preferences (PI-NSGA-II)
7. Learn value function and steer evolution
8. Enforce feasibility via repair mutation
9. Analyze convergence, runtime, and solution relevance

---

## 📁 Project Structure


