# Paper 4 - Fused Gromov-Wasserstein Graph Mixup (FGWMixup) for Graph-level Classifications

## 1. Problem Motivation and Proposed Approach

**Context & Challenges:**  
Graph Neural Networks (GNNs) excel in tasks such as molecular property prediction, social network analysis, and healthcare diagnostics. However, like other deep learning models, they suffer from data insufficiency and noise, necessitating robust data augmentation techniques.

**Existing Mixup Methods in Graphs:**  
- **ifMixup:** Performs Euclidean mixup on node features but neglects the preservation of essential graph topologies.  
  > "ifMixup [19] conducts Euclidean mixup in the graph signal space, yet fails to preserve key topologies of the original graphs."  
- **G-Mixup:** Focuses on mixing graph structures via estimated graphons but does not assign semantically meaningful graph signals.  
  > "G-Mixup [20] realizes graph structure mixup based on the estimated graphons, yet fails to assign semantically meaningful graph signals."

**Limitation Identified:**  
The independent augmentation of graph signals and structures ignores their strong interdependency. The two spaces are entangled, meaning a joint modeling approach is necessary.

**FGWMixup Proposal:**  
FGWMixup addresses this limitation by formulating graph mixup as an optimal transport (OT) problem in the Fused Gromov-Wasserstein (FGW) metric space. In doing so, it seeks a synthetic "midpoint" graph that blends both node features and structural information from two source graphs.

> **IMPORTANT**  
> FGWMixup presents an approach to graph data augmentation by jointly considering graph signal and structure spaces via the Fused Gromov-Wasserstein (FGW) metric. This method formulates the mixup problem as an optimal transport task to generate a synthetic graph that is a “midpoint” between two source graphs. A Block Coordinate Descent algorithm is employed to alternately optimize node matching (through optimal transport couplings) and update the synthetic graph, with an accelerated version (FGWMixup∗) reducing computational overhead by relaxing constraints. While the method significantly enhances GNN performance by improving generalizability and robustness, its computational complexity and current validation scope (mainly on classification tasks) highlight areas for further research and adaptation.

---

## 2. Methodological Framework

### A. Fused Gromov-Wasserstein (FGW) Distance

**Definition & Purpose:**  
The FGW distance jointly measures differences in graph signals (node features) and structures (topologies). It does so by finding an optimal coupling matrix $\pi$ that “softly” aligns nodes between two graphs.

**Key Formula:**  

$$
\text{FGW}_q(G_1, G_2) = \min_{\pi \in \Pi(\mu_1, \mu_2)} \sum_{i,j,k,l} \left[ (1-\alpha) \, d\big(x_1^{(i)}, x_2^{(j)}\big)^q + \alpha \, \big|A_1(i, k) - A_2(j, l)\big|^q \right] \pi_{i,j} \pi_{k,l}
$$

**Components:**
- $d\big(x_1^{(i)}, x_2^{(j)}\big)$: Distance between node features.
- $\big|A_1(i, k) - A_2(j, l)\big|$: Difference in structural properties.
- $\alpha$: Balancing parameter between features and structure.
- $\pi \in \Pi(\mu_1, \mu_2)$: Optimal coupling that aligns nodes.

> "FGWq(G1, G2) = min π∈Π(µ1,µ2) ∑ i,j,k,l ((1− α)d(x(1)^(i), x(2)^(j))^q + α |A1(i, k)−A2(j, l)|^q) πi,j πk,l"

### B. Graph Mixup Formulation

**Objective:**  
Generate a synthetic graph $\tilde{G} = (\tilde{\mu}, \tilde{X}, \tilde{A})$ that minimizes a weighted sum of FGW distances to two source graphs $G_1$ and $G_2$.

**Mixup Objective Formula:**  

$$
\arg \min_{\tilde{G}\in(\Delta_{n},\mathbb{R}^{n \times d},S_n(\mathbb{R}))} \left[ \lambda\, \text{FGW}(\tilde{G}, G_1) + (1-\lambda)\, \text{FGW}(\tilde{G}, G_2) \right]
$$

**Interpretation:**  
- $\lambda$ is a mixing ratio (typically drawn from a Beta distribution) that governs the contribution of each source graph.
- The synthetic graph is treated as a "midpoint" that blends both the node features and the structural attributes of the two input graphs.
- The synthetic label is also a blend: 
  $$
  y_{\tilde{G}} = \lambda\, y_{G_1} + (1-\lambda)\, y_{G_2}.
  $$

> "our objective is to solve a synthetic graph G̃ = (µ̃, X̃, Ã) of size ñ that minimizes the weighted sum of FGW distances between G̃ and two source graphs G1 = (µ1,X1,A1) and G2 = (µ2,X2,A2) respectively."

### C. Algorithmic Strategy and Acceleration

**Block Coordinate Descent (BCD) Approach:**  
- **Lower-Level Optimization:**  
  Estimate the optimal coupling matrices ($\pi_1$ and $\pi_2$) between the synthetic graph and each source graph using methods like Mirror Descent or Conditional Gradient. These couplings represent the optimal “soft” node matching.
  
- **Upper-Level Optimization:**  
  Update the synthetic graph’s structure and node features using closed-form updates:
  
  **Structure Update:**

  $$
  \tilde{A}^{(k+1)} \leftarrow \frac{1}{\tilde{\mu}\tilde{\mu}^\top} \left( \lambda\, \pi_1^{(k)} A_1 \big(\pi_1^{(k)}\big)^\top + (1-\lambda)\, \pi_2^{(k)} A_2 \big(\pi_2^{(k)}\big)^\top \right)
  $$

  **Feature Update:**

  $$
  \tilde{X}^{(k+1)} \leftarrow \lambda\, \operatorname{diag}(1/\tilde{\mu})\, \pi_1^{(k)} X_1 + (1-\lambda)\, \operatorname{diag}(1/\tilde{\mu})\, \pi_2^{(k)} X_2
  $$

> "Update Ã(k+1) ← 1/µ̃µ̃⊤ (λ π(k)_1 A1 π(k)_1⊤ + (1−λ) π(k)_2 A2 π(k)_2⊤)"

**Accelerated FGWMixup (FGWMixup∗):**  
- **Motivation:**  
  The original algorithm involves a computationally heavy triple-loop structure due to strict polytope constraints.
- **Acceleration Technique:**  
  The method relaxes the joint polytope constraint into two separate simplex constraints (one for rows and one for columns). This relaxation allows for alternating Bregman projections (with negative entropy) that simplify the optimization to a single-loop process and improve the convergence rate from $O(t^{-1})$ to $O(t^{-2})$.

> "we do not strictly project π to fit the polytope constraint ... Instead, we relax the constraint into two simplex constraints of rows and columns respectively"

---

## 3. Advantages and Disadvantages

| **Advantages**                                                                                                       | **Disadvantages / Limitations**                                                                                                             |
|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **Joint Modeling:** Integrates graph signals and structures, effectively capturing their interdependencies for more robust data augmentation. | **Computational Complexity:** The original method involves a triple-loop nested optimization, which is computationally intensive—mitigated, but not entirely removed, by the accelerated version. |
| **Optimal Transport Framework:** Utilizes the FGW distance to determine a semantically invariant "midpoint" through optimal node matching. | **Approximation Gap:** The relaxation in the accelerated FGWMixup∗ introduces a controlled feasibility gap, potentially affecting precision in some cases.           |
| **Improved Convergence:** The accelerated variant improves the convergence rate from $O(t^{-1})$ to $O(t^{-2})$, enhancing scalability.          | **Task Specificity:** Validated primarily on graph-level classification; extending the approach to other graph prediction tasks (e.g., node classification, link prediction) may require further adaptation. |
| **Enhanced GNN Performance:** Empirical results demonstrate improved generalizability and robustness across various GNN architectures.           | **Implementation Complexity:** Incorporating advanced optimal transport solvers and constraint relaxation methods increases the overall algorithmic complexity. |

---

## Experiments

### 3.1 Experimental Settings

**Datasets & Backbones:**  
The evaluation is performed on five benchmark graph classification datasets (NCI1, NCI109, PROTEINS, IMDB-B, and IMDB-M). In cases where node features are missing (IMDB-B, IMDB-M), node degree features are added. The experiments use both classic MPNN backbones (modified as vGIN and vGCN) and advanced Transformer-based GNNs (Graphormer and GraphormerGD).

**Comparison Methods:**  
Baselines include DropEdge, DropNode, M-Mixup, ifMixup, and G-Mixup. FGWMixup (and its accelerated variant FGWMixup∗) are compared against these methods using a consistent 10-fold cross-validation strategy.

### 3.2 Experimental Results

**Main Results:**  
- **Performance Gains:** FGWMixup and FGWMixup∗ consistently outperform baseline methods across almost all settings.
- **Relative Improvement:** On MPNN backbones, an average improvement of ~1.79% is observed, while on Graphormer-based backbones, the improvement is ~2.67%.
- **Graphormer Sensitivity:** Unlike other mixup methods, which sometimes degrade Graphormer performance, FGWMixup methods show substantial gains—up to 7.94% relative improvement on Graphormer.

**Robustness to Label Corruptions:**  
- **Experimental Setup:** Experiments introduce 20%, 40%, and 60% random label corruptions on IMDB-B and NCI1 (using vGCN as the backbone).
- **Outcome:** Mixup methods, particularly FGWMixup and FGWMixup∗, demonstrate stable performance and significantly outperform the in-sample augmentation (e.g., DropEdge) under noisy conditions.

**Efficiency Analysis:**  
- **Execution Time:** FGWMixup∗ shows a significant speedup (ranging from 2.03× to 3.46×) compared to the original FGWMixup across multiple datasets.
- **Infeasibility Analysis:** Using 1,000 graph pairs from PROTEINS, the relaxed single-loop FGW solver exhibits minimal deviation (MAE ≈ 0.0126 and MAPE ≈ 7.48%) from the strict solver, indicating that the approximation introduces only a negligible gap in accuracy.

### 3.3 Further Analyses

**Trade-off Coefficient (α) Sensitivity:**  
- **Observation:** Setting $\alpha = 1.0$ (i.e., falling back to the GW metric, disregarding node features) causes a significant drop in performance. The best results are consistently achieved at $\alpha = 0.95$, emphasizing the importance of structural alignment.

**Mixup Graph Sizes:**  
- **Adaptive vs. Fixed:** An adaptive strategy (weighted average of source graph sizes) yields more stable and superior performance compared to fixed graph sizes (0.5× or 2× the median). This addresses issues related to graph size distribution shifts.

**GNN Depth Effects:**  
- **Layer Variations:** Experiments with vGCN backbones (varying from 3 to 8 layers) show that both FGWMixup and FGWMixup∗ consistently boost performance across different depths, with only minor exceptions.

---

## Summary of Experimental Interpretations

| **Aspect**                     | **Key Findings**                                                                                                   | **Interpretation**                                                                                                  |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Classification Accuracy**    | FGWMixup/FGWMixup∗ outperform all baselines; ~1.79% improvement on MPNNs, ~2.67% on Graphormers; up to 7.94% gain on Graphormer. | The joint modeling of node features and graph structures via FGW effectively enhances GNN performance.            |
| **Robustness to Noise**        | Under label corruption (20%, 40%, 60%), FGWMixup methods consistently outperform methods like DropEdge.            | The soft-labeling mixup strategy reduces sensitivity to mislabeled data, leading to more robust models.             |
| **Computational Efficiency**   | FGWMixup∗ achieves speedup factors between 2.03× to 3.46× compared to FGWMixup; minimal difference in FGW estimation. | The accelerated single-loop solver, with relaxed constraints, maintains accuracy while significantly reducing computation time. |
| **Trade-off Coefficient (α)**  | Best performance is achieved at $\alpha = 0.95$; performance drops at $\alpha = 1.0$ (ignoring node features).       | Balancing structure and feature alignment is crucial, with a slight bias towards structure proving optimal.         |
| **Mixup Graph Size Strategy**  | Adaptive mixup graph sizes (weighted average of source sizes) lead to stable and superior performance over fixed sizes. | Adaptive sizing helps mitigate distributional shifts between training and testing graph sizes.                        |
| **GNN Depth Variability**      | Consistent performance improvement across various depths (3–8 layers), with few exceptions.                        | The FGWMixup approach is robust to changes in GNN model depth, suggesting broad applicability.                      |

---

## Hyperparameters
| **Hyperparameter**             | **Short Description**                                                                                                                                       | **Options / Range**                                                                                                                                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **mixup\_alpha** \($\alpha$\)  | Balances costs between node-feature alignment and structure alignment in the FGW distance.                                                                   | $\alpha \in [0,1]$. Typically tested values: $\{0.05,0.5,0.95\}$. Best empirical choice often reported as $0.95$.                                                                                            |
| **mixup\_ratio** \($\lambda$\) | The weight that blends two source graphs, determining each one’s contribution when constructing the synthetic graph.                                         | $\lambda$ is sampled from $\mathrm{Beta}(k,k)$. By default, $k=0.2$, so $\lambda \sim \mathrm{Beta}(0.2,\,0.2)$. In practice, use $\max(\lambda',\,1 - \lambda')$ to keep $\lambda \in [0.5,1]$.             |
| **Beta parameter** \($k$\)     | Controls the shape of the Beta distribution used to sample the mixup ratio \($\lambda$\). A smaller $k$ leads to more extreme $\lambda$ values, and vice versa. | $k>0.$ The paper typically sets $k=0.2$ to produce a reasonably wide spread for $\lambda$.                                                                                                                   |
| **step\_size** \($\gamma$\)    | Step size for Mirror Descent updates in the accelerated (single-loop) FGW solver \(\text{FGWMixup}^*\).                                                      | Chosen from $\{\,0.1,\,1,\,10\}$, or any small positive real.                                                                                                                                                |
| **mixup\_graph\_size\_strategy** | Determines how many nodes the synthetic (mixed) graph has; influences the synthetic graph’s size distribution.                                            | “adaptive” (default) uses weighted average of source-graph sizes: $n_{\mathrm{mixed}} = \lambda\,n_1 + (1 - \lambda)\,n_2.$ Alternatively, “fixed” strategies (e.g., median, $2 \times$ median) may be used. |
| **max\_outer\_iterations**     | Maximum iteration count for the outer loop of the Block Coordinate Descent (BCD) solver, which updates features and structure in each cycle.                 | No range provided, set to $200$.                                                                                                                                                                             |
| **max\_inner\_iterations**     | Maximum iteration count for the inner FGW solver that computes the optimal couplings between synthetic and source graphs.                                    | No range provided, set to $300$.                                                                                                                                                                             |
| **stopping threshold**         | Convergence criterion; ends the BCD loop once the relative change in the objective is below a certain threshold.                                             | No range provided, set to $5 \times 10^{-4}$.                                                                                                                                                                |
| **mixup ratio (proportion)**   | How many FGWMixup-synthesized samples vs. real samples go into a training batch or epoch (data-level mixing ratio).                                          | By default $0.25$, meaning $25\%$ of samples per epoch are synthetic, $75\%$ real.                                                                                                                           |

These hyperparameters collectively govern how FGWMixup blends graph signals and structures, how the optimal transport problem is solved, and ultimately how the synthetic graphs are generated.

### Hyperparameters used to train GNNs

| **Hyperparameter**      | **Description**                                                                                                                | **Values / Settings**                                                                                          |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **Epochs**             | The total training iterations.                                                                                                | Integer, 400 for MPNNs; 300 for Graphormers.                                                                   |
| **Optimizer**          | Optimization method used for parameter updates.                                                                               | AdamW.                                                                                                         |
| **Weight Decay**       | L2-regularization factor (applied to weights) to help prevent overfitting.                                                     | No range provided, set to $5e−4$.                                                                              |
| **Batch Size**         | Number of graphs per mini-batch during training.                                                                               | Chosen from {32, 128}, decided by a grid search.                                                               |
| **Learning Rate**      | Step size for gradient-based optimization.                                                                                     | Chosen from {1e−3, 5e−4, 1e−4}, tuned via grid search.                                                         |
| **Dropout Rate**       | Probability of dropping units to reduce overfitting in training.                                                               | No range provided, fixed at 0.5.                                                                               |
| **Model Architecture** | GNN backbone type; either Message-Passing Neural Networks (MPNNs) or Transformer-based Graphormers (depending on experiment).   | MPNNs: GCN or GIN. Graphormers: Graphormer or Graphormer-GD|

---

# Synopsis of Additional Sections

This document provides a structured synopsis of the following sections:
- **Problem Properties** and **Proofs of Theoretical Results**
- **Complete Mixup Procedure**
- **Experimental Settings** and **Further Experimental Analyses**

---

## 1. Problem Properties & Theoretical Proofs

The paper establishes essential theoretical foundations for FGWMixup by analyzing the properties of the FGW distance and proving convergence guarantees for the proposed solvers.

| **Proposition**  | **Key Statement**                                                                                                                                                               | **Interpretation**                                                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Proposition 1** | The strict FGW solver using Sinkhorn iterations converges at a rate of *O(t⁻¹)* for the joint constraint, while the relaxed solver (with separate simplex constraints) improves the convergence rate to *O(t⁻²)*. | Faster convergence in the relaxed setting implies that the accelerated FGWMixup∗ reaches an approximate optimum significantly faster while maintaining acceptable accuracy. |
| **Proposition 2** | There is a controlled, bounded gap between the optimal solutions of the strict FGW formulation and the relaxed formulation when the step size is appropriately small (γ = 1/ρ).               | The relaxation in FGWMixup∗ introduces only a minimal deviation from the ideal optimum, ensuring that the acceleration strategy remains reliable.                             |

---

## 2. Complete Mixup Procedure

The mixup procedure is detailed via two algorithmic approaches:

| **Aspect**              | **Algorithm 1 (FGWMixup)**                                                                                      | **Algorithm 2 (FGWMixup∗)**                                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Constraint Handling** | Strict projection onto the joint polytope **Π(μ₁, μ₂)** using nested Bregman projections.                       | Constraint is relaxed into two separate simplex constraints (one for rows and one for columns) with alternating Bregman projections. |
| **Loop Structure**      | Triple-loop structure: Outer loop for graph updates and an inner loop with nested iterations for the FGW solver.  | Simplified to a single-loop FGW solver within the outer loop, significantly reducing computational burden.                        |
| **Convergence Rate**    | Convergence rate of *O(t⁻¹)* due to strict constraint enforcement.                                              | Improved convergence rate of *O(t⁻²)* thanks to the relaxation of the polytope constraints.                                         |

---

## 3. Experimental Settings & Further Experimental Analyses

The experimental sections detail dataset configurations, backbone selections, and sensitivity analyses. Key empirical findings include:

| **Aspect**                  | **Details**                                                                                                                                                                    | **Implication**                                                                                                                       |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Datasets & Backbones**    | Evaluations are performed on five graph classification datasets (NCI1, NCI109, PROTEINS, IMDB-B, IMDB-M) with modified MPNNs (vGIN, vGCN) and advanced Transformer-based GNNs (Graphormer, GraphormerGD). | Demonstrates the method's broad applicability across diverse datasets and network architectures.                                      |
| **Main Performance**        | FGWMixup and FGWMixup∗ consistently outperform baselines (e.g., DropEdge, ifMixup, G-Mixup) with improvements of ~1.79% on MPNNs and ~2.67% on Graphormers; gains up to 7.94% on Graphormer in some cases. | Confirms the effectiveness of joint modeling of node features and structures using the FGW metric.                                     |
| **Robustness to Label Noise** | Under 20%, 40%, and 60% label corruption conditions, FGWMixup methods maintain stable and superior performance compared to in-sample methods like DropEdge.             | The soft-labeling mixup strategy effectively reduces sensitivity to label noise, enhancing model robustness.                          |
| **Computational Efficiency**| FGWMixup∗ achieves speedups between 2.03× and 3.46× over the original FGWMixup, with minimal FGW estimation error (MAE ≈ 0.0126, MAPE ≈ 7.48%).                           | The accelerated solver significantly improves scalability without compromising estimation accuracy.                                 |
| **Sensitivity Analyses**    | - **Trade-off coefficient (α):** Best performance at 0.95; performance drops when α = 1.0 (i.e., ignoring node features). <br> - **Mixup Graph Size:** Adaptive (weighted average) strategy outperforms fixed sizes. <br> - **GNN Depth:** Consistent improvements across network depths (3–8 layers). | Highlights the importance of balancing feature and structure alignment, adapting graph sizes to mitigate distributional shifts, and ensuring robustness across model complexities. |

