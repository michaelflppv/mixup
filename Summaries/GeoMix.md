# Paper 5 - Graph Mixup on Approximate Gromov–Wasserstein Geodesics

## 1. Introduction

**Motivation:**  
- **Mixup Success in Euclidean Data:**  
  Mixup—creating synthetic training samples by linear interpolation of input–label pairs—has shown to enhance model generalization in vision and NLP.
- **Graph Data Challenges:**  
  Graphs are non-Euclidean, lie in disparate spaces, vary in size, and are often not well-aligned. These characteristics make a straightforward application of Euclidean mixup unsuitable.
- **Sample–Label Consistency Issue:**  
  Existing graph mixup approaches often transform graphs into embeddings or other aligned forms, yet many transformations are not equivalence-preserving, leading to inconsistent sample–label pairs.
  
**Proposed Solution (GEOMIX):**  
- **Core Idea:**  
  GEOMIX proposes to perform mixup along the geodesics in the Gromov–Wasserstein (GW) space. By mapping graphs into a joint space via equivalence-preserving transformations (EPT), the method defines a GW geodesic between two graphs.
- **Key Guarantee:**  
  Linear interpolation in this well-aligned GW space guarantees that the synthetic samples are consistent with their labels.
- **Efficiency:**  
  To reduce the high computational cost of exact GW geodesics, the authors introduce an accelerated algorithm that approximates the geodesics in a low-dimensional setting.

---

## 2. Preliminaries

**Notations and Graph Representation:**  
- Matrices: **A**, vectors: **s**, sets: *𝒢*, scalars: *α*.  
- A graph is represented as:  
  $$ G = (A, \mu) $$  
  where $$A \in \mathbb{R}^{n \times n}$$ (adjacency matrix) and $$\mu \in \Delta_n$$ (node weight with $$\Delta_n = \{ \mu \in \mathbb{R}_+^n \mid \sum_{i=1}^{n} \mu(i)=1 \}$$).

**Standard Graph Mixup:**  
- In Euclidean settings, mixup is defined as:  
  $$
  \begin{aligned}
  x(\lambda) &= (1-\lambda)x_1 + \lambda x_2, \\
  y(\lambda) &= (1-\lambda)y_1 + \lambda y_2.
  \end{aligned}
  $$
- For graphs, due to size and alignment issues, many methods first transform graphs with functions $$\Gamma_1, \Gamma_2$$ and then perform mixup on the transformed representations.

**Gromov–Wasserstein (GW) Space and Distance:**  
- **GW Distance:**  
  For graphs $G_1=(A_1, \mu_1)$ and $G_2=(A_2, \mu_2)$, the $p$-GW distance is defined as:
  $$
  d_{GW}(G_1, G_2) = \min_{T \in \Pi(\mu_1, \mu_2)} \left( \sum_{i,j,k,l} \left| A_1(i,j) - A_2(k,l) \right|^p \, T(i,k)\, T(j,l) \right)^{1/p}.
  $$
- **GW Geodesics:**  
  By considering an equivalence relation (i.e., $G_1 \sim G_2$ if $d_{GW}(G_1,G_2)=0$), the induced metric on the space of equivalence classes is defined as:
  $$
  d^*_{GW}(\mathcal{J}_{G_1}, \mathcal{J}_{G_2}) = d_{GW}(G_1, G_2).
  $$
- **Equivalence-Preserving Transformations:**  
  A transformation $\Gamma$ is equivalence-preserving if for all $G$ in the graph space, $G \sim \Gamma(G)$.

**Geodesic Graph Mixup Problem:**  
- **Sample–Label Consistency:**  
  A mixup sample $\tilde{G}(\lambda)$ and label $\tilde{y}(\lambda)$ are consistent if the relative distances in the GW space match the interpolation weights.
- **Formal Definition:**  
  The geodesic graph mixup problem seeks a GW geodesic:
  $$
  \gamma(\lambda) = \mathcal{J}_{\tilde{G}(\lambda)}
  $$
  connecting the equivalence classes $\mathcal{J}_{G_1}$ and $\mathcal{J}_{G_2}$ so that the interpolation is consistent with the underlying structure and labels.

---

## 3. Methodology

The methodology is divided into two major parts:

### 3.1 Mixup on Exact GW Geodesics

- **Principled Interpolation:**  
  GEOMIX generalizes Euclidean mixup to graphs by defining the mixup sample as a point along the GW geodesic between two graphs.
- **Construction:**  
  - **Graph Geodesic Representation:**  
    The geodesic is defined via:
    $$
    \begin{aligned}
    \tilde{A}(\lambda) &:= (1-\lambda) A_1 \otimes \mathbf{1}_{n_2 \times n_2} + \lambda \mathbf{1}_{n_1 \times n_1} \otimes A_2, \\
    \tilde{\mu}(\lambda) &= \operatorname{vec}\big(\operatorname{OT}(G_1, G_2)\big),
    \end{aligned}
    $$
    where $\operatorname{OT}(G_1,G_2)$ is the optimal transport coupling.
  - **Equivalence-Preserving Transformation:**  
    Linear transformations $P_1$ and $P_2$ (via Kronecker products) are applied to obtain well-aligned transformed graphs $\tilde{G}_1$ and $\tilde{G}_2$.
  - **Final Interpolation:**  
    The mixup graph and its label are obtained by:
    $$
    \begin{aligned}
    \tilde{A}(\lambda) &= (1-\lambda)\tilde{A}_1 + \lambda \tilde{A}_2, \\
    \tilde{\mu}(\lambda) &= (1-\lambda)\tilde{\mu}_1 + \lambda \tilde{\mu}_2, \\
    \tilde{y}(\lambda) &= (1-\lambda)y_1 + \lambda y_2.
    \end{aligned}
    $$

- **Guarantee:**  
  Theorem 3.1 shows that these operations yield a valid GW geodesic, ensuring that the mixup samples are consistent with their labels.

### 3.2 Accelerating Mixup on Approximate GW Geodesics

- **Challenge of High Dimensionality:**  
  Exact GW geodesics operate in a space with dimension proportional to $n_1 n_2$, which is computationally prohibitive.
- **Low-Rank Approximation:**  
  A hyperparameter $r$ (with $r \leq n$) is introduced to reduce dimensionality.
  - New variables:
    $$
    Q_1 = P_1 \operatorname{diag}(g), \quad Q_2 = P_2 \operatorname{diag}(g),
    $$
    with $g \in \Delta_r$.
- **Reformulated Optimization:**  
  The problem becomes:
  $$
  \begin{aligned}
  \min_{Q_1, Q_2, g} \quad & \left( \epsilon_{A_1,A_2}\big(Q_1\,\operatorname{diag}(1/g)\,Q_2^T\big) \right)^{\frac{1}{2}} \\
  \text{s.t.} \quad & Q_1 \in \Pi(\mu_1, g), \quad Q_2 \in \Pi(\mu_2, g), \quad g \in \Delta_r.
  \end{aligned}
  $$
- **Optimization via Mirror Descent:**  
  A mirror descent scheme with generalized KL divergence and Dykstra’s algorithm is used to update $Q_1$, $Q_2$, and $g$ iteratively.
- **Transformation Recovery and Final Output:**  
  Once convergence is reached, the transformation matrices $P_1$ and $P_2$ are recovered by column normalization of $Q_1$ and $Q_2$. The final mixup graph is then computed similarly to the exact case, but with significantly reduced computational cost.

---

## 4. Advantages and Disadvantages

| **Advantages**                                                                                                                          | **Disadvantages / Limitations**                                                                                                                     |
|-----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Principled Geodesic Interpolation:** Mixup samples are generated along the GW geodesics, ensuring sample–label consistency.           | **Computational Overhead in Exact Formulation:** Exact GW geodesics operate in very high-dimensional spaces, which is computationally expensive.    |
| **Equivalence-Preserving Transformations:** Aligns graph structures so that interpolation is meaningful in the GW space.                | **Approximation Trade-off:** The low-rank approximation introduces a hyperparameter $r$ that must be tuned to balance accuracy and efficiency.      |
| **Accelerated Computation:** The proposed mirror descent scheme with Dykstra’s algorithm reduces complexity from $O(n^4)$ to $O(n^2r)$. | **Model Complexity:** The overall method involves multiple stages (transformation, optimization, projection), increasing implementation complexity. |
| **Theoretical Guarantees:** The paper provides proofs ensuring that the interpolated samples truly lie on the GW geodesics.             | **Applicability:** Primarily validated for graph classification; extending the method to other graph tasks may require additional adaptation.       |

---

# Experiments

## 4.1 Understanding the GEOMIX Process

- **Visualization of Mixup Graphs:**  
  GEOMIX is applied on graphs generated by the Stochastic Block Model (SBM) with varying numbers of blocks (2, 3, 5). Mixup samples are generated with different values of $\lambda$ (0.2, 0.4, 0.6, 0.8), and it is observed that as $\lambda$ increases, the mixup graph becomes more similar to $G_2$.

- **Sample–Label Consistency Evaluation:**  
  The ratio 
  $$
  \frac{d_{GW}(\tilde{G}(\lambda), G_1)}{d_{GW}(\tilde{G}(\lambda), G_2)}
  $$
  is compared with 
  $$
  \frac{\|\tilde{y}(\lambda) - y_2\|}{\|\tilde{y}(\lambda) - y_1\|}.
  $$
  GEOMIX achieves a Pearson correlation coefficient of 0.91, which is significantly higher than that of G-Mixup (0.25) and FGWMixup (0.71).  
  Additionally, TSNE visualization of embeddings (using a GCN) on IMDB-B and MUTAG shows that mixup samples lie on a geodesic connecting the original graph pairs.

---

## 4.2 Generalization of GNNs

- **Experimental Setup:**  
  Graph classification is performed on five real-world datasets (PROTEINS, MUTAG, MSRC-9, IMDB-B, IMDB-M) using an 80/10/10 split and 10-fold cross validation. Two GNN models (GCN and GIN) are used as backbones.

- **Comparative Methods:**  
  GEOMIX is compared with eight state-of-the-art graph augmentation methods, including DropEdge, DropNode, Subgraph, M-Mixup, SubMix, G-Mixup, S-Mixup, and FGWMixup.

- **Results:**  
  GEOMIX achieves the best performance on 8 out of 10 settings and second-best on the remaining 2, with an improvement of up to 6.6% over the best competitor. It is particularly effective on datasets with fewer graphs and larger graph sizes (e.g., MUTAG and PROTEINS).

---

## 4.3 Robustness of GNNs

- **Experiment Setup:**  
  Robustness is evaluated under two types of corruption:
  - **Topology Corruption:** Randomly removing/adding 10% or 20% of edges.
  - **Label Corruption:** Randomly changing the labels for 10% or 20% of training graphs.

- **Results:**  
  GEOMIX outperforms competitor methods under both types of corruption. It achieves up to 4.7% improvement against topology corruption and 1.8% improvement against label corruption compared with the best competitor.

---

## 4.4 Effect of Mixup Graph Size

- **Analysis:**  
  The performance of GEOMIX is studied with varying mixup graph sizes (hyperparameter $r$).  
  - **Accuracy:** Classification accuracy fluctuates within ±3% across different values of $r$, indicating robust performance.
  - **Efficiency:** The running time increases approximately linearly with $r$.
  - **Transformation Quality:** Visualizations of the learned transformation matrices show that as $r$ increases, the matrices become sparser, approximating the exact GW geodesic solution.

---

## Summary of Experimental Interpretations

| **Aspect**                         | **Key Findings**                                                                                                                | **Interpretation**                                                                                      |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **Mixup Graph Visualization**      | GEOMIX produces mixup graphs that smoothly interpolate between $G_1$ and $G_2$ as $\lambda$ varies.                             | The method effectively captures the continuous transition (geodesic) in the joint GW space.              |
| **Sample–Label Consistency**       | Pearson coefficient of 0.91 between GW distance ratios and label differences, outperforming G-Mixup (0.25) and FGWMixup (0.71). | GEOMIX ensures that the interpolated graphs maintain consistency between the sample features and labels.  |
| **Generalization of GNNs**         | Outperforms competitors in 8/10 settings with up to 6.6% improvement, especially on datasets with fewer, larger graphs.         | GEOMIX enhances GNN generalization by providing diverse and representative augmented training samples.   |
| **Robustness to Corruption**       | Achieves up to 4.7% improvement against topology corruption and 1.8% against label corruption over best competitors.            | The geodesic mixup strategy reduces overfitting and improves robustness against noisy inputs.            |
| **Mixup Graph Size Effect**        | Accuracy remains stable (±3% fluctuation) with varying $r$; running time scales linearly with $r$.                              | The approach effectively compresses redundant/noisy graph information while maintaining essential structure. |

---

## Hyperparameters
| Hyperparameter             | Short Description                                                                  | Options / Range                                                                                                     |
|----------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Mixup Ratio ($\lambda$)    | Interpolation weight determining the contribution of each input graph            | $\lambda \in [0,1]$; typically sampled from a Beta$(k,k)$ distribution (e.g., with $k$ chosen from $\{0.5, 1, 2\}$) |
| Mixup Graph Size ($r$)     | Controls the degree of low-rank approximation for the mixup graph (i.e., output size)| $r \in \mathbb{Z}$ with $r \leq n$; in experiments, $r$ is chosen from small integers (e.g., 10, 20, …, 100)        |
| Step Size ($\gamma$)       | Step size used in mirror descent updates for the low-rank FGW optimization          | $\gamma > 0$; a small positive real number (typically selected via grid search, e.g., in the range 0.001–0.1)       |
| Number of Iterations ($T$) | Number of iterations for convergence in the mirror descent (low-rank FGW optimization) | $T \in \mathbb{N}$; chosen based on convergence criteria (set between 50 and 200 iterations in practice)            |
| Order of GW Distance ($p$) | Exponent in the GW distance calculation                                            | Typically fixed as $p = 2$                                                                                          |

# Extracted Key Information from Related Work and Appendices

---

## 1. Related Work

| **Area**                    | **Extracted Key Information**                                                                                                                                                             | **Significance for GEOMIX**                                                                                         |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Graph Data Augmentation** | - Prior approaches are categorized as feature-, structure-, and label-based methods.<br>- Many methods transform graphs to align them but often lack equivalence preservation.      | GEOMIX directly addresses the sample–label inconsistency issue by enforcing geodesic-preserving transformations.       |
| **Optimal Transport on Graphs** | - The Gromov–Wasserstein (GW) distance is used to compare and align graphs in disparate spaces.<br>- Previous works use OT for graph alignment, comparison, and representation learning. | GEOMIX leverages GW geodesics to define a principled interpolation method that naturally respects the intrinsic geometry of graphs. |
| **Graph Neural Networks (GNNs)** | - GNNs’ performance depends on high-quality training data.<br>- Data augmentation is critical to mitigate noise and overfitting.                                                         | GEOMIX enhances GNN generalization and robustness by providing consistent, geometrically sound mixup samples.            |

---

## 2. Appendices

### Appendix A: Theoretical Proofs and Derivations

| **Key Topic**                         | **Extracted Information**                                                                                                                                                                     | **Importance**                                                                                       |
|-----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Proof of Theorem 3.1**                | Demonstrates that the transformed graphs $(\tilde{G}_1$ and $\tilde{G}_2$) remain in the equivalence classes of the original graphs and that linear interpolation yields a valid GW geodesic. | Provides the theoretical guarantee that mixup samples lie on the correct geodesic, ensuring sample–label consistency. |
| **Reformulation of GW Geodesics**       | Derives an optimization formulation for GW geodesics that underpins the interpolation process in GEOMIX.                                                                                      | Establishes a rigorous mathematical framework for the geodesic mixup operation.                        |

---

### Appendix B: Algorithm Details and Pseudocode

| **Key Component**                          | **Extracted Information**                                                                                                                                                                              | **Importance**                                                                                               |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **Accelerated GEOMIX Pseudocode**             | Full pseudocode is provided for the accelerated GEOMIX algorithm.<br>- Describes mirror descent updates using a generalized KL divergence.<br>- Details the use of Dykstra’s algorithm for projection. | Enables reproducibility and deep understanding of the low-rank optimization process that accelerates GEOMIX. |
| **Variable and Constraint Reformulation**    | Introduces variables $Q_1 = P_1\,\operatorname{diag}(g)$ and $Q_2 = P_2\,\operatorname{diag}(g)$, and reformulates the problem to a low-dimensional setting controlled by $r$.                         | Reduces computational complexity from exact $O(n^4)$ to approximate $O(n^2r)$ while preserving accuracy.     |

---

### Appendix C: Additional Experimental Visualizations

| **Visualizations**                             | **Extracted Information**                                                                                                                                        | **Importance**                                                                                             |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| **TSNE and Embedding Plots**                   | Visualizations (e.g., TSNE plots) demonstrate that GEOMIX-generated samples lie along geodesics connecting original graphs (e.g., on IMDB-B and MUTAG datasets). | Provides qualitative evidence that GEOMIX preserves the geodesic property and yields consistent sample–label pairs. |
| **Transformation Matrix Visualizations**       | Shows that the learned transformation matrices become sparser with increased mixup graph size $r$, indicating convergence to the exact GW geodesic solution.     | Validates that the equivalence-preserving transformations are effective in aligning graphs for proper mixup.     |

---

### Appendix D: Comprehensive Experimental Settings

| **Aspect**                        | **Extracted Information**                                                                                                                                                       | **Importance**                                                                                      |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Dataset Descriptions**          | Detailed descriptions of the datasets used (e.g., PROTEINS, MUTAG, MSRC-9, IMDB-B, IMDB-M) including splitting strategies and data statistics.                              | Critical for understanding the experimental context and ensuring reproducibility of GEOMIX experiments. |
| **Model Architectures and Hyperparameters** | Specifications for GNN models (e.g., GCN, GIN) and hyperparameter settings used in GEOMIX and baseline methods.                                                                 | Provides insights into the experimental setup and facilitates comparison between GEOMIX and other methods. |
| **Training and Validation Protocols**       | Description of cross-validation techniques (e.g., 10-fold CV) and training procedures employed.                                                                               | Ensures fairness and consistency in the evaluation of GEOMIX performance.                            |

---

### Appendix E: Additional Analyses and Sensitivity Studies

| **Analysis Area**                         | **Extracted Information**                                                                                                                                                                                          | **Importance**                                                                                       |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Hyperparameter Sensitivity**            | Studies the impact of key hyperparameters such as mixup graph size $r$ and step size $\gamma$ on GEOMIX’s performance and efficiency.                                                                              | Guides the tuning of the method for optimal trade-offs between computational cost and accuracy.       |
| **Robustness Tests and Ablation Studies**   | Provides further robustness evaluations (e.g., under topology and label corruptions) and ablation studies that isolate the effects of different components of GEOMIX.                                              | Demonstrates the stability and robustness of GEOMIX across various conditions and reinforces its benefits. |
| **Efficiency and Time Complexity Analyses** | Presents detailed analyses showing that the accelerated GEOMIX achieves a complexity of $O(n^2r)$ compared to the much higher cost of exact GW geodesics, with experimental runtime results supporting this claim. | Validates that the low-rank approximation yields significant efficiency gains with minimal loss of effectiveness.  |
