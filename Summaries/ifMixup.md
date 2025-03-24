# ifMixup: Interpolating Graph Pair to Regularize Graph Classification


| **Category**               | **Extracted Citation (Verbatim)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | **Notes / Explanation**                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Definition of GNNs**     | *"Graph Neural Networks (GNNs) (Kipf and Welling 2017) have recently shown promising performance in many challenging applications, including predicting molecule property (Wu et al. 2018), forecasting protein activation (Jiang et al. 2017), and estimating circuit functionality (Zhang, He, and Katabi 2019)."*  <br>                                                                                                                            | **GNNs** are deep learning models designed to work with graph-structured data. They use message passing to aggregate node features and capture the topology of graphs.                                                                                                 |
| **Problems of GNNs**       | *"GNNs also suffer from the data-hungry issue due to their over-parameterized learning paradigm."* <br>  <br><br> *"Regularization techniques have been actively proposed, aiming to empower the learning of GNNs while avoiding over-smoothing (Li, Han, and Wu 2018), over-squashing (Alon and Yahav 2021) and over-fitting (Zhang et al. 2021)."* <br>   | **Problems:** <br> - GNNs require large amounts of data (data-hungry) due to their high number of parameters. <br> - They face issues such as **over-smoothing** (node representations become too similar), **over-squashing** (inability to capture distant relationships), and **over-fitting**.                                                                                       |
| **Definition of Mixup (Adapted)** | *"Mixup was originally introduced by (Zhang et al. 2018a) as an interpolation-based regularizer for image classification. It regularizes the learning of deep classification models by training with synthetic samples, which are created by linearly interpolating a pair of randomly selected training samples, naturally well-aligned, as well as their training targets."* <br>                                                                                   | **Mixup:** A data augmentation technique that creates synthetic samples by linearly interpolating pairs of inputs and their labels. It has been highly effective in image and text domains and serves as the basis for the proposed graph-level method.                                                                                                                                               |
| **Pros of ifMixup for GNNs**    | *"To answer these two questions, we propose a simple input mixing strategy for Mixup on graph, coined ifMixup for graph-level classification. ifMixup first samples random graph pairs from the training data, and then creates a synthetic graph through mixing each selected sample pair, using a mixing ratio sampled from a Beta distribution."* <br>  <br><br> *"We conduct extensive experiments... showing that our strategy can effectively regularize the graph classification to improve its predictive accuracy, outperforming popular graph augmentation approaches and GNN methods."* <br> | **ifMixup Advantages:** <br> - **Simplicity:** Applies the straightforward Mixup idea to graphs by aligning graph pairs with dummy nodes and interpolating node features and edge representations. <br> - **Regularization:** Effectively increases the diversity of training data, thus reducing over-fitting and improving generalization. <br> - **Performance:** Empirical results show superior predictive accuracy compared to other graph augmentation methods. |

---

## The Proposed Method: ifMixum

### 1. Overview and Objective

> **"We here propose a simple approach, ifMixup, for generating mixed node-featured graph G̃ from a pair of training graphs GA and GB."**  

- **Note:**  
  - **Objective:** The goal is to create synthetic graphs by mixing pairs of graphs from the training set. This process is intended to regularize (i.e., improve the generalization of) Graph Neural Networks (GNNs) for graph classification tasks.

### 2. Graph Mixing Strategy

#### a. Representing a Graph for Mixing

> **"Given a node featured graph G = (v, E), we represent E as a binary matrix e with n rows and n columns, in which e(i, j) = 1 if (i, j) ∈ E, and e(i, j) = 0 otherwise."**  

- **Note:**  
  - Each graph is expressed by its node feature matrix *v* and its edge matrix *e*. The edge matrix is binary: a 1 indicates the existence of an edge, and 0 indicates no edge.

#### b. The Mixing Formulas

> **"The mixing of GA = (vA, eA) with GB = (vB, eB) to obtain G̃ = (ṽ, ẽ) can simply be done as follows:  
ẽ = λeA + (1− λ)eB,  
ṽ = λvA + (1− λ)vB."**  

- **Notes:**  
  - **Formula for Edges:** The new edge representation *ẽ* is obtained by taking a weighted (convex) combination of the two original edge matrices.  
  - **Formula for Node Features:** Similarly, the new node feature matrix *ṽ* is formed by linearly interpolating the features from both graphs.  
  - **Mixing Ratio (λ):** A scalar value in (0, 1) that determines the contribution of each graph. It is typically sampled from a Beta distribution.

#### c. Handling Graph Size Mismatch

> **"In order for the above mixing rule to be well defined, we need the two graphs to have the same number of nodes. For this purpose we define n = max(nA, nB), where nA and nB are the number of nodes in instances A and B respectively. If GA or GB has less than n nodes, we simply introduce dummy nodes to the graph and make them disconnected from the existing nodes. The feature vectors for the dummy nodes are set to the all-zero vector."**  

- **Note:**  
  - **Dummy Nodes:** To mix graphs with different numbers of nodes, dummy nodes (with zero features and no connections) are added so that both graphs reach the same size. This allows for a straightforward element-wise interpolation.

### 3. Incorporating the Mixed Graph into GNNs

Since the mixing produces edge weights that are now continuous values in the range [0, 1], the GNN must handle weighted edges. The paper explains how popular GNN architectures adapt to this:

#### a. GCN (Graph Convolutional Network)

> **"GCN handles edge weights naturally by enabling the adjacency matrix to have values between zero and one ...  
hki = σ ( Wk · (∑j∈N(i)∪{i} e(i, j) / √(d̂j d̂i) hk−1j) ),  
where d̂i = 1 + ∑j∈N(i) e(i, j)."**

- **Notes:**  
  - **GCN Update:** The GCN uses the weighted edge values during neighborhood aggregation, normalizing by the degrees (d̂) of nodes.  
  - **Non-linearity:** The function σ (often ReLU) is applied after the weighted sum.

#### b. GIN (Graph Isomorphism Network)

> **"To enable GIN to handle soft edge weight, we replace the sum operation of the isomorphism operator in GIN with a weighted sum calculation. That is,  
hki = MLPk ((1 + εk) · hk−1i + ∑j∈N(i) e(i, j) · hk−1j ),  
where εk is a learnable parameter."**  

- **Note:**  
  - **GIN Update:** In GIN, the neighborhood aggregation is modified to include the soft (weighted) edge values, ensuring that the mixing is appropriately incorporated in node representation updates.

### 4. Algorithm Pseudo-code

> **"Algorithm 1: The mixing schema in ifMixup"**

**Pseudo-code Overview:**

1. **Input:**  
   - Graph pair GA = (vA, eA) and GB = (vB, eB) (assume original edges have weight 1)  
   - Mixing ratio λ ∈ (0, 1)

2. **Compute n:**  
   - n = max(nA, nB), where nA and nB are the number of nodes in GA and GB, respectively.

3. **Add Dummy Nodes:**  
   - If a graph has fewer than n nodes, add dummy nodes (with all-zero features) until both graphs have n nodes.

4. **Mixing Process:**  
   - Compute mixed edge matrix: ẽ = λeA + (1−λ)eB  
   - Compute mixed node features: ṽ = λvA + (1−λ)vB

5. **Output:**  
   - Return the mixed graph G̃ = (ṽ, ẽ)

- **Note:**  
  - The pseudo-code clearly outlines the steps to create a synthetic graph that is a linear interpolation of two input graphs. This process is computationally light and straightforward.

### 5. Lemmas

The paper claims that the ifMixup process is **information lossless**—meaning the mixed graph contains all the information from the original graph pair.

#### a. Edge Invertibility (Lemma 0.1)

> **"Lemma 0.1 (Edge Invertibility). Let ẽ be constructed using Equation 6 with λ ≠ 0.5. ... from the mixed edge representation ẽ, we can always recover eA and eB ..."**  

- **Note:**  
  - **Key Idea:** Provided that the mixing ratio λ is not exactly 0.5, one can uniquely determine the original edge matrices (and hence the graph topology) from the mixed matrix ẽ.

#### b. Node Feature Invertibility (Lemma 0.2)

> **"Lemma 0.2 (Node Feature Invertibility). Suppose that the node feature vectors for all instances take values from a finite set V ⊂ ℝᵈ and that V is linearly independent. ... there is exactly one solution for this equation."**  

- **Note:**  
  - **Key Idea:** Under the assumption that the node features come from a linearly independent set, the original node feature matrices can be uniquely recovered from the mixed node feature matrix ṽ.

#### c. Intrusion-Freeness (Theorem 0.3)

> **"Theorem 0.3 (Intrusion-Freeness). ... Then for any mixed node-featured graph G̃ constructed using Equations 6 and 7, the two original node-featured graphs GA and GB can be uniquely recovered."**  

- **Note:**  
  - **Conclusion:** The mixing process (ifMixup) does not cause **manifold intrusion** (i.e., ambiguity where a mixed graph could represent more than one distinct pair of original graphs). This guarantees that the synthetic data preserve all original information without conflict.

---

## Experiments

### 1. Overall Graph Classification Performance

> **"Results in Table 1 show that ifMixup outperformed all the five comparison models against all the eight datasets, except for Attribute Masking on the PTC MR dataset."**

- **Interpretation:**  
  - ifMixup consistently improves accuracy compared to baseline methods (e.g., GCN with Skip Connection) on eight benchmark datasets (e.g., PTC MR, NCI109, NCI1, MUTAG, ENZYMES, PROTEINS, IMDB-M, IMDB-B).  
  - Relative improvements reach up to around 5% on some datasets.

### 2. Comparison with Other Graph Augmentation Approaches

> **"When comparing with the Mixup-like approach MixupGraph, ifMixup also obtained superior accuracy on all the eight datasets."**  

- **Interpretation:** 
  - In addition to outperforming traditional augmentation methods such as DropEdge and DropNode, ifMixup beats existing Mixup variants (e.g., MixupGraph) and attention-based methods (GAT, GATv2) on several datasets.

### 3. Sensitivity of Mixing Ratio

> **"Results in Figure 2 show that both MixupGraph and ifMixup obtained superior results on the six testing datasets with Beta(20, 1). Nevertheless, MixupGraph seemed to be very sensitive to the mixing ratio distribution, while ifMixup was robust to the five Beta distributions we tested."**  

- **Interpretation:**  
  - ifMixup is less sensitive to the choice of the mixing ratio (i.e., the Beta distribution parameters), making it more robust in practice compared to MixupGraph.

### 4. Impact of GNN Depth

> **"Results in Figure 3 show that when increasing the GCN networks from 5 layers to 8 layers, both GCN and MixupGraph seemed to degrade their performance on all the six datasets. On the contrary, ifMixup was able to increase the accuracy on all the six datasets tested."**  

- **Interpretation:**  
  - ifMixup benefits from deeper GNN architectures; while baseline GCN and other Mixup variants suffer performance drops when the network depth is increased, ifMixup's accuracy improves, indicating its ability to regularize deeper models.

### 5. Impact of Node Order Shuffling

- **Results:**  
  - In experiments using an 8-layer GCN on the NCI109 and NCI1 datasets, shuffling node order for every epoch resulted in performance degradation (e.g., lower accuracy on NCI1).
  - When the network depth was increased to 16 layers, shuffling at every epoch improved performance—on NCI1, accuracy increased from 81.4% (8 layers, with frequent shuffling) to 82.2% (16 layers, with frequent shuffling).
  
- **Interpretation:**  
  Node order permutation (shuffling) affects the input diversity. In shallower models, excessive shuffling may overwhelm the limited modeling capacity, whereas deeper models can exploit the increased variability to achieve higher predictive accuracy.

> **"Results in Table 4 show that shuffling the graph for each epoch (high frequency) may hurt the performance (e.g., for the NCI1 dataset), but ifMixup seems to be insensitive to less frequent shuffling scenarios."**  

- **Interpretation:** 
  - Changing the order of nodes before mixing (via shuffling) can increase input variety.  
  - In shallower networks (e.g., 8 layers), very frequent shuffling may hurt performance, but with deeper networks (16 layers) the negative impact is mitigated and even becomes beneficial.

### 6. Impact of Graph Size

> **"Figure 4 shows the histograms of the number of graph nodes and edges for both the original graphs and mixed graphs on NCI109. From Figure 4, one can see that ifMixup slightly shifted the node and edge distributions, but with very similar distributions overall."**  

- **Results:**  
  - Comparing different pairing strategies, the "random pair" scenario achieved the highest accuracy (approximately 0.820 ± 0.005 on NCI109 and 0.819 ± 0.004 on NCI1).
  - Other scenarios—such as mixing graphs forced to be of the same size (with or without shuffling) or self-paired graphs—resulted in noticeably lower accuracies.

- **Interpretation:**  
  - Using dummy nodes to align graph sizes does not dramatically alter the graph statistics, meaning the mixed graphs still resemble the original graphs in terms of node and edge counts.
  -   The random pairing of graphs (with the addition of dummy nodes for alignment) is more effective than enforcing same-size constraints or self-pairing. This indicates that preserving the natural variability in graph structure is beneficial for regularization and ultimately leads to improved performance.

### 7. Comparison Using Different GNN Architectures

> **"Table 3 shows that, similar to the GCN case, the ifMixup with GIN as baseline outperformed all the five comparison models against all the eight datasets, except for Attribute Masking on one dataset."**  

- **Interpretation:** 
  - ifMixup’s improvements are consistent even when using different GNN architectures such as GIN, further supporting its general effectiveness.

---

### Summary Interpretation Table

| **Aspect**                        | **Key Findings / Interpretation**                                                                                                                                                           |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Overall Accuracy**              | ifMixup consistently outperforms baseline methods (e.g., GCN, MixupGraph, DropEdge, DropNode, Attribute Masking) on eight benchmark datasets, with improvements up to ~5%.             |
| **Comparison with Other Methods** | ifMixup also outperforms attention-based models (GAT, GATv2) and other Mixup variants (MixupGraph) on multiple datasets, demonstrating superior regularization for graph classification. |
| **Mixing Ratio Sensitivity**      | ifMixup is robust to various mixing ratios (i.e., different Beta distributions), whereas MixupGraph shows higher sensitivity to these parameters.                                           |
| **GNN Depth Effect**              | ifMixup benefits from deeper network architectures (e.g., moving from 5 to 8 or 16 layers improves accuracy), unlike baselines which suffer performance drops with increased depth.      |
| **Node Order Shuffling**          | Excessive node shuffling harms performance in shallow networks; however, with increased depth (e.g., 16 layers), frequent shuffling enhances accuracy by providing greater input variety.                         |
| **Graph Size Impact**             | The process of adding dummy nodes for aligning graph sizes results in only slight shifts in node and edge distributions, preserving the overall structural characteristics of graphs.      |

---

## Hyperparameters used for ifMixup (and related methods)

| **Hyperparameter**                | **Short Description**                                         | **Options / Range**                                                                    |
|-----------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Epochs**                        | The total training iterations.                                | Integer, set to 350                                                                    |
| **Mixing Ratio (λ)**              | Weight for linear interpolation between two graphs            | Sampled from Beta: Beta(1,1), Beta(2,2), Beta(5,1), Beta(10,1), Beta(20,1) (λ ∈ (0,1)) |
| **Initial Learning Rate**         | Starting step size for the optimizer                          | {0.01, 0.0005}                                                                         |
| **Hidden Unit Size**              | Dimensionality of hidden layers in the GNN                    | {64, 128}                                                                              |
| **Batch Size**                    | Number of graph samples per training batch                    | {32, 128}                                                                              |
| **Dropout Ratio (after dense layer)** | Probability of dropping units in the dense layer              | {0, 0.5}                                                                               |
| **DropNode / DropEdge Ratio**     | Percentage of nodes/edges randomly removed for augmentation   | {20%, 40%}                                                                             |
| **Number of GNN Layers**          | Depth of the Graph Neural Network                             | {3, 5, 8}                                                                              |