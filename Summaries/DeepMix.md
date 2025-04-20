# Paper 8 - DeepMix: Mixup for Node and Graph Classification

## 1. Introduction  
- **Problem**: GNNs overfit on limited graph data due to high capacity.  
- **Existing augmentation**: DropEdge randomly deletes edges, assuming node labels unchanged, only models intra‑class vicinity.  
- **Mixup idea**: In images, Mixup synthesizes $(\tilde x,\tilde y)$ by  
  $$\tilde x=\lambda x_i+(1-\lambda)x_j,\quad \tilde y=\lambda y_i+(1-\lambda)y_j,\quad \lambda\sim\mathrm{Beta}(\alpha,\alpha)\,. $$  
- **Graph challenges**: Irregular node placement and connectivity make topology interpolation non‑trivial; simultaneous Mixup on multiple node pairs interferes via message‑passing.  
- **Contributions**:  
  1. Two-branch Mixup convolution to mix topology.  
  2. Two-stage Mixup framework to avoid cross‑pair interference.  
  3. Semantic‑space Mixup for whole‑graph interpolation.  

---

## 2. Related Work  
- **Node classification**: GCN, GAT, etc.; DropEdge assumes labels invariant.  
- **Graph classification**: GIN, DiffPool, WL‑kernel, etc.  
- **Data augmentation**: Mixup in vision/NLP; graph‑specific heuristics (random edge/node deletion).  
- **Gap**: No principled Mixup for graphs handling both topology and cross‑node interference.  

---

## 3. Methodology  

### 3.1. Two‑Branch Mixup Convolution  
- To Mixup nodes $i,j$, mix their $L$‑hop neighborhoods via:  
  1. **Feature interpolation**:  
     $$\tilde x_{ij}=\lambda x_i + (1 - \lambda)x_j\,. $$  
  2. **Dual aggregations** at each layer $\ell$:  
     $$
     \begin{aligned}
     h^{(\ell)}_{ij,i} &=\mathrm{AGG}^{(\ell)}\bigl(h^{(\ell-1)}_{ij},\{h^{(\ell-1)}_k:k\in N(i)\}\bigr)\,, \\
     h^{(\ell)}_{ij,j} &=\mathrm{AGG}^{(\ell)}\bigl(h^{(\ell-1)}_{ij},\{h^{(\ell-1)}_k:k\in N(j)\}\bigr)\,, \\
     h^{(\ell)}_{ij} &=\lambda\,h^{(\ell)}_{ij,i}+(1-\lambda)\,h^{(\ell)}_{ij,j}\,,
     \end{aligned}
     $$
     with $h^{(0)}_{ij}=\tilde x_{ij}$.  

### 3.2. Two‑Stage Mixup Framework  
- **Issue**: Naive Mixup pollutes neighbor features across pairs.  
- **Solution**:  
  1. **Stage 1**: Run GNN on original features to compute all $h^{(\ell)}_k$.  
  2. **Stage 2**: For each sampled pair $(i,j)$, interpolate features, then apply two‑branch Mixup convolution, but always use neighbors’ Stage 1 representations when aggregating.  
- **Result**: Isolates each pair’s Mixup; no cross‑pair interference.  

### 3.3. Mixup for Graph Classification  
- Given graph embeddings $h_{G_1},h_{G_2}$ and one‑hot labels $y_{G_1},y_{G_2}$, interpolate:  
  $$\tilde h_{G}=\lambda\,h_{G_1}+(1-\lambda)\,h_{G_2},\quad \tilde y=\lambda\,y_{G_1}+(1-\lambda)\,y_{G_2}\,. $$  

### 3.4. Complexity  
- Both node‑ and graph‑Mixup retain baseline GNN complexity $O\bigl(|E|\sum d_\ell + |V|\sum d_{\ell-1}d_\ell\bigr)$.  

---

## 4. Experiments  

### 4.1. Setup  
- **Node datasets**: Cora, Citeseer, Pubmed, Flickr, Yelp, Amazon.  
- **Graph datasets**: D&D, NCI1, PROTEINS, COLLAB, IMDB‑M, REDDIT‑5K.  
- **Baselines**: GCN, GAT, JKNet, LGCN, GMNN, ResGCN, DropEdge; GraphSAGE, GraphSAINT.  
- **Graph classification baselines**: GRAPHLET, WL, GCN, DGCNN, DiffPool, EigenPool, GIN.  
- $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$, default $\alpha=1$.  

### 4.2. Node Classification  
- **Transductive**: Mixup+GCN yields +1.6…+1.9% on Cora/Citeseer/Pubmed; Mixup+JKNet +1.5…+2.6%.  
- **Inductive**: Mixup+GraphSAGE +1.9…+3.0% F1; Mixup+GraphSAINT‑GCN +0.6…+2.5%.  

### 4.3. Graph Classification  
- Mixup+GCN gains +1.1…+1.6% on chemical & +1% on social; Mixup+GIN +1.1…+1.9% & +2%.  

### 4.4. Data‑scarce Regime  
- With only 30–50% labeled nodes, Mixup+GCN still leads by ~2%.  
- With only 60–80% labeled graphs, Mixup+GCN/GIN improves by ~2–3%.  

### 4.5. Visualization & Ablations  
- **t‑SNE** on Cora: tighter class clusters with Mixup.  
- **Training curves**: Mixup suppresses late‑stage over‑fitting.  
- **Two‑stage ablation**: omitting Stage 1 *hurts* performance.  
- **$\alpha$‑sensitivity**: robust for $\alpha\in[0.2,2]$, best at $\alpha=1$. 

### 4.6. Hyperparameter Settings

| Hyperparameter      | Description                                                                | Options / Range                      |
|---------------------|----------------------------------------------------------------------------|--------------------------------------|
| **mixup_alpha (α)** | Beta‐distribution shape for Mixup weights; λ∼Beta(α, α)                    | {0.2, 0.5, 1, 2, 5}                  |
| **λ (Mixup weight)**| Interpolation coefficient for feature & label mixing                       | Drawn from Beta(α, α) on {0, ..., 1} |
| **p (SubMix ratio)**| Fraction of nodes (resp. edges) to replace when mixing two graphs (SubMix) | any real in (0, 1)                   |
| **K (node count)**  | Number of nodes to sample / use when generating synthetic graphs (G‐Mixup) | set to ⌊mean V⌋ of dataset           |
| **m (samples)**     | Number of synthetic graphs drawn per Mixup                                 | same as # originals (m)              |
| **batch_size**      | Mini‐batch size for training                                               | {32, 128}                            |
| **dropout**         | Dropout probability (both input and/or hidden activations)                 | {0, 0.5}                             |
| **optimizer**       | Stochastic optimizer used to train GNN                                     | Adam                                 |
| **learning_rate**   | Initial learning rate (decayed by ½ every 50 epochs until max 350 epochs)  | 0.01                                 |
| **epochs**          | Total number of training epochs                                            | up to 350                            |


---

## 5. Advantages & Disadvantages  

| Aspect                | Advantages                                                                 | Disadvantages                                 |
|-----------------------|----------------------------------------------------------------------------|-----------------------------------------------|
| **Generality**        | Works for both node & graph classification; backbone‑agnostic.             | Requires two full forward passes per batch.   |
| **Regularization**    | 1–3% accuracy/F1 gains; reduces over‑fitting on small data.                | Hyperparameter $\alpha$ needs modest tuning.  |
| **Topology Handling** | Mixes topology via dual aggregations without explicit graph interpolation. | More complex implementation than image Mixup. |
| **Scalability**       | No extra asymptotic cost; linear in $ E, V $                               |  ~2× per‑iteration compute due to two stages. |
| **Implementation**    | Integrates into existing GNN loops; no new trainable params.               | Slight overhead wiring two‑stage, two‑branch. |
