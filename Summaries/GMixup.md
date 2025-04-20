# Paper 6 - GMixup: Graph Data Augmentation for Graph Classification

# 1. Introduction  
- **Motivation:** Classical Mixup interpolates Euclidean inputs (e.g., images) but cannot directly apply to graphs because graphs:  
  1. Vary in number of nodes,  
  2. Lack natural node alignment,  
  3. Exhibit non‑Euclidean topologies.  
    
- **Key idea:** Treat each class’s graphs as samples from a *graphon* $W$.  Estimate class‑specific graphons $W_G,W_H$, interpolate  
  $$W_I = \lambda W_G + (1-\lambda)W_H,\quad y_I = \lambda y_G + (1-\lambda)y_H,$$  
  and sample synthetic graphs from $W_I$.  This *between‑graph* augmentation enables information exchange across classes .  
- **Contributions:**  
  1. **G‑Mixup:** class‑level graph augmentation via graphon mixup.  
  2. **Theoretical guarantees:** preserves discriminative motifs (key subgraph patterns).  
  3. **Empirical validation:** improves accuracy, generalization, and robustness.

---

# 2. Preliminaries  

## 2.1 Notations  
- A graph $G$: node set $V(G)$, edge set $E(G)$, $|V(G)|=v(G)$, $|E(G)|=e(G)$.  
- Graph set $\mathcal{G}=\{G_1,\dots,G_m\}$ with label vector $y_G\in\mathbb{R}^C$.  
- Discriminative motif $F_G$: minimal subgraph deciding $G$’s class.  
- Graphon $W$: infinite‑node generator; step‑function approximations $W\in[0,1]^{K\times K}$.  
- Random graph sampling $G(K,W)$:  
  $$u_i\sim\mathrm{Unif}(0,1),\quad A_{ij}\sim\mathrm{Bernoulli}\bigl(W(u_i,u_j)\bigr).$$  
  

## 2.2 Graph Homomorphism & Graphons  
- **Homomorphism density**  
  $$t(F,G)=\frac{\mathrm{hom}(F,G)}{|V(G)|^{|V(F)|}}\,. $$  
- **Graphon homomorphism density**  
  $$t(F,W)=\int_{[0,1]^{|V(F)|}}\!\!\prod_{(i,j)\in E(F)}W(x_i,x_j)\,\mathrm{d}x\,. $$ 

## 2.3 GNN‑based Graph Classification  
Standard GCN layer:  
$$
  a_i^{(k)} = \mathrm{AGG}^{(k)}\bigl(\{h_j^{(k-1)}:j\in N(i)\}\bigr),\quad
  h_i^{(k)} = \mathrm{COMBINE}^{(k)}\bigl(h_i^{(k-1)},a_i^{(k)}\bigr).
$$  
Readout and prediction:  
$$
  h_G=\mathrm{READOUT}(\{h_i^{(K)}:i\in V(G)\}),\quad
  \hat y=\mathrm{softmax}(h_G).
$$ 

---

# 3. Methodology  

## 3.1 G‑Mixup Formulation  
Given two classes’ graph sets $\mathcal{G},\mathcal{H}$ with graphons $W_G,W_H$ and labels $y_G,y_H$:  
1. **Graphon Estimation:** $\mathcal{G}\to W_G,\;\mathcal{H}\to W_H.$  
2. **Interpolation:**  
   $$
     W_I = \lambda W_G + (1-\lambda)W_H,\quad
     y_I = \lambda y_G + (1-\lambda)y_H,\quad
     \lambda\in[0,1].
   $$
3. **Graph Generation:** Sample $\{I_i\}\sim G(K,W_I)$.  
   

## 3.2 Implementation  

### Graphon Estimation  
- Approximate $W$ by a **step function** $W\in[0,1]^{K\times K}$ via LG, SAS, SBA, MC, USVT.  
- Align nodes by degree sorting, average adjacency matrices into $\bar A$, then estimate $W$.  
- **Complexity**:  

| **Method** | **Complexity**                  |
|------------|----------------------------------|
| MC         | $$O(N^3)$$                       |
| USVT       | $$O(N^3)$$                       |
| LG         | $$O(mN^2)$$                      |
| SBA        | $$O(mK\,N\log N)$$               |
| SAS        | $$O(mN\log N + K^2\log K)$$      |  
  

### Synthetic Graph Generation  
- For each synthetic graph with $K$ nodes:  
  $$
    u_i\sim\mathrm{Unif}(0,1),\quad
    A_{ij}\sim\mathrm{Bernoulli}\bigl(W[\lfloor Ku_i\rfloor,\lfloor Ku_j\rfloor]\bigr).
  $$
- Node features inherited by pooling original aligned features. 

---

# 4. Theoretical Justification  

## 4.1 Preserving Discriminative Motifs in $W_I$  
**Definition 4.1** (Discriminative Motif)  
Smallest subgraph $F_G$ that determines class of $G$. .

**Theorem 4.2**  
For any discriminative motif $F$,  
$$
  \bigl|t(F,W_I)-t(F,W_G)\bigr|\;\le\;(1-\lambda)\,e(F)\,\lVert W_H-W_G\rVert_\square,
$$  
and similarly swapping $G,H$.  Hence $W_I$ retains key motif densities up to a controlled bound. 

## 4.2 Concentration in Synthetic Graphs  
**Theorem 4.3**  
Sampling $n$ graphs from $W_I$ yields motif density concentration:  
$$
  \Pr\bigl(|t(F,G)-t(F,W_I)|>\varepsilon\bigr)\;\le\;2\exp\!\Bigl(-\tfrac{\varepsilon^2\,n}{8\,v(F)^2}\Bigr).
$$  
Thus synthetic graphs mirror mixed motif densities with high probability. 

## 4.3 Discussion  
- **Edge perturbation:** DropEdge/Graphon‑based are degenerate cases at $\lambda=0$.  
- **Manifold Mixup:** model‑dependent; G‑Mixup is pre‑training and model‑agnostic .

---

# 5. Experiments  

## 5.1 Graphon Divergence  
Distinct estimated graphons per class (e.g., IMDB‑BINARY, REDDIT‑BINARY, IMDB‑MULTI) validate basis for mixup .

## 5.2 Case Study  
On REDDIT‑BINARY, graphs from $0.5W_0+0.5W_1$ display both a high‑degree node and a dense subcommunity, evidencing motif mixing .

## 5.3 Classification Performance  
G‑Mixup outperforms vanilla and DropEdge/Manifold Mixup across GCN, GIN, DiffPool, MinCutPool, GMT backbones, improving average accuracy by ~2.8% and stabilizing test loss curves .

## 5.4 Robustness  
- **Label noise:** maintains higher accuracy under 10–40% corruption.  
- **Topology noise:** more resilient to random edge removal/addition.  
G‑Mixup yields superior robustness over baselines .

## 5.5 Hyperparameter Settings

| Hyperparameter           | Description                                                                                  | Options/Range                            |
|--------------------------|----------------------------------------------------------------------------------------------|------------------------------------------|
| **λ (mixup_ratio)**      | Trade‑off weight for interpolating two class graphons                                         | [0.1, 0.2]                               |
| **α (aug_ratio)**        | Fraction of additional synthetic graphs generated                                             | 0.2 (20%)                                |
| **g (graphon_estimator)**| Algorithm to estimate the step‑function approximation of a graphon                           | {LG, SAS, SBA, MC, USVT} (experiments use LG) |
| **K (synth_nodes)**      | Number of nodes per sampled synthetic graph; also number of step‑function partitions          | Average number of nodes in training graphs |
| **initial_lr**           | Adam optimizer initial learning rate                                                         | 0.01                                     |
| **lr_decay**             | Learning‑rate decay schedule                                                                 | Multiply by 0.5 every 100 epochs         |
| **batch_size**           | Number of graphs per training minibatch                                                      | 128                                      |
| **split_ratio**          | Train/validation/test dataset proportions                                                    | 70%/10%/20%                              |
| **GCN_layers**           | Depth of the GCN backbone (number of GCN layers)                                             | 4                                        |
| **GIN_layers**           | Depth of the GIN backbone (number of GIN layers)                                             | 5                                        |
| **MLP_layers (GIN)**     | Number of layers in each GIN’s internal MLP                                                  | 2                                        |
| **hidden_units**         | Dimensionality of hidden node‑embedding vectors                                              | 64                                       |
| **activation**           | Nonlinear activation used in all GNN layers                                                  | ReLU                                     |
| **pooling_methods**      | Set of graph‑level pooling or augmentation methods evaluated (for comparison)                | {TopKPool, DiffPool, MinCutPool, GMT}    |


---

# 6. Advantages & Disadvantages  

| Aspect                    | Advantages                                        | Disadvantages                                        |
|---------------------------|---------------------------------------------------|------------------------------------------------------|
| **Generality**            | Model‑agnostic; pre‑training augmentation         | Graphon estimation adds overhead                     |
| **Theoretical support**   | Motif preservation bounds (Theorems 4.2, 4.3)     | Error depends on $\lambda$ and cut‑norm gap          |
| **Empirical gains**       | Improves accuracy, generalization, robustness     | Sensitive to chosen $K$ (#nodes in synthetic graphs) |
| **Implementation**        | Uses standard estimators (LG, SAS, SBA, MC, USVT) | Requires careful alignment and sampling              |

---