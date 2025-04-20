# Paper 7 - SubMix: Model-Agnostic Augmentation for Accurate Graph Classification

## Abstract  
- **Goal**: Augment graph datasets for better graph‐classification accuracy.  
- **Limitations of prior work**:  
  - *Model‐specific* methods lose generality.  
  - *Model‐agnostic* heuristics (e.g. random edge drop) lack guarantees.  
- **This paper** introduces **five desiderata** for safe/diverse augmentation, then proposes two **model‐agnostic** algorithms:  
  1. **NodeSam**: balanced node‐split/merge to minimally perturb structure.  
  2. **SubMix**: *Mixup‑style* subgraph‐swapping to richly intermix class evidence.   
- **Result**: On social and molecular benchmarks, NodeSam & SubMix outperform existing model‐agnostic augmenters.

---

## 1.Introduction  
- **Graph augmentation** enlarges the training distribution to improve classifiers (akin to images/text/time‐series).  
- **Community structure** is often key to graph labels in social/molecular data.  
- **Model‐specific** augmenters tailor to one GNN but don’t transfer; **model‐agnostic** ones (e.g. DropEdge, GraphCrop, NodeAug, MotifSwap) are general but rely on heuristics, lacking unbiasedness or scalability guarantees—and sometimes *hurt* accuracy.   
- **Contributions**:  
  1. Five *desired properties* (§2).  
  2. Two new augmenters—NodeSam (§3.1) & **SubMix** (§3.2)—that satisfy all properties.  
  3. Theoretical proofs.  
  4. Empirical gains up to 2.1× over baselines.

---

## 2. Problem & Desired Properties  
- **Problem 1**: Given graphs $G=(V,E,X)$, produce augmented $\bar G$ that improves classifier $f$ on graph classification.  
- **Five properties** (§2.2, Table 1 of paper):  
  1. **P1: Preserving size**  
     $$\mathbb{E}[|\bar V|-|V|]=0,\;\mathbb{E}[|\bar E|-|E|]=0$$  
  2. **P2: Preserving connectivity** —$G$ connected iff $\bar G$ is.  
  3. **P3: Changing nodes** — either $\mathbb{E}[(|\bar V|-|V|)^2]>0$ or $\mathbb{E}\|\bar X-X\|_F^2>0$.  
  4. **P4: Changing edges** — $\mathbb{E}[(|\bar E|-|E|)^2]>0$.  
  5. **P5: Linear complexity** — time/space $O(d|V|+|E|)$.   
- **Existing model‑agnostic** methods each violate ≥1 property (see Table 1).

---

## 3. SubMix: *Subgraph Mix* (§3.2)  
### 3.2.1 Core Algorithm (Alg 5)  
1. **Select** target $G=(V,E,X)$ and another $G'=(V',E',X')$ uniformly from pool.  
2. **Sample** two **connected** node‐sets $S\subset V,\;S'\subset V'$ of equal size $k\sim\mathrm{Uniform}(0,p)\cdot\min(|\mathrm{comp}(G,r)|,|\mathrm{comp}(G',r')|)$ via _diffusion_+BFS (Alg 6) .  
3. **Map** nodes via order: $\phi:S'\to S$  
4. **Build** new edge set  
   $$
     \bar E = E_1 \cup E_2
     \quad\text{where}\quad
     E_1=\{(u,v)\in E:\neg(u\in S\land v\in S)\},\;
     E_2=\{(\phi(u),\phi(v)):(u,v)\in E',\,u,v\in S'\}.
   $$  
5. **Replace** features:  
   $$\bar X[\phi(S')] \leftarrow X'[S'].$$  
6. **Soft‐mix label**:  
   $$
     \bar y = \tfrac{|E_1|}{|\bar E|}\,y + \Bigl(1-\tfrac{|E_1|}{|\bar E|}\Bigr)\,y'.
   $$  
> **Interpretation**: Like CutMix for images, SubMix replaces a “patch” of one graph with a patch from another, with weights proportional to edge count .

### 3.2.2 Subgraph Sampling (Alg 6)  
- **Diffusion**: Use **Personalized PageRank**  
  $$
    S = \sum_{k=0}^\infty \alpha(1-\alpha)^k (D^{-1/2}AD^{-1/2})^k,\quad \alpha=0.15,
  $$  
  then pick top‑$k$ nodes by score for connected subgraph; if disconnected, fallback to BFS selection. .

### 3.2.3 Theoretical Guarantees  
- **P1** (size unbiasedness):  
  $$\mathbb{E}[|\bar V|-|V|]=0,\;\mathbb{E}[|\bar E|-|E|]=0$$  
  under uniform‑random graph pairing (Lemma 5).   
- **P2** (connectivity): preserved since replacement within connected components (Lemma 6).  
- **P3–P4**: obviously changes node/edge sets.  
- **P5**: Diffusion + edge‐set ops are $$O(pd|V|+|E|+|E'|)$$ (Lemma 7).

---

## 4. Experiments  

### 4.1 Setup  
- **Datasets**: 9 benchmarks (molecular & social; e.g. D&D, ENZYMES, COLLAB, Twitter).  
- **Classifier**: GIN, fixed hyperparams (batch ∈{32,128}, dropout∈{0,0.5}), 10‐fold CV.  
- **Augment ratio** $ p=0.4$.  
- **Baselines**: DropEdge, DropNode, AddEdge, ChangeAttr, GraphCrop, NodeAug, MotifSwap. 

### 4.2 Accuracy (Q1)  
- **Mean ↑**: SubMix +1.75 pp, NodeSam +1.71 pp over GIN baseline—≈2× gain of best competitor.  
- **Robustness**: Every other model‑agnostic sometimes *decreases* accuracy; SubMix always improves. 

### 4.3 Desiderata (Q2)  
- **Unbiased size**: Boxplots of $\Delta|E|$ are centered at 0 for both SubMix & NodeSam (Fig 4).  
- **Scalability**: Runtime ∝|E| for SubMix; MotifSwap blows up on large Reddit (232K nodes, 11.6M edges) (Fig 5).  
- **Diversity**: t‐SNE on WL‐kernel embeddings shows SubMix spans the largest augmentation space; MotifSwap the smallest (Fig 6).

### 4.4 Ablation (Q3)  
- **SubMixBase** (random node‐swap) vs. full SubMix: full yields highest accuracy/rank.  
- **NodeSamBase** (no adjustment) vs. full NodeSam: adjustment step is crucial to preserve edge/triangle counts (Fig 7).

### 4.5 Hyperparameter Settings

| Hyperparameter    | Description                                                                                 | Options / Range              |
|-------------------|---------------------------------------------------------------------------------------------|------------------------------|
| **p**             | Target augmentation ratio for SubMix: controls size of the subgraph patch to swap            | Continuous ∈ (0, 1), set to 0.4 |
| **α**             | Teleport (restart) probability in Personalized PageRank diffusion used for subgraph sampling | Continuous ∈ (0, 1), set to 0.15 |


---