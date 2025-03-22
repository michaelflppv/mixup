# Paper 1 - mixup: BEYOND EMPIRICAL RISK MINIMIZATION

## 1. Empirical Risk Minimization (ERM)

**Definition:**  
ERM seeks a function *f* ∈ *F* that minimizes the average loss over the training data. It relies on the empirical risk computed from a finite set of examples.

**Formulas:**  
- **Expected Risk:**  
  ```
  R(f) = ∫ `(f(x), y) dP(x, y)
  ```
- **Empirical Risk:**  
  ```
  Rδ(f) = 1/n ∑₍ᵢ₌₁₎ⁿ `(f(xᵢ), yᵢ)
  ```  

| Advantages | Disadvantages |
|------------|---------------|
| - *"minimize their average error over the training data"* <br>- Straightforward and widely used in successful applications | - *"allows large neural networks to memorize (instead of generalize from) the training data even in the presence of strong regularization, or in classification problems where the labels are assigned at random"* <br>- *"change their predictions drastically when evaluated on examples just outside the training distribution"* |

---

## 2. Vicinal Risk Minimization (VRM)

**Definition:**  
VRM approximates the unknown data distribution by defining a vicinity (or neighborhood) around each training example using a vicinity distribution *ν*.

**Formulas:**  
- **VRM Distribution:**  
  ```
  Pν(x̃, ỹ) = 1/n ∑₍ᵢ₌₁₎ⁿ ν(x̃, ỹ | xᵢ, yᵢ)
  ```  

- **Example (Gaussian Vicinity):**  
  ```
  ν(x̃, ỹ|xᵢ, yᵢ) = N(x̃ − xᵢ, σ²) δ(ỹ = yᵢ)
  ```  
  > "Chapelle et al. (2000) considered Gaussian vicinities ν(x̃, ỹ|xᵢ, yᵢ) = N (x̃− xᵢ, σ²) δ(ỹ = yᵢ)."  

| Advantages | Disadvantages |
|------------|---------------|
| - *"data augmentation consistently leads to improved generalization"* <br>- Enlarges the training distribution by drawing virtual examples from the vicinity of each training example | - *"the procedure is dataset-dependent, and thus requires the use of expert knowledge"* <br>- *"assumes that the examples in the vicinity share the same class, and does not model the vicinity relation across examples of different classes"* | 

---

### Definitions

> [!NOTE]  
> The classic VC-dimension bounds provide a way to quantify how the generalization error (true risk) of a hypothesis $\( h \)$ in a hypothesis class $\(\mathcal{H}\)$ relates to its empirical error, taking into account the complexity of $\(\mathcal{H}\)$ (measured by its VC-dimension $\(d\))$ and the number of training samples $\(n\)$.

> [!NOTE]  
> An adversarial example is an input that has been slightly and intentionally perturbed to fool a model into making a wrong prediction, even though the change is imperceptible to humans.

---

## 3. mixup

**Definition:**  
mixup is a data-agnostic augmentation technique that generates virtual training examples by forming convex combinations of pairs of training examples and their labels.

**Formulas:**  
- **mixup Vicinal Distribution:**  
  ```
  µ(x̃, ỹ|xᵢ, yᵢ) = 1/n ∑ⱼ Eₗ [δ(x̃ = λ · xᵢ + (1−λ) · xⱼ, ỹ = λ · yᵢ + (1−λ) · yⱼ)],  where λ ∼ Beta(α, α)
  ```  

- **Simplified Virtual Example Creation:**  
  ```
  x̃ = λxᵢ + (1−λ)xⱼ  
  ỹ = λyᵢ + (1−λ)yⱼ
  ```  

**Advantages:**  
- **Data-Agnostic:** No need for domain-specific augmentations.  
- **Improved Generalization & Robustness:**  
  - Encourages the model to behave linearly between examples, reducing undesirable oscillations.  
  - Reduces memorization and increases resistance to adversarial perturbations.  
  > "mixup extends the training distribution by incorporating the prior knowledge that linear interpolations of feature vectors should lead to linear interpolations of the associated targets."  
- **Minimal Overhead:**  
  > "can be implemented in a few lines of code, and introduces minimal computation overhead."  

**Disadvantages:**  
- **Hyperparameter Sensitivity:** The choice of α is crucial; large α may lead to underfitting.  
- **Design Choices Matter:**  
  - Interpolating only between examples of the same class did not yield performance gains.  
  - Using convex combinations of three or more examples increases computational cost without further benefit.  
  > "convex combinations of three or more examples with weights sampled from a Dirichlet distribution does not provide further gain, but increases the computation cost of mixup." 
