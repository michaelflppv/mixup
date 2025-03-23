# Paper 1 - mixup: BEYOND EMPIRICAL RISK MINIMIZATION

## 1. Empirical Risk Minimization (ERM)

**Definition:**  
ERM seeks a function $ f $ that minimizes the average loss over the training data. It relies on the empirical risk computed from a finite set of examples.

> [!IMPORTANT]  
> ERM is a standard training method where the model is trained by minimizing the average error on the training data. In other words, it tries to fit the training examples as well as possible, which can sometimes lead to memorizing the data rather than learning general patterns.

**Formulas:**  
- **Expected Risk:**  
  $$ R(f) = \int \ell(f(x), y) \, dP(x, y) $$
- **Empirical Risk:**  
  $$ R_\delta(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(x_i), y_i) $$  

| Advantages | Disadvantages |
|------------|---------------|
| - *"minimize their average error over the training data"* <br>- Straightforward and widely used in successful applications | - *"allows large neural networks to memorize (instead of generalize from) the training data even in the presence of strong regularization, or in classification problems where the labels are assigned at random"* <br>- *"change their predictions drastically when evaluated on examples just outside the training distribution"* |

---

## 2. Vicinal Risk Minimization (VRM)

**Definition:**  
VRM approximates the unknown data distribution by defining a vicinity (or neighborhood) around each training example using a vicinity distribution \( \nu \).

> [!IMPORTANT]  
> VRM extends ERM by considering not just the exact training examples but also the nearby (or "vicinal") points. This is done by generating new examples through small modifications (data augmentation) of the original ones. The idea is to better approximate the true data distribution and help the model generalize beyond just the exact training points.

**Formulas:**  
- **VRM Distribution:**  
  $$ P_\nu(\tilde{x}, \tilde{y}) = \frac{1}{n} \sum_{i=1}^n \nu(\tilde{x}, \tilde{y} \mid x_i, y_i) $$  

- **Example (Gaussian Vicinity):**  
  $$ \nu(\tilde{x}, \tilde{y} \mid x_i, y_i) = \mathcal{N}(\tilde{x} - x_i, \sigma^2) \, \delta(\tilde{y} = y_i) $$  
  > "Chapelle et al. (2000) considered Gaussian vicinities \(\nu(\tilde{x}, \tilde{y} \mid x_i, y_i) = \mathcal{N}(\tilde{x} - x_i, \sigma^2) \, \delta(\tilde{y} = y_i)\)."  

| Advantages | Disadvantages |
|------------|---------------|
| - *"data augmentation consistently leads to improved generalization"* <br>- Enlarges the training distribution by drawing virtual examples from the vicinity of each training example | - *"the procedure is dataset-dependent, and thus requires the use of expert knowledge"* <br>- *"assumes that the examples in the vicinity share the same class, and does not model the vicinity relation across examples of different classes"* | 

---

### Definitions

> [!NOTE]  
> The classic VC-dimension bounds provide a way to quantify how the generalization error (true risk) of a hypothesis \( h \) in a hypothesis class \(\mathcal{H}\) relates to its empirical error, taking into account the complexity of \(\mathcal{H}\) (measured by its VC-dimension \( d \)) and the number of training samples \( n \).

> [!NOTE]  
> An adversarial example is an input that has been slightly and intentionally perturbed to fool a model into making a wrong prediction, even though the change is imperceptible to humans.

---

## 3. mixup

**Definition:**  
mixup is a data-agnostic augmentation technique that generates virtual training examples by forming convex combinations of pairs of training examples and their labels.

> [!IMPORTANT]  
> Mixup is a data augmentation technique that creates new virtual training examples by combining (or "mixing") two randomly chosen examples and their labels. For example, if you mix 70% of one image with 30% of another, you also mix their labels in the same proportion. This encourages the model to learn smooth transitions between different classes, which improves generalization and robustness.

**Formulas:**  
- **mixup Vicinal Distribution:**  
  $$ \mu(\tilde{x}, \tilde{y} \mid x_i, y_i) = \frac{1}{n} \sum_j \mathbb{E}_{\lambda} \left[\delta\left(\tilde{x} = \lambda x_i + (1-\lambda)x_j,\ \tilde{y} = \lambda y_i + (1-\lambda)y_j\right)\right], \quad \text{where } \lambda \sim \mathrm{Beta}(\alpha, \alpha) $$
  
- **Simplified Virtual Example Creation:**  
  $$ \tilde{x} = \lambda x_i + (1-\lambda)x_j $$
  $$ \tilde{y} = \lambda y_i + (1-\lambda)y_j $$

**Advantages:**  
- **Data-Agnostic:** No need for domain-specific augmentations.  
- **Improved Generalization & Robustness:**  
  - Encourages the model to behave linearly between examples, reducing undesirable oscillations.  
  - Reduces memorization and increases resistance to adversarial perturbations.  
  > "mixup extends the training distribution by incorporating the prior knowledge that linear interpolations of feature vectors should lead to linear interpolations of the associated targets."  
- **Minimal Overhead:**  
  > "can be implemented in a few lines of code, and introduces minimal computation overhead."  

**Disadvantages:**  
- **Hyperparameter Sensitivity:** The choice of \( \alpha \) is crucial; large \( \alpha \) may lead to underfitting.  
- **Design Choices Matter:**  
  - Interpolating only between examples of the same class did not yield performance gains.  
  - Using convex combinations of three or more examples increases computational cost without further benefit.  
  > "convex combinations of three or more examples with weights sampled from a Dirichlet distribution does not provide further gain, but increases the computation cost of mixup." 

---

## 3 EXPERIMENTS – Key Results & Interpretations

### 3.1 IMAGENET CLASSIFICATION

> **"We use mixup and ERM to train several state-of-the-art ImageNet-2012 classification models, and report both top-1 and top-5 error rates in Table 1."**  

- **Note:**  
  - **Result:** Mixup yields lower top-1 and top-5 errors compared to standard ERM.  
  - **Interpretation:** The improvement in error rates demonstrates that mixup effectively enhances generalization on large-scale image data. Models with higher capacity (e.g., ResNet-101, ResNeXt-101) see more significant gains.

### 3.2 CIFAR-10 AND CIFAR-100

> **"In both CIFAR-10 and CIFAR-100 classification problems, the models trained using mixup significantly outperform their analogues trained with ERM."**  

- **Note:**  
  - **Result:** Across multiple architectures (PreAct ResNet-18, WideResNet-28-10, DenseNet-BC-190), mixup achieves notably lower test errors.  
  - **Interpretation:** The consistency of performance improvements indicates that mixup is effective in reducing overfitting and enhances convergence speed, while reaching similar convergence rates as ERM (see also Figure 3b).

### 3.3 SPEECH DATA

> **"Table 4 shows that mixup outperforms ERM on this task, specially when using VGG-11, the model with larger capacity."**  

- **Note:**  
  - **Result:** Experiments on the Google commands dataset reveal that mixup reduces test errors compared to ERM in speech recognition.  
  - **Interpretation:** The advantage is more pronounced for models with greater capacity (e.g., VGG-11), suggesting that mixup’s regularizing effect is beneficial even beyond image tasks.

### 3.4 MEMORIZATION OF CORRUPTED LABELS

> **"mixup with a large $ \alpha $ (e.g. 8 or 32) outperforms dropout on both the best and last epoch test errors, and achieves lower training error on real labels while remaining resistant to noisy labels."**  

- **Note:**  
  - **Result:** Mixup not only resists memorizing corrupted labels but also shows lower test errors compared to both standard ERM and dropout-based methods.  
  - **Interpretation:** By generating virtual examples that interpolate between real data points, mixup reduces the network's tendency to memorize noise, thereby improving robustness when labels are partially corrupted.

### 3.5 ROBUSTNESS TO ADVERSARIAL EXAMPLES

> **"For the FGSM white box attack, the mixup model is 2.7 times more robust than the ERM model in terms of Top-1 error."**  

> **"mixup is about 40% more robust than ERM in the black box I-FGSM setting."**  

- **Note:**  
  - **Result:** Under both white box and black box adversarial attacks (FGSM and I-FGSM), mixup-trained models consistently yield lower classification errors.  
  - **Interpretation:** This increased robustness is attributed to the enforced linearity between examples, which results in smoother decision boundaries and lower sensitivity to adversarial perturbations.

### 3.6 TABULAR DATA

> **"Table 4 shows that mixup improves the average test error on four out of the six considered datasets, and never underperforms ERM."**  

- **Note:**  
  - **Result:** Experiments on various UCI datasets demonstrate that mixup generally outperforms or matches ERM.  
  - **Interpretation:** The method’s benefits are not limited to image or speech data; mixup’s data-agnostic approach makes it applicable to tabular data as well, leading to more consistent generalization.

### 3.7 STABILIZATION OF GENERATIVE ADVERSARIAL NETWORKS (GANS)

> **"Figure 5 illustrates the stabilizing effect of mixup on the training of GAN (orange samples) when modeling two toy datasets."**  

- **Note:**  
  - **Result:** Mixup contributes to stabilizing GAN training, as evidenced by smoother transitions in the generated data.  
  - **Interpretation:** By acting as a regularizer on the discriminator’s gradients, mixup provides more stable feedback to the generator, which is crucial for successful GAN training.

### 3.8 ABLATION STUDIES

> **"From the ablation study experiments, we have the following observations. First, mixup is the best data augmentation method we test, and is significantly better than the second best method (mix input + label smoothing)."**  

- **Note:**  
  - **Result:** Systematic variations in design choices (e.g., using different weight decay values, interpolating latent representations vs. raw inputs, mixing only same-class examples) consistently show that the standard mixup (random convex combinations of raw inputs and their labels) outperforms alternatives.  
  - **Interpretation:** These ablation studies confirm that every component of mixup contributes to its performance, underscoring the importance of both input and label interpolation as well as the chosen hyperparameters (like the \( \alpha \) parameter).

## Summary Interpretation

| **Aspect**                  | **Interpretation**                                                                                                                                           |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Generalization Improvement** | Mixup consistently reduces test errors across diverse datasets (image, speech, tabular), indicating enhanced generalization compared to ERM.               |
| **Robustness Enhancements**    | Models trained with mixup exhibit increased robustness to adversarial attacks and corrupted labels, attributed to smoother, linear decision boundaries.      |
| **Stability in GANs**          | When applied to GANs, mixup stabilizes training by regularizing the discriminator’s gradients, providing more reliable feedback to the generator.              |
| **Design Validation**          | Ablation studies confirm that each component of mixup (convex combination of inputs and labels, choice of hyperparameter \( \alpha \)) is crucial for its performance.  |

---

## Hyperparameters Description

| **Hyperparameter**                  | **Short Description**                                                                                  | **Options / Range**                                                                                                                                                          |
|-------------------------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **$ \alpha $ (mixup\_alpha)**       | Controls the strength of interpolation between two examples (the Beta distribution’s shape).           | $ \alpha \in (0, \infty) $; typical values: $ 0.1 – 0.4 $ for ImageNet, $ 1 $ for CIFAR, $\{1, 2, 8, 32\}$ for experiments with corrupted labels, $0.2$ for GAN experiments. |
| **$ \lambda $ (mixup coefficient)** | The convex combination weight used to mix two examples; drawn from Beta $( \beta $).                   | $ \lambda \in [0, 1] $; its distribution is governed by $ \alpha $.                                                                                                          |
| **Pair Selection Strategy**         | Determines how the two examples are selected for mixing.                                               | Default is "Random Pairing"; alternatives include "k-Nearest Neighbors (KNN)" (e.g., k=200), "Same Class (SC)" vs. "All Classes (AC)".                                       |
| **Dropout Probability $( p )$**     | (When combined with mixup) The rate at which units are randomly dropped for additional regularization. | $ p \in \{0.5, 0.7, 0.8, 0.9\} $ in standard settings; in mixup+dropout experiments, $ p \in \{0.3, 0.5, 0.7\} $                                                             |
| **Weight Decay**                    | Regularization parameter applied to network weights during training.                                   | Common values are $ 1 \times 10^{-4} $ (preferred with mixup) or $ 5 \times 10^{-4} $ (used for ERM in ablation studies).                                                    |

---

### 4 RELATED WORK

- **Data Augmentation’s Role:**  
  - *"Data augmentation lies at the heart of all successful applications of deep learning..."*  
    - **Explanation:** Many deep learning successes (e.g., image classification, speech recognition) rely on using domain-specific transformations (rotations, translations, cropping, etc.) to improve generalization.

- **Prior Approaches:**  
  - **Nearest Neighbor Interpolation:**  
    - Methods by Chawla et al. (2002) and DeVries & Taylor (2017) augment rare classes by interpolating between nearest neighbors in feature space.  
    - **Limitation:** These techniques only work within a class and do not adjust the corresponding labels.
  - **Label Smoothing:**  
    - Techniques like label smoothing (Szegedy et al., 2016) or penalizing overconfident softmax outputs (Pereyra et al., 2017) regularize outputs by using “soft” labels.  
    - **Limitation:** They regularize labels independently from the input features.

- **mixup’s Advantages Over Prior Work:**  
  - *"mixup enjoys several desirable aspects of previous data augmentation and regularization schemes without suffering from their drawbacks."*  
    - **Key Points:**
      - **Data-Agnostic:** Does not require specific domain knowledge.
      - **Joint Interpolation:** Simultaneously interpolates inputs and labels, establishing a linear relation between them.
      - **Strong Regularizer:** The imposed linearity improves generalization, connecting mixup with ideas from Sobolev training and WGAN-GP.

---

### 5 DISCUSSION

- **Summary of mixup:**  
  - *"We have proposed mixup, a data-agnostic and straightforward data augmentation principle."*  
    - **Explanation:** mixup uses linear interpolation of two random training examples (and their labels) to generate new virtual examples, making it easy to integrate into existing training pipelines with minimal overhead.

- **Empirical Benefits:**  
  - mixup consistently improves the generalization error on ImageNet, CIFAR, speech, and tabular datasets.  
  - It combats issues like memorization of corrupt labels, sensitivity to adversarial examples, and instability in adversarial training.

- **Trade-Off and Model Complexity:**  
  - *"with increasingly large \( \alpha \), the training error on real data increases, while the generalization gap decreases."*  
    - **Explanation:** This trend suggests that mixup implicitly controls model complexity. However, the exact “sweet spot” for this bias-variance trade-off remains an open question.

- **Future Directions:**  
  - Explore the extension of mixup to other types of supervised learning (e.g., regression, structured prediction).
  - Investigate its potential in unsupervised, semi-supervised, and reinforcement learning.
  - Consider adapting mixup for feature-label extrapolation to enhance robustness far from the training data.
