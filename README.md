# Paper 1 - mixup: BEYOND EMPIRICAL RISK MINIMIZATION

## Introduction

### Background: ERM and Model Complexity

* Modern deep networks achieve breakthroughs in vision, speech, and reinforcement learning by training under the ERM principle (minimizing training error) and by scaling model size (number of parameters) proportionally with the amount of training data.

* According to learning theory, ERM should only reliably converge (generalize well) if the model’s complexity does not grow with the dataset size.
  *  In practice, however, state-of-the-art networks violate this condition by using more parameters as more data is available, raising concerns because it contradicts the classical VC-dimension bounds.
 
_Note_: The classic VC-dimension bounds provide a way to quantify how the generalization error (true risk) of a hypothesis $\( h \)$ in a hypothesis class $\(\mathcal{H}\)$ relates to its empirical error, taking into account the complexity of $\(\mathcal{H}\)$ (measured by its VC-dimension $\(d\))$ and the number of training samples $\(n\)$.

### Shortcomings of ERM in Practice

* Deep networks trained with ERM can simply memorize the training set rather than learn general patterns. Remarkably, they can even fit random labels (indicating that ERM does not inherently prevent overfitting or ensure meaningful learning when model capacity is very high).

* Another observed issue is sensitivity to slight input perturbations. Networks optimized with ERM can be fooled by adversarial examples – inputs that are only slightly different from the training data cause unpredictable, often incorrect outputs -> ERM-trained models may not generalize well to even small shifts in distribution, underscoring a robustness problem.
