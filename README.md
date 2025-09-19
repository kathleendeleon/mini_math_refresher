# mini_math_refresher
### The goal is to understand the “math under the hood” of AI (instead of only calling PyTorch/TensorFlow black boxes). This AI-focused math refresher emphasizes linear algebra, probability, optimization, and numerical methods core topics

## 1. Linear Algebra (the backbone)
  a) Vectors, matrices, tensors \
  b) Matrix multiplication and properties (associativity, distributivity) \
  c) Transpose, inverse, orthogonal, and symmetric matrices \
  d) Dot products, norms, cosine similarity \
  e) Eigenvalues, eigenvectors, diagonalization, SVD \
  f) Projections and subspaces \
  g) Tensor reshaping & broadcasting (how data is reorganized) \
    👉 Application: embeddings, attention (q @ k^T), PCA, SVD-based dimensionality reduction. \
 \
 \
## 2. Calculus & Optimization
  a) Derivatives, gradients, Jacobians \
  b) Chain rule (core for backpropagation) \
  c) Partial derivatives and multivariable calculus \
  d) Gradient descent & variants (SGD, momentum, Adam) \
  e) Convexity, local vs. global minima \
    👉 Application: training neural nets, loss minimization. \
 \
 \
## 3. Probability & Statistics
  a) Probability distributions (Gaussian, categorical, Bernoulli, softmax as categorical) \
  b) Expectation, variance, covariance \
  c) Bayes’ theorem \
  d) Law of large numbers, central limit theorem \
  e) KL divergence, cross-entropy \
    👉 Application: softmax classifier, variational inference, language modeling. \
 \
 \
## 4. Numerical Methods & Linear Systems
  a) Iterative methods for solving Ax = b \
  b) Stability, precision, overflow/underflow \
  c) Normalization techniques \
  e) Approximation (Taylor series, numerical integration) \
    👉 Application: why training diverges or gradients explode. \
 \
 \
## 5. Discrete Math & Algorithms
  a) Graphs and adjacency matrices \
  b) Complexity analysis (O(n), O(n²), etc.) \
  c) Sampling methods (Monte Carlo, importance sampling) \
    👉 Application: transformers (attention graphs), random walks, GNNs. \
