# mini_math_refresher
The goal is to understand the “math under the hood” of AI (instead of only calling PyTorch/TensorFlow black boxes). This AI-focused math refresher emphasizes linear algebra, probability, optimization, and numerical methods core topics

1. Linear Algebra (the backbone)

Vectors, matrices, tensors

Matrix multiplication and properties (associativity, distributivity)

Transpose, inverse, orthogonal, and symmetric matrices

Dot products, norms, cosine similarity

Eigenvalues, eigenvectors, diagonalization, SVD

Projections and subspaces

Tensor reshaping & broadcasting (how data is reorganized)

👉 Application: embeddings, attention (q @ k^T), PCA, SVD-based dimensionality reduction.

2. Calculus & Optimization

Derivatives, gradients, Jacobians

Chain rule (core for backpropagation)

Partial derivatives and multivariable calculus

Gradient descent & variants (SGD, momentum, Adam)

Convexity, local vs. global minima

👉 Application: training neural nets, loss minimization.

3. Probability & Statistics

Probability distributions (Gaussian, categorical, Bernoulli, softmax as categorical)

Expectation, variance, covariance

Bayes’ theorem

Law of large numbers, central limit theorem

KL divergence, cross-entropy

👉 Application: softmax classifier, variational inference, language modeling.

4. Numerical Methods & Linear Systems

Iterative methods for solving Ax = b

Stability, precision, overflow/underflow

Normalization techniques

Approximation (Taylor series, numerical integration)

👉 Application: why training diverges or gradients explode.

5. Discrete Math & Algorithms

Graphs and adjacency matrices

Complexity analysis (O(n), O(n²), etc.)

Sampling methods (Monte Carlo, importance sampling)

👉 Application: transformers (attention graphs), random walks, GNNs.
