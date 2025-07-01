---
title: "Common Vector Norm "
date: 2025-06-02T00:00:00+00:00
author: Gaurav Kumar
layout: post
permalink: linear-algebra-common-vector-norms
categories: DLML
tags: [Linear-Algebra, DLML]
---
Common Vector Norms In Machine Learning

If you are reading this post. it is likely that you have already go through [Vector](/linear-algebra-scalar-vector-matrix-tensor)

In machine learning, vector norms are essential for measuring the size or length of vectors, which are fundamental in various algorithms and techniques. Here are some of the most common vector norms:

### L1 Norm (Manhattan Norm)
The L1 norm, also known as the Manhattan norm, is the sum of the absolute values of the vector's components. It is defined as:
$$ | \mathbf{x} |1 = \sum{i=1}^{n} |x_i| $$
This norm is useful in machine learning for promoting sparsity in models, such as in Lasso regression.

### L2 Norm (Euclidean Norm)
The L2 norm, or Euclidean norm, is the square root of the sum of the squares of the vector's components. It is defined as:
$$ | \mathbf{x} |2 = \sqrt{\sum{i=1}^{n} x_i^2} $$
The L2 norm is commonly used in machine learning for regularization techniques like Ridge regression, as it penalizes large coefficients more than the L1 norm.

### Infinity Norm (Max Norm)
The infinity norm, or max norm, is the maximum absolute value of the vector's components. It is defined as:
$$ | \mathbf{x} |_\infty = \max_i |x_i| $$
This norm is useful in scenarios where the largest component dominates the behavior of the vector, such as in certain optimization problems.

### Visual Representation
To better understand these norms, imagine a vector as an arrow in a multi-dimensional space. 
- The L1 norm measures the total distance traveled along each axis
- the L2 norm measures the straight-line distance from the origin to the point
- the infinity norm measures the distance to the farthest point along any axis.

### How these vector norms are used in specific machine learning algorithms:

### L1 Norm in Lasso Regression
Lasso regression is a type of linear regression that uses the L1 norm for regularization. The L1 norm helps in feature selection by shrinking some coefficients to zero, effectively removing less important features from the model. This is particularly useful when dealing with high-dimensional data where many features may be irrelevant.

### L2 Norm in Ridge Regression
Ridge regression, on the other hand, uses the L2 norm for regularization. The L2 norm penalizes the sum of the squared coefficients, which helps in reducing the model complexity and preventing overfitting. Unlike Lasso, Ridge regression does not perform feature selection but rather shrinks all coefficients towards zero.

### Elastic Net
Elastic Net is a regularization technique that combines both L1 and L2 norms. It is particularly useful when there are multiple correlated features. The Elastic Net penalty is a linear combination of the L1 and L2 penalties, allowing it to inherit the benefits of both Lasso and Ridge regression.

### Infinity Norm in Optimization Problems
The infinity norm is often used in optimization problems where the goal is to minimize the maximum deviation. For example, in robust optimization, the infinity norm can be used to ensure that the solution is not overly sensitive to any single component of the vector.
Practical Example
Consider a dataset with features representing different attributes of houses (e.g., size, number of rooms, location). In a regression model predicting house prices:
- Using the L1 norm (Lasso regression) might result in a model that only includes the most significant features, such as size and location, while ignoring less important ones. 
- Using the L2 norm (Ridge regression) would include all features but with smaller coefficients, ensuring that no single feature dominates the model. 
- Using the infinity norm would focus on minimizing the maximum error in the predictions, ensuring that the model is robust to outliers.

### How to implement these norms in machine learning algorithms? 😊


#### Infinity Norm in Optimization Problems

Here's how you can calculate the infinity norm using NumPy:
{% highlight python %}
import numpy as np

x = np.array([1, -2, 3, -4, 5])

infinity_norm = np.max(np.abs(x))
print(f'Infinity Norm: {infinity_norm}')

{% endhighlight %}