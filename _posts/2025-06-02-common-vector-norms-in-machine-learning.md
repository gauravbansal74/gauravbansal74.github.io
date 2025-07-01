---
title: "Common Vector Norms In Machine Learning"
date: 2025-06-02T00:00:00+00:00
author: Gaurav Kumar
layout: post
permalink: linear-algebra-common-vector-norms
categories: DLML
tags: [Linear-Algebra, DLML]
---

If you are reading this post. it is likely that you have already go through [Vector](/linear-algebra-scalar-vector-matrix-tensor)

In machine learning, vector norms are essential for measuring the size or length of vectors, which are fundamental in various algorithms and techniques. Here are some of the most common vector norms:

### L1 Norm (Manhattan Norm)
The L1 norm, also known as the Manhattan norm, is the sum of the absolute values of the vector's components. It is defined as:
$$ | \mathbf{x} |1 = \sum{i=1}^{n} |x_i| $$

When used to compute the loss, the L1 norm is also referred to as the **Mean Absolute Error**.
This norm is useful in machine learning for promoting sparsity in models, such as in Lasso regression.

### L2 Norm (Euclidean Norm)
The L2 norm, or Euclidean norm, is the square root of the sum of the squares of the vector's components. It is defined as:
$$ | \mathbf{x} |2 = \sqrt{\sum{i=1}^{n} x_i^2} $$

The above equation is often referred to as the **root mean squared error** when used to compute the error.
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

{% highlight python %}

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Predict and evaluate
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

{% endhighlight %}

### L2 Norm in Ridge Regression
Ridge regression, on the other hand, uses the L2 norm for regularization. The L2 norm penalizes the sum of the squared coefficients, which helps in reducing the model complexity and preventing overfitting. Unlike Lasso, Ridge regression does not perform feature selection but rather shrinks all coefficients towards zero.

{% highlight python %}

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict and evaluate
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

{% endhighlight %}

### Elastic Net
Elastic Net is a regularization technique that combines both L1 and L2 norms. It is particularly useful when there are multiple correlated features. The Elastic Net penalty is a linear combination of the L1 and L2 penalties, allowing it to inherit the benefits of both Lasso and Ridge regression.

{% highlight python %}

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

# Predict and evaluate
y_pred = elastic_net.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

{% endhighlight %}

### Infinity Norm in Optimization Problems
The infinity norm is often used in optimization problems where the goal is to minimize the maximum deviation. For example, in robust optimization, the infinity norm can be used to ensure that the solution is not overly sensitive to any single component of the vector.
Practical Example
Consider a dataset with features representing different attributes of houses (e.g., size, number of rooms, location). In a regression model predicting house prices:
- Using the L1 norm (Lasso regression) might result in a model that only includes the most significant features, such as size and location, while ignoring less important ones. 
- Using the L2 norm (Ridge regression) would include all features but with smaller coefficients, ensuring that no single feature dominates the model. 
- Using the infinity norm would focus on minimizing the maximum error in the predictions, ensuring that the model is robust to outliers.

{% highlight python %}
import numpy as np

# Sample vector
x = np.array([1, -2, 3, -4, 5])

# Infinity norm
infinity_norm = np.max(np.abs(x))
print(f'Infinity Norm: {infinity_norm}')
{% endhighlight %}

### How to implement these norms in machine learning algorithms? 😊

#### L1 Norm in Optimization Problems
{% highlight python %}

import numpy as np

# Sample vector
x = np.array([1, -2, 3, -4, 5])

# Calculate L1 norm using numpy.linalg
l1_norm = np.linalg.norm(x, ord=1)

print(f'L1 Norm: {l1_norm}')
{% endhighlight %}


#### L2 Norm in Optimization Problems

{% highlight python %}

import numpy as np

# Sample vector
x = np.array([1, -2, 3, -4, 5])

# Calculate L2 norm using numpy.linalg
l2_norm = np.linalg.norm(x, ord=2)
print(f'L2 Norm: {l2_norm}')

{% endhighlight %}

#### Infinity Norm in Optimization Problems

Here's how you can calculate the infinity norm using NumPy:
{% highlight python %}

import numpy as np

# Sample vector
x = np.array([1, -2, 3, -4, 5])

# Calculate infinity norm using numpy.linalg
infinity_norm = np.linalg.norm(x, ord=np.inf)
print(f'Infinity Norm: {infinity_norm}')

{% endhighlight %}