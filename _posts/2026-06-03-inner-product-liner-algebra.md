---
title: "Inner Product (Dot Product)"
date: 2025-06-02T00:00:00+00:00
author: Gaurav Kumar
layout: post
permalink: linear-algebra-common-vector-normss
categories: DLML
tags: [Linear-Algebra, DLML]
mathjax: true
---
$$\newcommand\mycolv[1]{\begin{bmatrix}#1\end{bmatrix}}$$

An inner product (or scalar product) is a way to multiply two vectors in a vector space, resulting in a scalar value. It's a generalization of the dot product to more abstract vector spaces and provides a way to measure the "similarity" between vectors and define concepts like length and angle.

Inner products, also known as dot products, are mathematical operations that quantify the similarity between vectors.

Inner product of two vectors ($$\vec{u}$$ and $$\vec{v} $$) called **Dot Product**. Denoted $$\vec{u}  \cdot \vec{v}$$

is computed by $$\vec{u } ^{ \mathrm{ T } }\vec{v} $$ and result in a 1x1 martrix which is scalar.

$$ \begin{eqnarray}
\mycolv{1\\2\\3} \cdot \mycolv{4\\5\\6}
= 1.4 + 2.5 + 3.6
= 32
\end{eqnarray}$$


{% highlight python %}

import numpy as np

def inner_dot(x, y):
return sum(x_i * y_i for x_i, y_i in zip(x, y))

x = np.array([2, 7, 1])
y = np.array([8, 2, 8])

print("The dot product of x and y is: ", inner_dot(x, y))

#Alternatively, we can use the np.inner() function.
dot_product = np.inner(x, y)
print("The dot product of x and y is: ", dot_product)

# We can also use numpy.dot() function
# This must be used for 2D and 3D arrays
print("The dot product of x and y is: ", np.dot(x, y))

# From Python 3.5 we can use an explicit operator @
# for the dot product, so you can write the following
print("The dot product of x and y is: ", x @ y)

{% endhighlight %}

We can also obtain the angle/similarity between two vectors with this approach. We take the dot product between vectors $$\vec{u}$$ and $$\vec{v} $$ and if we normalize this product we will obtain the angle between vectors $$\vec{u}$$ and $$\vec{v} $$ .

$$\cos \theta = \frac{\vec{u}\cdot \vec{v}}{\left \| \vec{u} \right \|\left \| \vec{v} \right \|}$$

Example - Lets find out how similar are the two documents <br />
 - D1 : The Dog
 - D2 : The Cat 

in D1 and D2, we have total 3 different words i.e The, Dog, Cat

$$ \begin{eqnarray}
The = \mycolv{0\\1\\0}
\end{eqnarray}$$

$$ \begin{eqnarray}
Dog = \mycolv{0\\0\\1}
\end{eqnarray}$$

$$ \begin{eqnarray}
Cat = \mycolv{1\\0\\0}
\end{eqnarray}$$


$$ \begin{eqnarray}
D1 = \mycolv{0\\1\\1}
\end{eqnarray}$$

$$ \begin{eqnarray}
D2 = \mycolv{1\\1\\0}
\end{eqnarray}$$



$$ \begin{eqnarray}
\cos \theta = \frac{\vec{D1}\cdot \vec{D2}}{\left \| \vec{D1} \right \|\left \| \vec{D2} \right \|}
= \frac{0.1 + 1.1 + 1.0 }{ \sqrt{0^2+1^2+1^2} . \sqrt{1^2+1^2+0^2}}
= \frac{1}{ \sqrt{2} . \sqrt{2}}
= \frac{1}{2}
\end{eqnarray}$$

$$ \theta = 60^{ \circ } $$


√2 * √2 = 2: The square root of a number multiplied by itself equals the original number.