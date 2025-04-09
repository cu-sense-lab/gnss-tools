


The Lagrange polynomial over $k+1$ data points  $\{(x_0, y_0), \dots, (x_k,y_k)\}$ in barycentric form is:


$$
L(x) = \sum_j L_j(x) y_j = l(x) \sum_{j=0}^k \frac{w_j}{x - x_j} y_j
$$

where

$$
l(x) = \prod_{m=0}^k(x-x_m)
$$

and

$$
w_j = \prod_{m\neq j} (x_j - x_m)^{-1}
$$

Note that the derivative w.r.t. $x$ is:

$$
L'(x) = \sum_j L'_j(x) y_j
$$

Consider the following:

$$
\begin{align*}
\log(L_j(x)) &= \log(l(x)) - \log(x - x_j) + \log(w_j) + \log(y_j)  \\
&= \left(\sum_{m=0}^k\log(x-x_m)\right) - \log(x-x_j) - \left(\sum_{m\neq j} \log(x_j - x_m) \right) + \log(y_j) \\
&= \left(\sum_{m\neq j} \log(x-x_m) \right) - \left(  \sum_{m\neq j} \log(x_j - x_m) \right) + \log y_j
\end{align*}
$$

Differentiating w.r.t. $x$, we get:

$$
\frac{L_j'(x)}{L_j(x)} = \left(\sum_{m\neq j} \frac{1}{x-x_m}\right)
$$

And so:

$$
\begin{align*}
L_j'(x) &= \left(\sum_{m\neq j} \frac{1}{x-x_m} \right) L_j(x)  \\
&= \left(\left(\sum_{m=0}^k \frac{1}{x-x_m} \right) - \frac{1}{x - x_j}\right) l(x) \frac{w_j}{x - x_j} y_j \\
% &= \left(d\log(l(x)) - \frac{1}{x-x_j} \right) l(x) \frac{w_j}{x - x_j} y_j
\end{align*}
$$



###

Note: normally when $x \equiv x_j$, we have:

$$
L_m(x) = \delta_{mj}
$$

however, with the derivative terms, when $x \equiv x_p$:

$$
L_j'(x)= \begin{cases}
\left(\sum_{m\neq j} \frac{1}{x - x_m} \right) & j = p \\
\left(\prod_{m\neq p}(x-x_m) \right) \frac{w_j}{x-x_j} & \text{otherwise}
\end{cases}
$$

<!-- and

$$
L'_j(x) = \delta_{mj} \sum_{m\neq j} \frac{1}{x - x_m}
$$


### -->

<!-- Note: -->

<!-- $$
\begin{align*}
\sum_{m\neq j} \frac{1}{x-x_m}
\end{align*}
$$ -->