Before writing any code this is my TLDR to make sure I understand and can break down the model that I am studying. 
These readme's will be broken down in the assumptions made, the context in which it's used, and a breakdown of the algorithm.

Assumptions: 
- Independence: the first assumption is that each data point is independent of each other, meaning there is no observations depending on another observation
- Binary target: The target variable is normally binary - (0 or 1, yes or no, etc.)
- Linearity: As this is a regression model utilizing log-odds, the independent variables should maintain linearity with the log-odds of the dependent variable

Algorithm Context: 

This algorithm is a regression algorithm used to determine the class an input belongs to rather than a continuous numeric value
For example, we have a dataset that is trying to classify whether a person has diabetes or not. We can utilize logistic regression, assign the value of 0 to no diabetes and 1 to diabetes - and then develop a model to determine the factors and their weights to classifying whether a person has diabetes or not.

Algorithm Breakdown: 

Starting with logistic regression because we apply a sigmoid function to a linear regression function in order to maintain an output from 0 to 1. Our lienar regression function is simple as its a linear combination of our input features (z = w*X + b) where X is our input matrix, w are our weights, and b is our constant term applied. 


Linear Score:

$$
z_i = w^\top x_i + b
$$


Sigmoid Function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

To get the probability of class i, we can use this formula that replaces our variable in our sigmoid function with our linear score

$$
\hat{p}_i = P(y_i = 1 \mid x_i) = \sigma(w^\top x_i + b)
$$


To further extend this, we can use Maximum Likelihood Estimation (MLE) to estimate our coefficients of our logistic regression algorithm. To do this, we want to find the weights and the bias that maximize the likelihood of observing the data. 

Let's break down our understanding of differenting our log-likelihood equation and ultimately let's see how finding the steepest gradient ascent will help us determine what the best set of weights is (MLE).

To better understand this, first we need to understand what we can and can't control. We can't control/manipulate the observations (our X and y). What we can control are the weights and bias of each variable. To improve our model, we utilize partial derivatives (changing weight_j by a tiny amount, how does our function, l, change). We take the derivative of each part as the derivative of a sum is the sum of its derivatives - by differentiating each component, we are also differentiating the entire thing.

We want to find hte point where our derivative is at a maximum which is known as our MLE. we need to iterate, and compute the graident until we are close to 0. We know this is true becasue our log-likelihood function has an global max meaning the maximum we've reached will be the maximum for the function.

*Copied from claude*

# Deriving the Logistic Regression Gradient (MLE with L2)

## Step 1: The Objective

$$
\ell(w, b) = \frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \Big] \;-\; \frac{\lambda}{2n} \sum_{k=1}^{d} w_k^2
$$

where

$$
z_i = w^\top x_i + b, \qquad p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

Each weight affects $\ell$ through a chain: $w_j \rightarrow z_i \rightarrow p_i \rightarrow \ell$.

## Step 2: Chain Rule for One Sample

Let $\ell_i = y_i \log p_i + (1 - y_i)\log(1 - p_i)$. Then:

$$
\frac{\partial \ell_i}{\partial w_j} = \underbrace{\frac{\partial \ell_i}{\partial p_i}}_{\text{link 1}} \cdot \underbrace{\frac{\partial p_i}{\partial z_i}}_{\text{link 2}} \cdot \underbrace{\frac{\partial z_i}{\partial w_j}}_{\text{link 3}}
$$

## Step 3: Compute Each Link

**Link 1** (log-likelihood with respect to $p$), using $\frac{d}{dp}\log p = \frac{1}{p}$ and $\frac{d}{dp}\log(1-p) = -\frac{1}{1-p}$:

$$
\frac{\partial \ell_i}{\partial p_i} = \frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i}
$$

**Link 2** (sigmoid derivative):

$$
\frac{\partial p_i}{\partial z_i} = p_i (1 - p_i)
$$

**Link 3** (only one term of $z_i$ contains $w_j$):

$$
\frac{\partial z_i}{\partial w_j} = x_{ij}
$$

## Step 4: Multiply Links 1 and 2

$$
\left( \frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i} \right) p_i (1 - p_i)
$$

Distribute and cancel:

$$
= y_i (1 - p_i) - (1 - y_i) p_i
$$

Expand:

$$
= y_i - y_i p_i - p_i + y_i p_i
$$

The $y_i p_i$ terms cancel:

$$
\frac{\partial \ell_i}{\partial z_i} = y_i - p_i
$$

## Step 5: Add Link 3

$$
\frac{\partial \ell_i}{\partial w_j} = (y_i - p_i)\, x_{ij}
$$

## Step 6: Average Over All Samples

$$
\frac{\partial}{\partial w_j} \left( \frac{1}{n} \sum_{i=1}^{n} \ell_i \right) = \frac{1}{n} \sum_{i=1}^{n} (y_i - p_i)\, x_{ij}
$$

## Step 7: Differentiate the L2 Penalty

Only the $w_j^2$ term depends on $w_j$:

$$
\frac{\partial}{\partial w_j} \left( -\frac{\lambda}{2n} \sum_{k=1}^{d} w_k^2 \right) = -\frac{\lambda}{2n} \cdot 2 w_j = -\frac{\lambda}{n} w_j
$$

## Step 8: Combine and Vectorize

For one weight:

$$
\frac{\partial \ell}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (y_i - p_i)\, x_{ij} \;-\; \frac{\lambda}{n} w_j
$$

For all weights at once:

$$
\nabla_w \ell = \frac{1}{n} X^\top (y - p) \;-\; \frac{\lambda}{n} w
$$

## Step 9: The Bias

Link 3 becomes $\frac{\partial z_i}{\partial b} = 1$, and the bias is not penalized:

$$
\frac{\partial \ell}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (y_i - p_i)
$$

## Shortcut: Using the Simplified Log-Likelihood

With $\ell_i = y_i z_i - \log(1 + e^{z_i})$:

$$
\frac{\partial \ell_i}{\partial z_i} = y_i - \frac{e^{z_i}}{1 + e^{z_i}} = y_i - \sigma(z_i) = y_i - p_i
$$

Same result as Step 4, in one line.

## Gradient Ascent Update

$$
w \leftarrow w + \alpha \, \nabla_w \ell, \qquad b \leftarrow b + \alpha \, \frac{\partial \ell}{\partial b}
$$

## Mapping to Code

| Derivation | Code |
|---|---|
| Step 4: $y_i - p_i$ | `error = y - p` |
| Step 6: $\frac{1}{n} X^\top (y - p)$ | `dw = X.T @ error / X.shape[0]` |
| Step 7: $-\frac{\lambda}{n}