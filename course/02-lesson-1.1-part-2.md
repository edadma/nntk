# A Concrete Example: Learning $y = 2x + 1$

Let's work through a complete example to see how gradient descent works in practice.

## Training Data

Suppose we want to learn the function $y = 2x + 1$, and we have these training examples:

$$\begin{aligned}
x_1 &= 1, \quad y_1 = 3\\
x_2 &= 2, \quad y_2 = 5\\  
x_3 &= 3, \quad y_3 = 7
\end{aligned}$$

## Initial Parameters

Let's start with random initial parameters:

$$w = 0.5, \quad b = 0.0$$

## Forward Pass (Making Predictions)

For each training example, compute the prediction using $\hat{y} = wx + b$:

$$\begin{aligned}
\hat{y}_1 &= 0.5 \times 1 + 0.0 = 0.5\\
\hat{y}_2 &= 0.5 \times 2 + 0.0 = 1.0\\
\hat{y}_3 &= 0.5 \times 3 + 0.0 = 1.5
\end{aligned}$$

## Computing the Loss

Calculate the Mean Squared Error:

$$\begin{aligned}
\text{MSE} &= \frac{1}{3} \times [(0.5 - 3)^2 + (1.0 - 5)^2 + (1.5 - 7)^2]\\
&= \frac{1}{3} \times [(-2.5)^2 + (-4.0)^2 + (-5.5)^2]\\
&= \frac{1}{3} \times [6.25 + 16.0 + 30.25] = \frac{52.5}{3} = 17.5
\end{aligned}$$

## Computing the Gradients

Calculate how the loss changes with respect to each parameter:

$$\begin{aligned}
\frac{\partial \text{MSE}}{\partial w} &= \frac{2}{3} \times [(0.5 - 3) \times 1 + (1.0 - 5) \times 2 + (1.5 - 7) \times 3]\\
&= \frac{2}{3} \times [(-2.5) \times 1 + (-4.0) \times 2 + (-5.5) \times 3]\\
&= \frac{2}{3} \times [-2.5 - 8.0 - 16.5] = \frac{2}{3} \times (-27.0) = -18.0
\end{aligned}$$

$$\begin{aligned}
\frac{\partial \text{MSE}}{\partial b} &= \frac{2}{3} \times [(0.5 - 3) + (1.0 - 5) + (1.5 - 7)]\\
&= \frac{2}{3} \times [-2.5 - 4.0 - 5.5] = \frac{2}{3} \times (-12.0) = -8.0
\end{aligned}$$

## Parameter Updates

Using learning rate $\alpha = 0.01$, apply gradient descent:

$$\begin{aligned}
w_{\text{new}} &= 0.5 - 0.01 \times (-18.0) = 0.5 + 0.18 = 0.68\\
b_{\text{new}} &= 0.0 - 0.01 \times (-8.0) = 0.0 + 0.08 = 0.08
\end{aligned}$$

## Interpretation

- Both gradients are **negative**, meaning the loss decreases when we **increase** both $w$ and $b$
- The gradient for $w$ is larger in magnitude, so we need to adjust $w$ more than $b$
- After one update step, our parameters moved from $(w=0.5, b=0.0)$ toward the true values $(w=2.0, b=1.0)$

**If we continue this process for many iterations, the parameters will converge to the true values!**

---

# Key Insights and Takeaways

## 1. Neurons as Function Approximators

A single neuron with no activation function can learn any linear relationship between inputs and outputs. The fundamental equation:

$$y = \mathbf{w}^T\mathbf{x} + b$$

is a powerful building block that scales from simple lines to high-dimensional hyperplanes.

## 2. Parameters Determine Behavior

The weights and biases completely determine what function the neuron represents. **Learning is the process of finding the right parameter values.**

## 3. Loss Functions Measure Quality

The loss function quantifies how wrong our predictions are:

$$\text{Loss} = \frac{1}{m} \sum_{i=1}^{m} (\text{prediction}_i - \text{truth}_i)^2$$

Minimizing the loss leads to better predictions.

## 4. Gradients Guide Learning

Gradients tell us how to adjust parameters to improve performance:

$$\text{gradient} = \frac{\partial \text{Loss}}{\partial \text{parameter}}$$

They're the mathematical foundation of neural network training.

## 5. Learning is Iterative

We don't solve for the best parameters directly—we improve them gradually through many small updates:

$$\text{parameter}_{\text{new}} = \text{parameter}_{\text{old}} - \alpha \times \text{gradient}$$

This iterative process is called **gradient descent**.

## 6. Learning Rate Controls Convergence

The learning rate $\alpha$ is crucial:
- **Too small**: Slow convergence
- **Too large**: Instability or divergence
- **Just right**: Fast, stable convergence

## 7. Foundation for Everything Else

These principles—forward passes, loss computation, gradients, and parameter updates—remain at the heart of **all** neural networks, from simple linear regression to complex language models.

---

# Looking Ahead

You now understand the mathematical foundations of neural networks:

- How a single neuron computes its output
- How to measure prediction quality with loss functions
- How gradients guide parameter updates
- How gradient descent finds good parameters

In the next part of this lesson, we'll implement these concepts in code, first from scratch to build deep understanding, then using the Neural Network Toolkit to see how the library makes these concepts easy to work with.

The journey from this simple neuron to complex deep learning models is one of adding more neurons, connecting them in sophisticated ways, and using advanced training techniques. But the fundamental principles you've learned here will serve as your foundation throughout this course.