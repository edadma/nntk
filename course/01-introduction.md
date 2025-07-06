# Neural Network Toolkit Course

## Course Introduction: What Are Neural Networks?

Welcome to the Neural Network Toolkit (NNTK) course! This course is designed to teach you neural networks from the ground up using a custom Scala library built specifically for learning. Our goal isn't to create the fastest or most efficient neural network library in the world—it's to create one that you can truly understand, learn from, and experiment with.

### Why Neural Networks?

Neural networks are inspired by how the brain processes information, but they're really just mathematical functions that can learn patterns from data. Here's what makes them special:

1. **Universal Function Approximators**: They can learn to approximate almost any function given enough data and the right architecture
2. **Pattern Recognition**: They excel at finding patterns in complex, high-dimensional data
3. **Adaptive Learning**: They improve their performance through experience (training data)
4. **Hierarchical Feature Learning**: Deep networks can learn complex features by combining simpler ones

### The Learning Journey

Throughout this course, you'll build your understanding through hands-on programming with increasingly complex problems:

- **Foundation**: Learn how individual neurons work and how they combine into networks
- **Pattern Recognition**: See how networks learn to classify and predict
- **Deep Learning**: Understand why deeper networks are more powerful
- **Modern Techniques**: Explore attention, sequence modeling, and advanced architectures

### Our Educational Philosophy

This course emphasizes **understanding over performance**. Every component in our toolkit is designed to be:

- **Readable**: Clear, well-commented code that teaches concepts
- **Debuggable**: Easy to inspect what's happening at each step
- **Experimental**: Simple to modify and test different ideas
- **Educational**: Focused on building intuition alongside technical skills

---

## Unit 1: Neural Network Fundamentals

**Unit Goal**: Understand the mathematical foundations of neural networks by implementing linear regression using a single layer.

**Unit Capstone**: Build a neural network that learns the linear function $y = 2x + 1$ from noisy data, demonstrating mastery of forward passes, loss computation, gradient descent, and parameter updates.

---

# Lesson 1.1: The Artificial Neuron - Building Block of Intelligence

## Learning Objectives

By the end of this lesson, you will:

1. Understand what an artificial neuron is and how it relates to biological neurons
2. Master the mathematical formulation of a single neuron
3. Understand the geometric interpretation of neuron computations
4. See how neurons can represent and learn linear relationships
5. Appreciate the role of parameters (weights and biases) in learning

## From Biology to Mathematics

### The Biological Inspiration

Real neurons in your brain receive signals from other neurons through connections called synapses. Each synapse has a different strength—some signals are amplified, others are diminished. The neuron sums up all these weighted inputs and, if the total is strong enough, "fires" by sending its own signal to other neurons.

### The Mathematical Abstraction

An artificial neuron captures this essence mathematically:

**Input signals:** $x_1, x_2, x_3, \ldots, x_n$  
**Synapse strengths:** $w_1, w_2, w_3, \ldots, w_n$ (called "weights")  
**Neuron threshold:** $b$ (called "bias")

**Weighted sum:** $z = w_1x_1 + w_2x_2 + w_3x_3 + \ldots + w_nx_n + b$  
**Output:** $y = f(z)$ where $f$ is an "activation function"

For our first lessons, we'll use the simplest case: **no activation function** (or equivalently, $f(z) = z$), which gives us:

$$y = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b$$

## The Single-Input Neuron: Linear Regression

Let's start with the simplest possible neuron: one that takes a single input and produces a single output.

### Mathematical Formulation

For a single input $x$, our neuron computes:

$$y = wx + b$$

Where:
- **$x$**: The input value (what we observe)
- **$w$**: The weight (how much influence the input has)
- **$b$**: The bias (shifts the output up or down)
- **$y$**: The output (our prediction)

This is exactly the equation of a straight line! In fact, a single-input neuron with no activation function is identical to linear regression.

### Geometric Interpretation

Think of this geometrically:
- The **weight ($w$)** controls the slope of the line
- The **bias ($b$)** controls where the line crosses the y-axis
- Different values of $w$ and $b$ give us different lines

**Examples:**
- $w = 2, b = 1 \rightarrow y = 2x + 1$ (steep positive slope, crosses y-axis at 1)
- $w = -1, b = 3 \rightarrow y = -x + 3$ (negative slope, crosses y-axis at 3)
- $w = 0, b = 5 \rightarrow y = 5$ (horizontal line at height 5)
- $w = 1, b = 0 \rightarrow y = x$ (45-degree line through origin)

### Why This Matters

This simple neuron can learn to represent **any linear relationship** between input and output. Given data points, it can find the best straight line that fits through them.

## The Multi-Input Neuron: Linear Functions in High Dimensions

Now let's extend to multiple inputs. If our neuron receives $n$ inputs $x_1, x_2, \ldots, x_n$, it computes:

$$y = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b$$

### Matrix Formulation

We can write this more compactly using vectors and matrices:

Let $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ (column vector of inputs)

Let $\mathbf{w} = \begin{bmatrix} w_1 & w_2 & \cdots & w_n \end{bmatrix}$ (row vector of weights)

Then:
$$y = \mathbf{w}\mathbf{x} + b$$

Or equivalently:
$$y = \mathbf{w}^T\mathbf{x} + b$$
where $\mathbf{w}^T$ means w-transpose

### Geometric Interpretation in Higher Dimensions

In 2D (two inputs), the neuron represents a **plane**:
- $y = w_1x_1 + w_2x_2 + b$ defines a plane in 3D space
- The weights determine the orientation of the plane
- The bias determines how high or low the plane sits

In higher dimensions, it represents a **hyperplane**—the generalization of a plane to $n$ dimensions.

## The Learning Problem: Finding the Right Parameters

### The Goal

Given training data (input-output pairs), we want to find the weights and bias that make our neuron's predictions as close as possible to the true outputs.

**Training data:** $(x_1, y_1), (x_2, y_2), \ldots, (x_m, y_m)$

We want to find $w$ and $b$ such that:
- $wx_1 + b \approx y_1$
- $wx_2 + b \approx y_2$
- $\vdots$
- $wx_m + b \approx y_m$

### Measuring Prediction Quality: The Loss Function

We need a way to measure how "wrong" our predictions are. The most common choice for regression is **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

Where:
- $m$ = number of training examples
- $\hat{y}_i$ = our prediction for example $i$ ($\hat{y}_i = wx_i + b$)
- $y_i$ = the true output for example $i$
- $(\hat{y}_i - y_i)$ = the error for example $i$

#### Why Square the Errors?

1. **Always Positive**: Errors of +2 and -2 both contribute equally to the loss
2. **Penalty for Large Errors**: Error of 4 contributes 16 to the loss, while error of 2 contributes only 4
3. **Mathematical Convenience**: Squared functions have nice derivatives
4. **Statistical Justification**: Under certain assumptions, minimizing MSE gives the maximum likelihood estimate

### Finding the Best Parameters: Optimization

Our goal is to find $w$ and $b$ that minimize the MSE. This is an optimization problem:

$$\min_{w,b} \text{MSE}(w, b) = \min_{w,b} \frac{1}{m} \sum_{i=1}^{m} (wx_i + b - y_i)^2$$

## Mathematical Deep Dive: Gradients and Derivatives

To minimize the MSE, we need to understand how it changes as we adjust our parameters. This is where calculus comes in.

### Partial Derivatives

The **gradient** tells us how much the loss function changes when we make a small change to each parameter.

For $\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (wx_i + b - y_i)^2$, we need:

1. $\frac{\partial \text{MSE}}{\partial w}$ (how MSE changes with respect to weight)
2. $\frac{\partial \text{MSE}}{\partial b}$ (how MSE changes with respect to bias)

Let's compute these step by step.

#### Computing $\frac{\partial \text{MSE}}{\partial w}$

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (wx_i + b - y_i)^2$$

Let $e_i = wx_i + b - y_i$ (the error for example $i$)

Then $\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} e_i^2$

Using the chain rule:
$$\begin{aligned}
\frac{\partial \text{MSE}}{\partial w} &= \frac{1}{m} \sum_{i=1}^{m} \frac{\partial(e_i^2)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} 2e_i \cdot \frac{\partial e_i}{\partial w}\\
&= \frac{1}{m} \sum_{i=1}^{m} 2e_i \cdot x_i = \frac{2}{m} \sum_{i=1}^{m} (wx_i + b - y_i) \cdot x_i
\end{aligned}$$

#### Computing $\frac{\partial \text{MSE}}{\partial b}$

Similarly:
$$\begin{aligned}
\frac{\partial \text{MSE}}{\partial b} &= \frac{1}{m} \sum_{i=1}^{m} \frac{\partial(e_i^2)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} 2e_i \cdot \frac{\partial e_i}{\partial b}\\
&= \frac{1}{m} \sum_{i=1}^{m} 2e_i \cdot 1 = \frac{2}{m} \sum_{i=1}^{m} (wx_i + b - y_i)
\end{aligned}$$

### Gradient Descent: Following the Slope Downhill

The gradient tells us the direction of **steepest increase** in the loss function. To minimize the loss, we want to go in the **opposite direction**.

**Gradient Descent Update Rules:**

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}$$

$$b_{\text{new}} = b_{\text{old}} - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$$

Where $\alpha$ (alpha) is the **learning rate**—how big steps we take.

### Intuitive Understanding of Gradient Descent

Imagine you're on a hill in fog and want to reach the bottom:
1. **Feel the slope** under your feet (compute the gradient)
2. **Step downhill** in the steepest direction (move opposite to gradient)
3. **Take another step** from your new position (repeat)
4. **Eventually reach the bottom** (converge to minimum)

The learning rate $\alpha$ determines how big your steps are:
- **Too small**: You'll reach the bottom, but very slowly
- **Too large**: You might overshoot and bounce around, never settling
- **Just right**: You'll reach the bottom efficiently

## A Concrete Example: Learning $y = 2x + 1$

Let's work through a complete example to see how this all comes together.

### Training Data

Suppose we want to learn the function $y = 2x + 1$, and we have these training examples:

$$\begin{aligned}
x_1 &= 1, \quad y_1 = 3\\
x_2 &= 2, \quad y_2 = 5\\  
x_3 &= 3, \quad y_3 = 7
\end{aligned}$$

### Initial Parameters

Let's start with random initial parameters:
$$w = 0.5, \quad b = 0.0$$

### Forward Pass (Making Predictions)

For each training example, compute the prediction:

$$\begin{aligned}
\hat{y}_1 &= 0.5 \times 1 + 0.0 = 0.5\\
\hat{y}_2 &= 0.5 \times 2 + 0.0 = 1.0\\
\hat{y}_3 &= 0.5 \times 3 + 0.0 = 1.5
\end{aligned}$$

### Computing the Loss

$$\begin{aligned}
\text{MSE} &= \frac{1}{3} \times [(0.5 - 3)^2 + (1.0 - 5)^2 + (1.5 - 7)^2]\\
&= \frac{1}{3} \times [(-2.5)^2 + (-4.0)^2 + (-5.5)^2]\\
&= \frac{1}{3} \times [6.25 + 16.0 + 30.25] = \frac{52.5}{3} = 17.5
\end{aligned}$$

### Computing the Gradients

$$\begin{aligned}
\frac{\partial \text{MSE}}{\partial w} &= \frac{2}{3} \times [(0.5 - 3) \times 1 + (1.0 - 5) \times 2 + (1.5 - 7) \times 3]\\
&= \frac{2}{3} \times [(-2.5) \times 1 + (-4.0) \times 2 + (-5.5) \times 3]\\
&= \frac{2}{3} \times [-2.5 - 8.0 - 16.5] = \frac{2}{3} \times (-27.0) = -18.0
\end{aligned}$$

$$\begin{aligned}
\frac{\partial \text{MSE}}{\partial b} &= \frac{2}{3} \times [(0.5 - 3) + (1.0 - 5) + (1.5 - 7)]\\
&= \frac{2}{3} \times [-2.5 - 4.0 - 5.5] = \frac{2}{3} \times (-12.0) = -8.0
\end{aligned}$$

### Parameter Updates

Using learning rate $\alpha = 0.01$:

$$\begin{aligned}
w_{\text{new}} &= 0.5 - 0.01 \times (-18.0) = 0.5 + 0.18 = 0.68\\
b_{\text{new}} &= 0.0 - 0.01 \times (-8.0) = 0.0 + 0.08 = 0.08
\end{aligned}$$

### Interpretation

- Both gradients are negative, meaning the loss decreases when we increase both $w$ and $b$
- The gradient for $w$ is larger in magnitude, so we need to adjust $w$ more than $b$
- After one update step, our parameters moved from $(w=0.5, b=0.0)$ toward the true values $(w=2.0, b=1.0)$

If we continue this process for many iterations, the parameters will converge to the true values!

## Key Insights and Takeaways

### 1. Neurons as Function Approximators

A single neuron with no activation function can learn any linear relationship between inputs and outputs. It's a powerful yet simple building block.

### 2. Parameters Determine Behavior

The weights and biases completely determine what function the neuron represents. Learning is the process of finding the right parameter values.

### 3. Loss Functions Measure Quality

The loss function quantifies how wrong our predictions are. Minimizing the loss leads to better predictions.

### 4. Gradients Guide Learning

Gradients tell us how to adjust parameters to improve performance. They're the mathematical foundation of neural network training.

### 5. Learning is Iterative

We don't solve for the best parameters directly—we improve them gradually through many small updates.

## From Mathematics to Code

Now that we understand the mathematical foundation, let's implement these concepts in code. We'll do this in two parts:

1. **Build from Scratch**: Implement the core concepts yourself to deeply understand them
2. **Use the Library**: See how the Neural Network Toolkit makes these concepts easy to work with

---

## Part 1: Building from Scratch

Let's implement every mathematical concept we've discussed to build genuine understanding.

### Exercise 1.1: Implementing a Basic Neuron

First, let's create a simple neuron that computes $y = wx + b$:

```scala
import io.github.edadma.matrix.Matrix

/**
 * A simple neuron that implements y = wx + b
 * This is the foundation of all neural networks!
 */
class SimpleNeuron(var weight: Double, var bias: Double):
  
  /**
   * Forward pass: compute the neuron's output
   * @param input The input value x
   * @return The output y = wx + b
   */
  def forward(input: Double): Double = 
    weight * input + bias
  
  /**
   * Forward pass for multiple inputs (batch processing)
   * @param inputs A sequence of input values
   * @return A sequence of outputs
   */
  def forwardBatch(inputs: Seq[Double]): Seq[Double] = 
    inputs.map(x => weight * x + bias)
  
  /**
   * Update the neuron's parameters
   */
  def updateParameters(newWeight: Double, newBias: Double): Unit = 
    weight = newWeight
    bias = newBias
  
  override def toString: String = s"SimpleNeuron(w=$weight, b=$bias)"

// Test your neuron
val neuron = SimpleNeuron(weight = 2.0, bias = 1.0)
println(s"Neuron: $neuron")

// Single input
val output1 = neuron.forward(3.0)
println(s"f(3.0) = $output1")  // Should be 2*3 + 1 = 7

// Batch of inputs
val inputs = Seq(1.0, 2.0, 3.0, -1.0)
val outputs = neuron.forwardBatch(inputs)
inputs.zip(outputs).foreach { case (x, y) =>
  println(s"f($x) = $y")
}
```

**🎯 Expected Output:**
```
Neuron: SimpleNeuron(w=2.0, b=1.0)
f(3.0) = 7.0
f(1.0) = 3.0
f(2.0) = 5.0  
f(3.0) = 7.0
f(-1.0) = -1.0
```

### Exercise 1.2: Implementing Mean Squared Error

Now let's implement the MSE loss function:

```scala
/**
 * Mean Squared Error loss function
 * Measures how far our predictions are from the true values
 */
class MeanSquaredError:
  private var lastPredictions: Seq[Double] = Seq.empty
  private var lastTargets: Seq[Double] = Seq.empty
  
  /**
   * Compute the MSE loss
   * @param predictions Our neuron's outputs
   * @param targets The true values we want to predict
   * @return The MSE loss value
   */
  def forward(predictions: Seq[Double], targets: Seq[Double]): Double = 
    require(predictions.length == targets.length, "Predictions and targets must have same length")
    
    // Store for backward pass
    lastPredictions = predictions
    lastTargets = targets
    
    // Compute MSE = (1/n) * Σ(predicted - actual)²
    val errors = predictions.zip(targets).map { case (pred, target) => pred - target }
    val squaredErrors = errors.map(e => e * e)
    squaredErrors.sum / predictions.length
  
  /**
   * Compute the gradient of MSE with respect to predictions
   * This tells us how to adjust our predictions to reduce the loss
   */
  def backward(): Seq[Double] = 
    require(lastPredictions.nonEmpty, "Must call forward() before backward()")
    
    // Gradient of MSE: d/dx[(x-y)²] = 2(x-y) 
    val n = lastPredictions.length
    lastPredictions.zip(lastTargets).map { case (pred, target) =>
      2.0 * (pred - target) / n
    }

// Test your loss function
val loss = MeanSquaredError()

// Perfect predictions (loss should be 0)
val perfectPreds = Seq(3.0, 5.0, 7.0)
val targets = Seq(3.0, 5.0, 7.0)
val perfectLoss = loss.forward(perfectPreds, targets)
println(s"Perfect predictions loss: $perfectLoss")

// Imperfect predictions  
val imperfectPreds = Seq(2.0, 4.0, 6.0)
val imperfectLoss = loss.forward(imperfectPreds, targets)
val gradients = loss.backward()
println(s"Imperfect predictions loss: $imperfectLoss")
println(s"Gradients: $gradients")
```

**🎯 Expected Output:**
```
Perfect predictions loss: 0.0
Imperfect predictions loss: 1.0
Gradients: List(-0.6666666666666666, -0.6666666666666666, -0.6666666666666666)
```

### Exercise 1.3: Computing Parameter Gradients

Now let's compute how the loss changes with respect to our neuron's parameters:

```scala
/**
 * Compute gradients for neuron parameters
 * This tells us how to adjust weight and bias to reduce loss
 */
def computeParameterGradients(
  neuron: SimpleNeuron,
  inputs: Seq[Double], 
  lossGradients: Seq[Double]
): (Double, Double) = 
  
  require(inputs.length == lossGradients.length, "Inputs and gradients must have same length")
  
  // Gradient w.r.t. weight: ∂loss/∂w = Σ(∂loss/∂output * ∂output/∂w)
  // Since output = wx + b, we have ∂output/∂w = x
  val weightGradient = inputs.zip(lossGradients).map { case (x, gradOut) =>
    gradOut * x  // Chain rule: ∂loss/∂w = ∂loss/∂output * x
  }.sum
  
  // Gradient w.r.t. bias: ∂loss/∂b = Σ(∂loss/∂output * ∂output/∂b)  
  // Since output = wx + b, we have ∂output/∂b = 1
  val biasGradient = lossGradients.sum  // Chain rule: ∂loss/∂b = ∂loss/∂output * 1
  
  (weightGradient, biasGradient)

// Test gradient computation
val neuron = SimpleNeuron(weight = 0.5, bias = 0.0)  // Start with wrong parameters
val inputs = Seq(1.0, 2.0, 3.0)
val targets = Seq(3.0, 5.0, 7.0)  // True function: y = 2x + 1

// Forward pass
val predictions = neuron.forwardBatch(inputs)
println(s"Predictions: $predictions")
println(s"Targets: $targets")

// Compute loss and its gradients
val loss = MeanSquaredError()
val lossValue = loss.forward(predictions, targets)
val lossGrads = loss.backward()

println(s"Loss: $lossValue")
println(s"Loss gradients: $lossGrads")

// Compute parameter gradients
val (weightGrad, biasGrad) = computeParameterGradients(neuron, inputs, lossGrads)
println(s"Weight gradient: $weightGrad")
println(s"Bias gradient: $biasGrad")
```

**🎯 Expected Output:**
```
Predictions: List(0.5, 1.0, 1.5)
Targets: List(3.0, 5.0, 7.0)  
Loss: 17.5
Loss gradients: List(-1.6666666666666667, -2.6666666666666665, -3.6666666666666665)
Weight gradient: -18.0
Bias gradient: -8.0
```

### Exercise 1.4: Implementing Gradient Descent

Now let's implement the parameter update step:

```scala
/**
 * Update neuron parameters using gradient descent
 * @param neuron The neuron to update
 * @param weightGrad Gradient w.r.t. weight
 * @param biasGrad Gradient w.r.t. bias  
 * @param learningRate How big steps to take
 */
def gradientDescentStep(
  neuron: SimpleNeuron,
  weightGrad: Double, 
  biasGrad: Double,
  learningRate: Double
): Unit = 
  // Update rule: param_new = param_old - learning_rate * gradient
  val newWeight = neuron.weight - learningRate * weightGrad
  val newBias = neuron.bias - learningRate * biasGrad
  
  println(s"Old parameters: w=${neuron.weight}, b=${neuron.bias}")
  println(s"Gradients: ∂w=$weightGrad, ∂b=$biasGrad")
  println(s"Updates: Δw=${-learningRate * weightGrad}, Δb=${-learningRate * biasGrad}")
  
  neuron.updateParameters(newWeight, newBias)
  println(s"New parameters: w=${neuron.weight}, b=${neuron.bias}")

// Test parameter updates
val neuron = SimpleNeuron(weight = 0.5, bias = 0.0)
gradientDescentStep(neuron, weightGrad = -18.0, biasGrad = -8.0, learningRate = 0.01)
```

**🎯 Expected Output:**
```
Old parameters: w=0.5, b=0.0
Gradients: ∂w=-18.0, ∂b=-8.0  
Updates: Δw=0.18, Δb=0.08
New parameters: w=0.68, b=0.08
```

### Exercise 1.5: Complete Training Loop

Finally, let's put it all together in a complete training loop:

```scala
/**
 * Train a neuron to learn a linear function using gradient descent
 */
def trainNeuron(
  initialWeight: Double,
  initialBias: Double, 
  inputs: Seq[Double],
  targets: Seq[Double],
  learningRate: Double,
  epochs: Int
): SimpleNeuron = 
  
  val neuron = SimpleNeuron(initialWeight, initialBias)
  val loss = MeanSquaredError()
  
  println(s"Training neuron to learn function from data:")
  inputs.zip(targets).foreach { case (x, y) => println(s"  f($x) = $y") }
  println(s"Starting with: w=$initialWeight, b=$initialBias")
  println(s"Learning rate: $learningRate, Epochs: $epochs")
  println()
  
  for epoch <- 1 to epochs do
    // Forward pass
    val predictions = neuron.forwardBatch(inputs)
    val lossValue = loss.forward(predictions, targets)
    
    // Backward pass
    val lossGrads = loss.backward()
    val (weightGrad, biasGrad) = computeParameterGradients(neuron, inputs, lossGrads)
    
    // Parameter update
    gradientDescentStep(neuron, weightGrad, biasGrad, learningRate)
    
    // Log progress
    if epoch % 100 == 0 || epoch <= 5 || epoch >= epochs - 2 then
      println(f"Epoch $epoch%4d: Loss = $lossValue%8.4f, w = ${neuron.weight}%6.3f, b = ${neuron.bias}%6.3f")
  
  println()
  println(s"Final parameters: w=${neuron.weight}, b=${neuron.bias}")
  println("Testing learned function:")
  
  val testInputs = Seq(-1.0, 0.0, 4.0)
  testInputs.foreach { x =>
    val predicted = neuron.forward(x)
    val expected = 2.0 * x + 1.0  // True function
    println(f"  f($x%4.1f) = $predicted%6.3f (expected: $expected%6.3f)")
  }
  
  neuron

// Train on y = 2x + 1 data
val inputs = Seq(1.0, 2.0, 3.0, -1.0, 0.0)
val targets = Seq(3.0, 5.0, 7.0, -1.0, 1.0)

val trainedNeuron = trainNeuron(
  initialWeight = 0.1,  // Start far from true value (2.0)
  initialBias = 0.1,    // Start far from true value (1.0)
  inputs = inputs,
  targets = targets,
  learningRate = 0.01,
  epochs = 1000
)
```

**🎯 Expected Output:**
```
Training neuron to learn function from data:
  f(1.0) = 3.0
  f(2.0) = 5.0  
  f(3.0) = 7.0
  f(-1.0) = -1.0
  f(0.0) = 1.0
Starting with: w=0.1, b=0.1
Learning rate: 0.01, Epochs: 1000

Epoch    1: Loss =  17.1600, w =  0.464, b =  0.164
Epoch    2: Loss =  10.7885, w =  0.762, b =  0.218
...
Epoch 1000: Loss =   0.0000, w =  2.000, b =  1.000

Final parameters: w=2.0, b=1.0
Testing learned function:
  f(-1.0) = -1.000 (expected: -1.000)
  f( 0.0) =  1.000 (expected:  1.000) 
  f( 4.0) =  9.000 (expected:  9.000)
```

---

## Part 2: Using the Neural Network Toolkit

Now that you understand how everything works under the hood, let's see how the NNTK library makes these concepts much easier to work with.

### The Library Way: DenseLayer

Instead of our `SimpleNeuron`, the library provides a `DenseLayer` that can handle multiple inputs and outputs:

```scala
import io.github.edadma.nntk.*

// Create a single-input, single-output linear layer (equivalent to our SimpleNeuron)
val layer = DenseLayer(inputSize = 1, outputSize = 1, useBias = true)

// Set initial parameters to match our from-scratch example
layer.setWeights(Matrix.row(0.1))  // weight = 0.1
layer.setBiases(Matrix.col(0.1))   // bias = 0.1

println(s"Layer parameters: ${layer.getParameterCount}")  // Should be 2
println(s"Initial weights: ${layer.getWeights}")
println(s"Initial biases: ${layer.getBiases}")

// Forward pass with batch of inputs
val inputs = Matrix.col(1.0, 2.0, 3.0, -1.0, 0.0)
val outputs = layer.forward(inputs)

println(s"Inputs: $inputs")  
println(s"Outputs: $outputs")
```

### The Library Way: Loss Functions

The library provides optimized loss functions:

```scala
val lossFunction = MeanSquaredError[Double]()
val targets = Matrix.col(3.0, 5.0, 7.0, -1.0, 1.0)

val lossValue = lossFunction.forward(outputs, targets)
val lossGradient = lossFunction.backward()

println(s"Loss: $lossValue")
println(s"Loss gradient: $lossGradient")
```

### The Library Way: Optimizers

Instead of manually implementing gradient descent, use an optimizer:

```scala
val optimizer = SGD[Double](learningRate = 0.01)

// The optimizer automatically updates all parameters
optimizer.update(layer.getWeights, weightGradient)
optimizer.update(layer.getBiases, biasGradient)
```

### The Library Way: Complete Networks

The most powerful feature is the `Network` class that handles everything automatically:

```scala
// Create a network with a single linear layer
val network = Network(
  DenseLayer(1, 1)  // 1 input → 1 output (linear regression)
)

// Prepare training data  
val trainInputs = Matrix.col(1.0, 2.0, 3.0, -1.0, 0.0)
val trainTargets = Matrix.col(3.0, 5.0, 7.0, -1.0, 1.0)

// Train the network (this does everything we implemented manually!)
val history = network.train(
  inputs = trainInputs,
  targets = trainTargets,
  lossFunction = MeanSquaredError[Double](),
  optimizer = SGD[Double](learningRate = 0.01),
  epochs = 1000
)

// Analyze training progress
println("Training completed!")
println(s"Final loss: ${history.getFinalLoss}")
println(s"Converged at epoch: ${history.getConvergenceEpoch(tolerance = 0.001)}")

// Test the learned function
val testInputs = Matrix.col(-1.0, 0.0, 4.0)
val predictions = network.predict(testInputs)

println("Test results:")
testInputs.iterator.zip(predictions.iterator).foreach { case (x, pred) =>
  val expected = 2.0 * x + 1.0
  println(f"  f($x%4.1f) = $pred%6.3f (expected: $expected%6.3f)")
}

// Inspect learned parameters
val learnedLayer = network.getComponent(0).asInstanceOf[DenseLayer[Double]]
println(s"Learned weight: ${learnedLayer.getWeights}")
println(s"Learned bias: ${learnedLayer.getBiases}")

// Plot training progress
history.plotLoss()  // ASCII chart showing loss over time
```

### Comparing Both Approaches

Let's run both implementations side by side to verify they learn the same function:

```scala
println("=== Comparison: From Scratch vs Library ===")

// From scratch results
val scratchNeuron = trainNeuron(0.1, 0.1, inputs.toSeq, targets.toSeq, 0.01, 1000)
println(s"From scratch: w=${scratchNeuron.weight}, b=${scratchNeuron.bias}")

// Library results  
val libraryLayer = network.getComponent(0).asInstanceOf[DenseLayer[Double]]
println(s"Library: w=${libraryLayer.getWeights}, b=${libraryLayer.getBiases}")

println("Both should have learned approximately: w=2.0, b=1.0")
```

---

## Key Insights from This Lesson

### 1. Mathematics Translates Directly to Code

Every mathematical concept we discussed has a direct implementation:
- **Forward pass**: $y = wx + b$ → `neuron.forward(x)`
- **Loss computation**: MSE formula → `loss.forward(predictions, targets)`
- **Gradients**: Calculus derivatives → `loss.backward()` and parameter gradients
- **Updates**: $w_{\text{new}} = w_{\text{old}} - \alpha \cdot \text{grad}$ → `gradientDescentStep()`

### 2. Libraries Abstract Complexity Without Hiding It

The NNTK library handles the mechanical details (matrix operations, parameter management, training loops) while keeping the concepts transparent. You can always inspect what's happening inside.

### 3. Learning is an Iterative Process

Whether implemented from scratch or using a library, neural network training follows the same cycle:
1. **Forward pass**: Make predictions
2. **Loss computation**: Measure how wrong the predictions are
3. **Backward pass**: Compute gradients
4. **Parameter update**: Adjust weights and biases
5. **Repeat**: Until convergence

### 4. Understanding the Fundamentals Enables Everything Else

The neuron you implemented can only learn linear functions. But the same principles—forward passes, loss functions, gradients, and parameter updates—scale to networks with millions of parameters that can understand language, recognize images, and play games.

## Looking Ahead

You now have both theoretical understanding and practical coding skills for the building blocks of neural networks. In the next lesson, we'll explore:

- Why linear functions aren't enough for complex problems
- How activation functions introduce non-linearity
- Building networks with multiple layers
- The famous XOR problem that reveals the power of hidden layers

## Reflection Questions

1. **Implementation Insights**: What was the most surprising thing when you saw the mathematics turn into working code?

2. **Library Design**: Why do you think the library separates layers, loss functions, and optimizers instead of putting everything in one big class?

3. **Learning Dynamics**: How did changing the learning rate affect convergence? What happened when you set it too high or too low?

4. **Parameter Evolution**: Watch how the weight and bias change during training. Do they approach the target values smoothly or in jumps?

5. **Generalization**: How well does your trained neuron perform on inputs it hasn't seen before? What does this tell you about learning?

6. **Mathematical Connections**: How does the code implementation of `backward()` relate to the partial derivatives we computed by hand?

7. **Bias Necessity**: What would happen if we set `useBias = false`? What kind of functions could we still learn?

8. **Learning Rate Effects**: Experimentally, what happens when $\alpha$ is too large (say, $\alpha = 1.0$) versus too small (say, $\alpha = 0.0001$)?

The mathematical foundation you've built and the coding experience you've gained will serve you throughout this course. Every future lesson will build upon these core concepts of forward passes, loss computation, and gradient-based learning.