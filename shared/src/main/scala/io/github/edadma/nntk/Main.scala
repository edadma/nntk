package io.github.edadma.nntk

import io.github.edadma.logger._
import io.github.edadma.matrix._
import io.github.edadma.table._
import scala.util.Random
import scala.compiletime.uninitialized

// =============================================================================
// CONFIGURATION - Easy to change these values!
// =============================================================================

/** Configuration object containing all hyperparameters and settings for the neural network.
  *
  * This centralizes all the key parameters that affect training behavior, making it easy to experiment with different
  * settings without hunting through the code.
  */
object Config {

  /** Learning rate: Controls how big steps the optimizer takes when updating weights.
    *
    *   - Small values (0.01-0.1): Slow but stable learning, less likely to overshoot
    *   - Medium values (0.1-1.0): Balanced learning speed
    *   - Large values (1.0+): Fast learning but may overshoot optimal values
    */
  val learningRate = 1.0

  /** Maximum number of training epochs (complete passes through all training data).
    *
    * Each epoch shows the network all training examples once. More epochs = more learning opportunities, but also risk
    * of overfitting on small datasets like XOR.
    */
  val epochs = 5000

  /** Network architecture: defines the shape of our neural network.
    *
    * For XOR: 2 inputs → hidden layer → 1 output The hidden layer size affects the network's capacity to learn complex
    * patterns.
    */
  val inputSize  = 2 // XOR has 2 binary inputs (first bit, second bit)
  val hiddenSize = 3 // Number of neurons in hidden layer (try 2, 4, 5, etc.)
  val outputSize = 1 // XOR has 1 binary output (result of XOR operation)

  /** Logging frequency: how often to print training progress.
    *
    * Lower values = more frequent updates but more console spam. Higher values = less frequent updates but cleaner
    * output.
    */
  val logEveryNEpochs = 100

  /** Target loss threshold: training stops when average loss drops below this value.
    *
    * Lower values = higher accuracy required before stopping. This implements "early stopping" to avoid overtraining.
    */
  val targetLoss = 0.01

  /** Random seed for reproducible results.
    *
    * Same seed = same initial weights = same training trajectory. Change this to experiment with different random
    * initializations.
    */
  val randomSeed = 42
}

// =============================================================================
// ACTIVATION FUNCTIONS
// =============================================================================

/** Trait defining the interface for activation functions used in neural networks.
  *
  * Activation functions introduce non-linearity, allowing networks to learn complex patterns. Without activation
  * functions, multiple layers would collapse to a single linear transformation.
  */
trait ActivationFunction {

  /** Apply the activation function to a single value.
    *
    * @param x
    *   Input value (typically the weighted sum of inputs to a neuron)
    * @return
    *   Activated output value
    */
  def apply(x: Double): Double

  /** Compute the derivative of the activation function at a given point.
    *
    * The derivative tells us how sensitive the function is to changes in input. This is crucial for backpropagation -
    * it determines how much to adjust weights.
    *
    * @param x
    *   Input value where derivative is computed
    * @return
    *   Derivative value (rate of change) at point x
    */
  def derivative(x: Double): Double

  /** Human-readable name for this activation function.
    */
  def name: String
}

/** Sigmoid activation function: maps any real number to range (0, 1).
  *
  * Mathematical formula: σ(x) = 1 / (1 + e^(-x))
  *
  * Properties:
  *   - S-shaped curve, smooth and differentiable everywhere
  *   - Output range: (0, 1), making it suitable for binary classification
  *   - Saturates at extremes (derivative approaches 0), which can slow learning
  *   - Historically popular but largely replaced by ReLU in modern deep networks
  *
  * Perfect for XOR because:
  *   - Outputs can be interpreted as probabilities
  *   - Smooth gradients help with the optimization landscape
  */
object Sigmoid extends ActivationFunction {

  /** Compute sigmoid function: 1 / (1 + e^(-x))
    *
    * @param x
    *   Input value (can be any real number)
    * @return
    *   Value between 0 and 1
    */
  def apply(x: Double): Double = 1.0 / (1.0 + math.exp(-x))

  /** Compute sigmoid derivative using the convenient property: σ'(x) = σ(x) * (1 - σ(x))
    *
    * This means we can compute the derivative from the function value itself, which is computationally efficient during
    * backpropagation.
    *
    * @param x
    *   Input value where derivative is needed
    * @return
    *   Derivative value, indicating sensitivity to input changes
    */
  def derivative(x: Double): Double = {
    val s = apply(x)
    s * (1.0 - s) // Elegant property of sigmoid function
  }

  val name = "Sigmoid"
}

// =============================================================================
// NEURAL NETWORK LAYER
// =============================================================================

/** Represents a single layer in a feedforward neural network.
  *
  * A layer performs two main operations:
  *   1. Forward pass: transforms input through weights, biases, and activation function
  *   2. Backward pass: computes gradients and updates weights based on error signal
  *
  * Mathematical operations:
  *   - Forward: output = activation(weights * input + biases)
  *   - Backward: compute gradients via chain rule, update weights via gradient descent
  *
  * @param inputSize
  *   Number of inputs this layer expects
  * @param outputSize
  *   Number of outputs (neurons) this layer produces
  * @param activation
  *   Activation function applied to weighted sums
  * @param name
  *   Human-readable identifier for logging/debugging
  * @param logger
  *   Implicit logger for debugging and monitoring
  */
class Layer(
    val inputSize: Int,
    val outputSize: Int,
    val activation: ActivationFunction,
    val name: String,
)(implicit logger: Logger) {

  // =============================================================================
  // LEARNABLE PARAMETERS
  // =============================================================================

  /** Weight matrix: the "knowledge" of this layer.
    *
    * Shape: (outputSize × inputSize)
    *   - Each row represents one output neuron
    *   - Each column represents how much that neuron cares about each input
    *   - Initialized with small random values to break symmetry
    *
    * Example for 2→3 layer: weights =
    * [[w11, w12], // neuron 1's connection strengths [w21, w22], // neuron 2's connection strengths [w31, w32]] //
    * neuron 3's connection strengths
    */
  var weights: Matrix[Double] = Matrix.fill(outputSize, inputSize)(Random.nextGaussian() * 0.5)

  /** Bias vector: baseline activation for each neuron.
    *
    * Shape: (outputSize × 1)
    *   - Each neuron gets its own bias term
    *   - Allows neurons to activate even when all inputs are zero
    *   - Initialized to zero (common practice)
    *
    * Biases shift the activation function left/right, providing more modeling flexibility.
    */
  var biases: Matrix[Double] = Matrix.fill(outputSize, 1)(0.0)

  // =============================================================================
  // CACHED VALUES FOR BACKPROPAGATION
  // =============================================================================

  /** Last input received during forward pass. Needed during backward pass to compute weight gradients.
    */
  private var lastInput: Matrix[Double] = uninitialized

  /** Last output produced during forward pass. Currently not used but helpful for debugging.
    */
  private var lastOutput: Matrix[Double] = uninitialized

  /** Last pre-activation values (before applying activation function). Needed during backward pass to compute
    * activation derivatives.
    */
  private var lastPreActivation: Matrix[Double] = uninitialized

  // =============================================================================
  // INITIALIZATION LOGGING
  // =============================================================================

  logger.debug(s"Created $name layer: ${inputSize} inputs → ${outputSize} outputs")
  logger.debug(s"Initial weights shape: ${weights.rows}x${weights.cols}, biases: ${biases.rows}x${biases.cols}")

  // Log weight/bias ranges to help diagnose initialization problems
  val weightRange = f"[${weights.min}%.3f to ${weights.max}%.3f]"
  val biasRange   = f"[${biases.min}%.3f to ${biases.max}%.3f]"
  logger.debug(s"$name initial weight range: $weightRange, bias range: $biasRange")

  /** Forward pass: compute layer output given input vector.
    *
    * Implements the fundamental neural network computation:
    *   1. Linear transformation: z = W * x + b
    *   2. Non-linear activation: a = σ(z)
    *
    * The linear part combines weighted inputs; the activation introduces non-linearity that enables learning complex
    * patterns.
    *
    * @param input
    *   Column vector of inputs, shape (inputSize × 1)
    * @return
    *   Column vector of outputs, shape (outputSize × 1)
    */
  def forward(input: Matrix[Double]): Matrix[Double] = {
    require(input.rows == inputSize, s"Input size mismatch: expected $inputSize, got ${input.rows}")
    require(input.cols == 1, s"Input must be a column vector, got ${input.rows}x${input.cols}")

    // Cache input for backpropagation
    lastInput = input

    // Step 1: Linear transformation (affine transformation)
    // z = W * x + b where W is weights, x is input, b is biases
    lastPreActivation = weights * input + biases

    // Step 2: Apply activation function element-wise
    // a = σ(z) where σ is the activation function
    lastOutput = lastPreActivation.map(activation.apply)

    // Safety check: detect numerical instability early
    if (lastOutput.exists(x => x.isNaN || x.isInfinite)) {
      logger.debug(s"WARNING: $name forward pass produced NaN/Infinite values!", category = "ERROR")
      logger.debug(s"  Input: ${input.transpose}", category = "ERROR")
      logger.debug(s"  Output: ${lastOutput.transpose}", category = "ERROR")
    }

    lastOutput
  }

  /** Backward pass: compute gradients and update weights using backpropagation.
    *
    * Implements the heart of neural network learning:
    *   1. Compute local gradients using chain rule
    *   2. Update weights using gradient descent: w := w - α * ∇w
    *   3. Compute error signal for previous layer
    *
    * The chain rule allows us to compute how much each weight contributed to the final error, enabling precise updates
    * that reduce the error.
    *
    * @param outputError
    *   Error signal from next layer, shape (outputSize × 1)
    * @param learningRate
    *   Step size for gradient descent updates
    * @return
    *   Error signal for previous layer, shape (inputSize × 1)
    */
  def backward(outputError: Matrix[Double], learningRate: Double): Matrix[Double] = {
    require(outputError.rows == outputSize, s"Output error size mismatch")
    require(outputError.cols == 1, s"Output error must be a column vector")

    // Step 1: Compute activation function derivatives
    // These tell us how sensitive each neuron's output is to changes in its input
    val activationDerivative = lastPreActivation.map(activation.derivative)

    // Step 2: Compute delta (local gradient) using chain rule
    // δ = error * σ'(z) where σ' is activation derivative
    // This combines the error signal with local sensitivity
    val delta = outputError.elemMul(activationDerivative)

    // Gradient monitoring: detect exploding/vanishing gradient problems
    val maxDelta = delta.map(math.abs).max
    if (maxDelta > 10.0) {
      logger.debug(f"WARNING: Large gradient detected in $name: max delta = $maxDelta%.6f", category = "ERROR")
    }
    if (maxDelta < 1e-8) {
      logger.debug(
        f"WARNING: Very small gradient in $name: max delta = $maxDelta%.6f (vanishing gradient?)",
        category = "ERROR",
      )
    }

    // Step 3: Compute weight and bias gradients
    // ∇W = δ * x^T (outer product of delta and input)
    // ∇b = δ (bias gradients equal the deltas)
    val weightGradient = delta * lastInput.transpose // Shape: (outputSize × inputSize)
    val biasGradient   = delta                       // Shape: (outputSize × 1)

    // Step 4: Update parameters using gradient descent
    // w := w - α * ∇w where α is learning rate
    // This moves weights in direction that reduces error
    weights = weights - weightGradient * learningRate
    biases = biases - biasGradient * learningRate

    // Weight monitoring: detect exploding weights
    val maxWeight = weights.map(math.abs).max
    if (maxWeight > 100.0) {
      logger.debug(f"WARNING: Large weights detected in $name: max weight = $maxWeight%.6f", category = "ERROR")
    }

    // Step 5: Compute error signal for previous layer
    // This allows error to propagate backwards through the network
    val inputError = weights.transpose * delta

    inputError
  }

  /** Get summary information about this layer's parameters.
    *
    * @return
    *   String describing weight and bias matrix dimensions
    */
  def getWeightsInfo: String = {
    s"$name weights: ${weights.rows}x${weights.cols}, biases: ${biases.rows}x${biases.cols}"
  }
}

// =============================================================================
// NEURAL NETWORK
// =============================================================================

/** Multi-layer feedforward neural network.
  *
  * Chains together multiple layers to create a deep learning model capable of learning complex non-linear mappings from
  * inputs to outputs.
  *
  * Key operations:
  *   - Forward pass: propagate input through all layers to produce prediction
  *   - Training: compute loss and propagate error backwards to update weights
  *
  * @param layers
  *   Variable number of Layer objects defining the network architecture
  * @param logger
  *   Implicit logger for monitoring and debugging
  */
class NeuralNetwork(layers: Layer*)(implicit logger: Logger) {

  logger.info(s"Created neural network with ${layers.length} layers")
  layers.zipWithIndex.foreach { case (layer, i) =>
    logger.info(s"  Layer ${i + 1}: ${layer.getWeightsInfo}")
  }

  /** Forward pass: compute network prediction for given input.
    *
    * Sequentially applies each layer transformation: input → layer1 → layer2 → ... → layerN → prediction
    *
    * Each layer applies: output = activation(weights * input + biases)
    *
    * @param input
    *   Column vector of input features, shape (inputSize × 1)
    * @return
    *   Column vector of predictions, shape (outputSize × 1)
    */
  def predict(input: Matrix[Double]): Matrix[Double] = {
    // Use fold to thread the output of each layer as input to the next
    layers.foldLeft(input) { (currentInput, layer) =>
      layer.forward(currentInput)
    }
  }

  /** Calculate mean squared error between predictions and targets.
    *
    * MSE = (1/n) * Σ(prediction - target)²
    *
    * This measures how far off our predictions are on average. Lower is better. MSE is differentiable, making it
    * suitable for gradient-based optimization.
    *
    * @param predictions
    *   Network outputs, shape (outputSize × 1)
    * @param targets
    *   Desired outputs, shape (outputSize × 1)
    * @return
    *   Scalar loss value (average squared error)
    */
  def calculateLoss(predictions: Matrix[Double], targets: Matrix[Double]): Double = {
    require(
      predictions.rows == targets.rows && predictions.cols == targets.cols,
      "Predictions and targets must have same dimensions",
    )

    // Compute element-wise differences
    val errors = predictions - targets

    // Square each error (penalizes large errors more than small ones)
    val squaredErrors = errors.elemMul(errors) // Element-wise multiplication

    // Compute mean: sum all errors and divide by number of elements
    val totalError = squaredErrors.toSeq.sum
    totalError / predictions.rows
  }

  /** Train the network on a single input-target pair.
    *
    * Implements one complete training step:
    *   1. Forward pass: compute prediction
    *   2. Loss calculation: measure error
    *   3. Backward pass: compute gradients and update weights
    *
    * This is where the network actually learns by adjusting its weights to reduce error.
    *
    * @param input
    *   Training input, shape (inputSize × 1)
    * @param target
    *   Desired output, shape (outputSize × 1)
    * @param learningRate
    *   Step size for weight updates
    * @return
    *   Loss value for this training example
    */
  def train(input: Matrix[Double], target: Matrix[Double], learningRate: Double): Double = {
    // Step 1: Forward pass - get network's current prediction
    val prediction = predict(input)

    // Step 2: Compute loss - how wrong are we?
    val loss = calculateLoss(prediction, target)

    // Safety check: detect numerical problems early
    if (loss.isNaN || loss.isInfinite) {
      logger.debug(s"WARNING: Loss is ${loss} for input ${input.transpose}", category = "ERROR")
    }

    // Step 3: Compute initial error signal
    // For MSE loss: error = prediction - target
    // This represents how much and in what direction each output should change
    val outputError = prediction - target

    // Step 4: Backward pass - propagate error and update weights
    // Process layers in reverse order, threading error backwards
    layers.foldRight(outputError) { (layer, error) =>
      layer.backward(error, learningRate)
    }

    loss
  }
}

// =============================================================================
// TRAINER
// =============================================================================

/** High-level training orchestrator for neural networks.
  *
  * Handles the training loop, progress monitoring, and convergence checking. Provides a clean interface for training
  * networks on datasets.
  *
  * @param network
  *   The neural network to train
  * @param logger
  *   Implicit logger for progress reporting
  */
class Trainer(network: NeuralNetwork)(implicit logger: Logger) {

  /** Train the neural network on a dataset for multiple epochs.
    *
    * An epoch is one complete pass through the entire training dataset. Training continues until either:
    *   - Maximum epochs reached, or
    *   - Target loss achieved (early stopping)
    *
    * @param trainingData
    *   Sequence of (input, target) pairs
    * @param epochs
    *   Maximum number of training epochs
    * @param learningRate
    *   Step size for gradient descent
    * @param logEveryNEpochs
    *   How often to log progress (every N epochs)
    * @param targetLoss
    *   Stop training when average loss drops below this value
    */
  def train(
      trainingData: Seq[(Matrix[Double], Matrix[Double])],
      epochs: Int,
      learningRate: Double,
      logEveryNEpochs: Int = 100,
      targetLoss: Double = 0.01,
  ): Unit = {

    logger.info(s"Starting training for $epochs epochs with learning rate $learningRate")
    logger.info(s"Training data: ${trainingData.length} samples")
    logger.debug(s"Logging every $logEveryNEpochs epochs")
    logger.debug(s"Target loss threshold: $targetLoss")

    var averageLoss = Double.MaxValue // Initialize to high value to enter training loop
    var epoch       = 1

    // Main training loop: continue until max epochs or target loss reached
    while (epoch <= epochs && averageLoss >= targetLoss) {
      var totalLoss = 0.0

      // Train on each example in the dataset (one epoch)
      for ((input, target) <- trainingData) {
        val loss = network.train(input, target, learningRate)
        totalLoss += loss
      }

      // Compute average loss for this epoch
      averageLoss = totalLoss / trainingData.length

      // Early training monitoring: log detailed info for first few epochs
      if (epoch <= 3) {
        logger.debug(f"Epoch $epoch: Detailed loss = $averageLoss%.6f", category = "EARLY")
      }

      // Problem detection: warn if loss is still high after many epochs
      if (averageLoss > 1.0 && epoch > 100) {
        logger.debug(f"WARNING: High loss at epoch $epoch: $averageLoss%.6f", category = "ERROR")
      }

      // Periodic progress reporting
      if (epoch % logEveryNEpochs == 0 || epoch == 1) {
        val progressMessage = s"Epoch $epoch/$epochs - Average Loss: ${f"$averageLoss%.6f"}"
        logger.info(progressMessage)
        println(s"${Console.CYAN}$progressMessage${Console.RESET}")

        // Show current predictions for debugging
        logger.debug("Current predictions:", category = "TRAINING")
        trainingData.foreach { case (input, target) =>
          val prediction = network.predict(input)
          logger.debug(f"  Input: ${input.transpose} → Target: ${target(1, 1)}%.0f, Predicted: ${prediction(1, 1)}%.4f")
        }
      }

      epoch += 1
    }

    // Report final training outcome
    if (averageLoss < targetLoss) {
      val message = f"Target loss $targetLoss%.6f reached at epoch ${epoch - 1}!"
      logger.info(message)
      println(s"${Console.BOLD}${Console.GREEN}$message${Console.RESET}")
    } else {
      val message = s"Training completed after $epochs epochs"
      logger.info(message)
      println(s"${Console.YELLOW}$message${Console.RESET}")
    }
  }

  /** Evaluate the trained network on test data and display results.
    *
    * Tests the network's performance and presents results in both clean log format and colorful console format.
    *
    * @param testData
    *   Sequence of (input, target) pairs for evaluation
    */
  def test(testData: Seq[(Matrix[Double], Matrix[Double])]): Unit = {
    logger.info("=== FINAL TEST RESULTS ===")

    // Create plain text table for clean log output (no ANSI escape codes)
    val logTable = new TextTable() {
      noansi() // Disable ANSI sequences for clean log files
      header("Input 1", "Input 2", "Target", "Predicted", "Correct?")

      testData.foreach { case (input, target) =>
        val prediction = network.predict(input)
        val predicted  = prediction(1, 1)
        val targetVal  = target(1, 1)
        val isCorrect  = math.abs(predicted - targetVal) < 0.5 // Binary classification threshold

        // Log detailed test case information
        logger.debug(
          f"Test case: Input(${input(1, 1)}%.0f,${input(2, 1)}%.0f) → Target: $targetVal%.0f, Predicted: $predicted%.6f, Correct: $isCorrect",
        )

        row(
          f"${input(1, 1)}%.0f",
          f"${input(2, 1)}%.0f",
          f"$targetVal%.0f",
          f"$predicted%.4f",
          if (isCorrect) "YES" else "NO",
        )
      }
    }

    // Log the plain text version (clean for log files)
    logger.info("Test Results Table:")
    logTable.toString.split('\n').foreach(line => logger.info(line))

    // Create colorful table for attractive console output
    val colorTable = new TextTable() {
      header("Input 1", "Input 2", "Target", "Predicted", "Correct?")

      testData.foreach { case (input, target) =>
        val prediction = network.predict(input)
        val predicted  = prediction(1, 1)
        val targetVal  = target(1, 1)
        val isCorrect  = math.abs(predicted - targetVal) < 0.5

        row(
          f"${input(1, 1)}%.0f",
          f"${input(2, 1)}%.0f",
          f"$targetVal%.0f",
          f"$predicted%.4f",
          if (isCorrect) s"${Console.GREEN}✓${Console.RESET}" else s"${Console.RED}✗${Console.RESET}",
        )
      }
    }

    // Display colorful version on console
    println()
    println(s"${Console.BOLD}${Console.CYAN}=== FINAL TEST RESULTS ===${Console.RESET}")
    println(colorTable)

    // Calculate and display overall accuracy
    val correctCount = testData.count { case (input, target) =>
      val prediction = network.predict(input)
      val predicted  = prediction(1, 1)
      val targetVal  = target(1, 1)
      math.abs(predicted - targetVal) < 0.5
    }
    val accuracy = (correctCount.toDouble / testData.length) * 100

    val accuracyMessage = f"Accuracy: $correctCount/${testData.length} ($accuracy%.1f%%)"
    logger.info(accuracyMessage)
    println(s"${Console.BOLD}${Console.GREEN}$accuracyMessage${Console.RESET}")
    println()
  }
}

// =============================================================================
// XOR EXAMPLE - PUTTING IT ALL TOGETHER
// =============================================================================

/** Main application demonstrating neural network learning on the XOR problem.
  *
  * XOR (exclusive OR) is the classic "hello world" of neural networks because:
  *   - Simple to understand: outputs 1 when inputs differ, 0 when they're the same
  *   - Impossible for single-layer networks (not linearly separable)
  *   - Requires hidden layer to learn the non-linear pattern
  *   - Small dataset (4 examples) trains quickly
  *
  * Network architecture: 2 → 3 → 1 (2 inputs, 3 hidden neurons, 1 output)
  *
  * @param args
  *   Command line arguments (currently unused)
  */
@main def run(args: String*): Unit = {
  // =============================================================================
  // SETUP AND CONFIGURATION
  // =============================================================================

  // Initialize logging system with both file and console output
  implicit val logger: Logger = LoggerFactory.getLogger
  logger.setLogLevel(LogLevel.DEBUG) // Enable detailed debug information
  LoggerFactory.setFileLogging()     // Create log file for debugging

  // Display colorful welcome message on console
  println(s"${Console.BOLD}${Console.BLUE}=== Neural Network Toolkit - XOR Example ===${Console.RESET}")
  println(
    s"${Console.YELLOW}Configuration: ${Config.epochs} epochs, learning rate ${Config.learningRate}${Console.RESET}",
  )
  println()

  // Log configuration details to file (plain text, no colors)
  logger.info("=== Neural Network Toolkit - XOR Example ===")
  logger.info(s"Configuration: ${Config.epochs} epochs, learning rate ${Config.learningRate}")
  logger.debug("=== CONFIGURATION DETAILS ===")
  logger.debug(s"Random seed: ${Config.randomSeed}")
  logger.debug(s"Learning rate (step size): ${Config.learningRate}")
  logger.debug(s"Max epochs (training cycles): ${Config.epochs}")
  logger.debug(s"Target loss (stopping criteria): ${Config.targetLoss}")
  logger.debug(s"Log frequency: every ${Config.logEveryNEpochs} epochs")
  logger.debug(s"Network architecture: ${Config.inputSize} → ${Config.hiddenSize} → ${Config.outputSize}")
  logger.debug(s"Activation function: ${Sigmoid.name}")
  logger.debug("================================")

  // Set random seed for reproducible experiments
  Random.setSeed(Config.randomSeed)

  // =============================================================================
  // XOR TRAINING DATA PREPARATION
  // =============================================================================

  /** XOR truth table - the complete dataset for this problem:
    *
    * Input1 | Input2 | Output -------|--------|------- 0 | 0 | 0 (same inputs → 0) 0 | 1 | 1 (different inputs → 1) 1 |
    * 0 | 1 (different inputs → 1) 1 | 1 | 0 (same inputs → 0)
    *
    * This pattern is not linearly separable - you can't draw a single straight line to separate the 0s from the 1s.
    * That's why we need a hidden layer.
    */
  val trainingData = Seq(
    (Matrix.col(0.0, 0.0), Matrix.col(0.0)), // 0 XOR 0 = 0
    (Matrix.col(0.0, 1.0), Matrix.col(1.0)), // 0 XOR 1 = 1
    (Matrix.col(1.0, 0.0), Matrix.col(1.0)), // 1 XOR 0 = 1
    (Matrix.col(1.0, 1.0), Matrix.col(0.0)), // 1 XOR 1 = 0
  )

  logger.info("XOR training data created:")
  trainingData.foreach { case (input, target) =>
    logger.info(s"  ${input.transpose} → ${target(1, 1)}")
  }

  // =============================================================================
  // NEURAL NETWORK CONSTRUCTION
  // =============================================================================

  // Create the network layers
  val hiddenLayer = new Layer(Config.inputSize, Config.hiddenSize, Sigmoid, "Hidden")
  val outputLayer = new Layer(Config.hiddenSize, Config.outputSize, Sigmoid, "Output")

  // Assemble the complete network
  val network = new NeuralNetwork(hiddenLayer, outputLayer)

  // =============================================================================
  // TRAINING PROCESS
  // =============================================================================

  // Create trainer and begin the learning process
  val trainer = new Trainer(network)

  println(s"${Console.CYAN}Training neural network...${Console.RESET}")
  trainer.train(
    trainingData,
    Config.epochs,
    Config.learningRate,
    Config.logEveryNEpochs,
    Config.targetLoss,
  )

  // =============================================================================
  // EVALUATION AND RESULTS
  // =============================================================================

  // Test the trained network on the same data (in a real project, you'd use separate test data)
  trainer.test(trainingData)

  // Final completion message
  logger.info("Training completed!")
  println(s"${Console.BOLD}${Console.GREEN}Training completed successfully!${Console.RESET}")
}
