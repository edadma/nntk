package io.github.edadma.nntk

import io.github.edadma.logger._
import io.github.edadma.matrix._
import io.github.edadma.table._
import scala.util.Random
import scala.compiletime.uninitialized

// =============================================================================
// CONFIGURATION - Easy to change these values!
// =============================================================================
object Config {
  // Learning rate: how big steps to take when updating weights (0.1 = small steps, 10.0 = big steps)
  val learningRate = 1.0

  // How many times to show the network all the training data
  val epochs = 5000

  // Network architecture: how many neurons in each layer
  val inputSize  = 2 // XOR has 2 inputs (first number, second number)
  val hiddenSize = 3 // 3 neurons in the middle layer (you can try 2, 4, 5, etc.)
  val outputSize = 1 // XOR has 1 output (the result)

  // How often to print progress (every N epochs)
  val logEveryNEpochs = 100

  // When to stop training (if loss gets this low, we're done!)
  val targetLoss = 0.01

  // Random seed for reproducible results (change this to get different random starting weights)
  val randomSeed = 42
}

// =============================================================================
// ACTIVATION FUNCTIONS
// =============================================================================
trait ActivationFunction {
  def apply(x: Double): Double      // The function itself
  def derivative(x: Double): Double // How much the function is changing at point x
  def name: String
}

// Sigmoid: squashes any number into range 0 to 1, looks like an S-curve
object Sigmoid extends ActivationFunction {
  def apply(x: Double): Double = 1.0 / (1.0 + math.exp(-x))

  def derivative(x: Double): Double = {
    val s = apply(x)
    s * (1.0 - s) // Nice property: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
  }

  val name = "Sigmoid"
}

// =============================================================================
// NEURAL NETWORK LAYER
// =============================================================================
class Layer(
    val inputSize: Int,
    val outputSize: Int,
    val activation: ActivationFunction,
    val name: String,
)(implicit logger: Logger) {

  // The "knowledge" of this layer: weights and biases
  // Weights: how much each input affects each output
  // Biases: baseline activation for each output neuron
  var weights: Matrix[Double] = Matrix.fill(outputSize, inputSize)(Random.nextGaussian() * 0.5)
  var biases: Matrix[Double]  = Matrix.fill(outputSize, 1)(0.0)

  // Remember these for backpropagation
  private var lastInput: Matrix[Double]         = uninitialized
  private var lastOutput: Matrix[Double]        = uninitialized
  private var lastPreActivation: Matrix[Double] = uninitialized

  logger.debug(s"Created $name layer: ${inputSize} inputs → ${outputSize} outputs")
  logger.debug(s"Initial weights shape: ${weights.rows}x${weights.cols}, biases: ${biases.rows}x${biases.cols}")

  // Log initial weight/bias ranges for debugging initialization issues
  val weightRange = f"[${weights.min}%.3f to ${weights.max}%.3f]"
  val biasRange   = f"[${biases.min}%.3f to ${biases.max}%.3f]"
  logger.debug(s"$name initial weight range: $weightRange, bias range: $biasRange")

  // Forward pass: calculate this layer's output given an input
  def forward(input: Matrix[Double]): Matrix[Double] = {
    require(input.rows == inputSize, s"Input size mismatch: expected $inputSize, got ${input.rows}")
    require(input.cols == 1, s"Input must be a column vector, got ${input.rows}x${input.cols}")

    lastInput = input

    // Linear transformation: output = weights * input + biases
    lastPreActivation = weights * input + biases

    // Apply activation function to each element
    lastOutput = lastPreActivation.map(activation.apply)

    // Debug check for numerical issues
    if (lastOutput.exists(x => x.isNaN || x.isInfinite)) {
      logger.debug(s"WARNING: $name forward pass produced NaN/Infinite values!", category = "ERROR")
      logger.debug(s"  Input: ${input.transpose}", category = "ERROR")
      logger.debug(s"  Output: ${lastOutput.transpose}", category = "ERROR")
    }

    lastOutput
  }

  // Backward pass: calculate gradients and update weights
  def backward(outputError: Matrix[Double], learningRate: Double): Matrix[Double] = {
    require(outputError.rows == outputSize, s"Output error size mismatch")
    require(outputError.cols == 1, s"Output error must be a column vector")

    // Calculate the derivative of the activation function
    val activationDerivative = lastPreActivation.map(activation.derivative)

    // Delta: how much each neuron should change (error * how sensitive the activation is)
    val delta = outputError.elemMul(activationDerivative)

    // Check for gradient explosion/vanishing
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

    // Calculate gradients for weights and biases
    val weightGradient = delta * lastInput.transpose // outer product
    val biasGradient   = delta

    // Update weights and biases (gradient descent)
    weights = weights - weightGradient * learningRate
    biases = biases - biasGradient * learningRate

    // Check for weight explosion
    val maxWeight = weights.map(math.abs).max
    if (maxWeight > 100.0) {
      logger.debug(f"WARNING: Large weights detected in $name: max weight = $maxWeight%.6f", category = "ERROR")
    }

    // Calculate error to pass to previous layer
    val inputError = weights.transpose * delta

    inputError
  }

  def getWeightsInfo: String = {
    s"$name weights: ${weights.rows}x${weights.cols}, biases: ${biases.rows}x${biases.cols}"
  }
}

// =============================================================================
// NEURAL NETWORK
// =============================================================================
class NeuralNetwork(layers: Layer*)(implicit logger: Logger) {

  logger.info(s"Created neural network with ${layers.length} layers")
  layers.zipWithIndex.foreach { case (layer, i) =>
    logger.info(s"  Layer ${i + 1}: ${layer.getWeightsInfo}")
  }

  // Forward pass: run input through all layers
  def predict(input: Matrix[Double]): Matrix[Double] = {
    layers.foldLeft(input) { (currentInput, layer) =>
      layer.forward(currentInput)
    }
  }

  // Calculate how wrong our prediction is (mean squared error)
  def calculateLoss(predictions: Matrix[Double], targets: Matrix[Double]): Double = {
    require(
      predictions.rows == targets.rows && predictions.cols == targets.cols,
      "Predictions and targets must have same dimensions",
    )

    val errors        = predictions - targets
    val squaredErrors = errors.elemMul(errors) // element-wise multiplication

    // Sum all squared errors and divide by number of samples
    val totalError = squaredErrors.toSeq.sum
    totalError / predictions.rows
  }

  // Backward pass: update all layers
  def train(input: Matrix[Double], target: Matrix[Double], learningRate: Double): Double = {
    // Forward pass
    val prediction = predict(input)

    // Calculate loss
    val loss = calculateLoss(prediction, target)

    // Check for training issues
    if (loss.isNaN || loss.isInfinite) {
      logger.debug(s"WARNING: Loss is ${loss} for input ${input.transpose}", category = "ERROR")
    }

    // Calculate initial error (how wrong we were)
    val outputError = prediction - target

    // Backward pass through all layers (in reverse order)
    layers.foldRight(outputError) { (layer, error) =>
      layer.backward(error, learningRate)
    }

    loss
  }
}

// =============================================================================
// TRAINER
// =============================================================================
class Trainer(network: NeuralNetwork)(implicit logger: Logger) {

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

    var averageLoss = Double.MaxValue // Initialize to a high value
    var epoch       = 1

    while (epoch <= epochs && averageLoss >= targetLoss) {
      var totalLoss = 0.0

      // Train on each sample
      for ((input, target) <- trainingData) {
        val loss = network.train(input, target, learningRate)
        totalLoss += loss
      }

      averageLoss = totalLoss / trainingData.length

      // Debug logging for first few epochs and problematic cases
      if (epoch <= 3) {
        logger.debug(f"Epoch $epoch: Detailed loss = $averageLoss%.6f", category = "EARLY")
      }
      if (averageLoss > 1.0 && epoch > 100) {
        logger.debug(f"WARNING: High loss at epoch $epoch: $averageLoss%.6f", category = "ERROR")
      }

      // Log progress periodically
      if (epoch % logEveryNEpochs == 0 || epoch == 1) {
        val progressMessage = s"Epoch $epoch/$epochs - Average Loss: ${f"$averageLoss%.6f"}"
        logger.info(progressMessage)
        println(s"${Console.CYAN}$progressMessage${Console.RESET}")

        // Test current predictions
        logger.debug("Current predictions:", category = "TRAINING")
        trainingData.foreach { case (input, target) =>
          val prediction = network.predict(input)
          logger.debug(f"  Input: ${input.transpose} → Target: ${target(1, 1)}%.0f, Predicted: ${prediction(1, 1)}%.4f")
        }
      }

      epoch += 1
    }

    // Check if we reached target loss
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

  def test(testData: Seq[(Matrix[Double], Matrix[Double])]): Unit = {
    logger.info("=== FINAL TEST RESULTS ===")

    // Create plain text table for logging (no ANSI codes)
    val logTable = new TextTable() {
      noansi() // Disable ANSI sequences for clean log output
      header("Input 1", "Input 2", "Target", "Predicted", "Correct?")

      testData.foreach { case (input, target) =>
        val prediction = network.predict(input)
        val predicted  = prediction(1, 1)
        val targetVal  = target(1, 1)
        val isCorrect  = math.abs(predicted - targetVal) < 0.5

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

    // Log the plain text version (no ANSI codes)
    logger.info("Test Results Table:")
    logTable.toString.split('\n').foreach(line => logger.info(line))

    // Create colorful table for console output
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

    // Print colorful version to console
    println()
    println(s"${Console.BOLD}${Console.CYAN}=== FINAL TEST RESULTS ===${Console.RESET}")
    println(colorTable)

    // Calculate and display accuracy
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
// XOR EXAMPLE
// =============================================================================
@main def run(args: String*): Unit = {
  // Set up logging
  implicit val logger: Logger = LoggerFactory.getLogger
  logger.setLogLevel(LogLevel.DEBUG) // Enable debug logging for detailed output
  LoggerFactory.setFileLogging()

  // Colorful console welcome
  println(s"${Console.BOLD}${Console.BLUE}=== Neural Network Toolkit - XOR Example ===${Console.RESET}")
  println(
    s"${Console.YELLOW}Configuration: ${Config.epochs} epochs, learning rate ${Config.learningRate}${Console.RESET}",
  )
  println()

  // Plain logging
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

  // Set random seed for reproducible results
  Random.setSeed(Config.randomSeed)

  // Create the XOR training data
  // XOR truth table: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
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

  // Create the neural network
  val hiddenLayer = new Layer(Config.inputSize, Config.hiddenSize, Sigmoid, "Hidden")
  val outputLayer = new Layer(Config.hiddenSize, Config.outputSize, Sigmoid, "Output")
  val network     = new NeuralNetwork(hiddenLayer, outputLayer)

  // Create trainer and start training
  val trainer = new Trainer(network)

  println(s"${Console.CYAN}Training neural network...${Console.RESET}")
  trainer.train(
    trainingData,
    Config.epochs,
    Config.learningRate,
    Config.logEveryNEpochs,
    Config.targetLoss,
  )

  // Test the final network
  trainer.test(trainingData)

  logger.info("Training completed!")
  println(s"${Console.BOLD}${Console.GREEN}Training completed successfully!${Console.RESET}")
}
