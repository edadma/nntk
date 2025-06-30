// Corrected Layer.scala with proper dimension handling

package io.github.edadma.nntk

import io.github.edadma.matrix.Matrix
import scala.compiletime.uninitialized
import scala.util.Random

class Layer(val inputSize: Int, val outputSize: Int, activationFn: String = "sigmoid") {
  // Initialize weights using Xavier initialization
  var weights: Matrix[Double] = Matrix.fill(outputSize, inputSize) {
    Random.nextGaussian() * math.sqrt(2.0 / (inputSize + outputSize))
  }

  // Initialize biases to small random values
  var biases: Matrix[Double] = Matrix.fill(outputSize, 1) {
    Random.nextGaussian() * 0.01
  }

  // Store values for backpropagation
  var lastInput: Matrix[Double]  = uninitialized
  var lastZ: Matrix[Double]      = uninitialized // Before activation
  var lastOutput: Matrix[Double] = uninitialized // After activation

  def forward(input: Matrix[Double]): Matrix[Double] = {
    require(
      input.rows == inputSize && input.cols == 1,
      s"Expected input of size ($inputSize, 1), got (${input.rows}, ${input.cols})",
    )

    // Store for backpropagation
    lastInput = input

    // Linear transformation: W * x + b
    lastZ = weights * input + biases

    // Apply activation function
    lastOutput = activationFn match {
      case "sigmoid"    => lastZ.map(Activation.sigmoid)
      case "relu"       => lastZ.map(Activation.relu)
      case "tanh"       => lastZ.map(Activation.tanh)
      case "leaky_relu" => lastZ.map(Activation.leakyRelu(_))
      case _            => throw new IllegalArgumentException(s"Unknown activation: $activationFn")
    }

    lastOutput
  }

  def backward(outputError: Matrix[Double], learningRate: Double): Matrix[Double] = {
    // Ensure outputError has correct dimensions
    require(
      outputError.rows == outputSize && outputError.cols == 1,
      s"Expected outputError of size ($outputSize, 1), got (${outputError.rows}, ${outputError.cols})",
    )

    // Calculate activation derivative
    val activationDerivative = activationFn match {
      case "sigmoid"    => lastZ.map(Activation.sigmoidDerivative)
      case "relu"       => lastZ.map(Activation.reluDerivative)
      case "tanh"       => lastZ.map(Activation.tanhDerivative)
      case "leaky_relu" => lastZ.map(Activation.leakyReluDerivative(_))
      case _            => throw new IllegalArgumentException(s"Unknown activation: $activationFn")
    }

    // Calculate delta (chain rule) using element-wise multiplication
    // Both outputError and activationDerivative should be (outputSize x 1)
    val delta = outputError.elemMul(activationDerivative)

    // Calculate gradients
    // Weight gradient: delta (outputSize x 1) * lastInput.transpose (1 x inputSize) = (outputSize x inputSize)
    val weightGradient = delta * lastInput.transpose
    val biasGradient   = delta // Already (outputSize x 1)

    // Update parameters
    weights = weights - weightGradient.map(_ * learningRate)
    biases = biases - biasGradient.map(_ * learningRate)

    // Return error for previous layer
    // weights.transpose (inputSize x outputSize) * delta (outputSize x 1) = (inputSize x 1)
    weights.transpose * delta
  }
}

// Also create a simple XOR test that's easier to debug
@main def simpleXorTest(): Unit = {
  println("Simple XOR Test")
  println("===============")

  // Create a very simple network: 2 -> 2 -> 1
  val network = new NeuralNetwork(List(2, 2, 1), List("sigmoid", "sigmoid"))

  val input  = Matrix.col(1.0, 0.0)
  val target = Matrix.col(1.0)

  println(s"Input: ${input.rows}x${input.cols}")
  println(s"Target: ${target.rows}x${target.cols}")

  try {
    val output = network.forward(input)
    println(s"Output: ${output.rows}x${output.cols}, value: ${output(1, 1)}")

    val loss = network.train(input, target, 0.1)
    println(s"Loss: $loss")

    println("✅ Simple test passed!")

  } catch {
    case e: Exception =>
      println(s"❌ Error: ${e.getMessage}")
      e.printStackTrace()
  }
}
