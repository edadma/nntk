package io.github.edadma.nntk

import io.github.edadma.matrix.Matrix

class NeuralNetwork(layerSizes: List[Int], activations: List[String]) {
  require(layerSizes.length >= 2, "Need at least input and output layers")
  require(
    activations.length == layerSizes.length - 1,
    "Need activation function for each layer transition",
  )

  // Create layers
  val layers: List[Layer] = layerSizes.zip(layerSizes.tail).zip(activations).map {
    case ((inputSize, outputSize), activation) => new Layer(inputSize, outputSize, activation)
  }

  def forward(input: Matrix[Double]): Matrix[Double] = {
    layers.foldLeft(input) { (current, layer) =>
      layer.forward(current)
    }
  }

  def train(input: Matrix[Double], target: Matrix[Double], learningRate: Double = 0.01): Double = {
    // Forward pass
    val output = forward(input)

    // Calculate mean squared error
    val error = target - output
    val loss  = error.map(x => x * x).sum / error.length

    // Backward pass
    var currentError = error.map(_ * 2) // MSE derivative

    for (layer <- layers.reverse) {
      currentError = layer.backward(currentError, learningRate)
    }

    loss
  }

  def predict(input: Matrix[Double]): Matrix[Double] = forward(input)

  // Add regularization
  def trainWithL2(
      input: Matrix[Double],
      target: Matrix[Double],
      learningRate: Double = 0.01,
      l2Lambda: Double = 0.001,
  ): Double = {
    val loss = train(input, target, learningRate)

    // Apply L2 regularization to weights
    for (layer <- layers) {
      layer.weights = layer.weights.map(w => w * (1 - learningRate * l2Lambda))
    }

    loss
  }
}
