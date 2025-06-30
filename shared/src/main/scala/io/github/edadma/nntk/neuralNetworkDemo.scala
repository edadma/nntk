package io.github.edadma.nntk

import io.github.edadma.matrix.Matrix

@main def neuralNetworkDemo(): Unit = {
  println("Neural Network with Matrix Library")
  println("=================================")

  // XOR Problem
  println("\n--- XOR Problem ---")
  val xorInputs = List(
    Matrix.col(0.0, 0.0),
    Matrix.col(0.0, 1.0),
    Matrix.col(1.0, 0.0),
    Matrix.col(1.0, 1.0),
  )

  val xorTargets = List(
    Matrix.col(0.0),
    Matrix.col(1.0),
    Matrix.col(1.0),
    Matrix.col(0.0),
  )

  // Create network: 2 -> 8 -> 4 -> 1
  val xorNetwork = new NeuralNetwork(
    List(2, 8, 4, 1),
    List("relu", "relu", "sigmoid"),
  )

  // Training with early stopping
  val earlyStopping  = new TrainingUtils.EarlyStopping(patience = 50)
  var epoch          = 0
  var shouldContinue = true

  while (epoch < 5000 && shouldContinue) {
    val learningRate = TrainingUtils.adaptiveLearningRate(0.1, epoch)
    val avgLoss      = TrainingUtils.trainMiniBatch(xorNetwork, xorInputs, xorTargets, 4, learningRate)

    if (epoch % 500 == 0) {
      println(f"Epoch $epoch: Loss = $avgLoss%.6f, LR = $learningRate%.4f")
    }

    shouldContinue = !earlyStopping.shouldStop(avgLoss)
    epoch += 1
  }

  println(s"\nTraining completed after $epoch epochs")

  // Test the network
  println("\nTesting XOR Network:")
  for ((input, expected) <- xorInputs.zip(xorTargets)) {
    val prediction   = xorNetwork.predict(input)
    val inputVals    = s"[${input(1, 1)}, ${input(2, 1)}]"
    val expectedVal  = expected(1, 1)
    val predictedVal = prediction(1, 1)

    println(f"Input: $inputVals -> Expected: $expectedVal%.0f, Predicted: $predictedVal%.4f")
  }

  // Iris Classification Demo (simplified)
  println("\n--- Simple Classification Demo ---")

  // Create some sample 2D classification data
  val classificationInputs = List(
    Matrix.col(0.2, 0.3),   // Class 0
    Matrix.col(0.1, 0.4),   // Class 0
    Matrix.col(0.8, 0.7),   // Class 1
    Matrix.col(0.9, 0.8),   // Class 1
    Matrix.col(0.15, 0.25), // Class 0
    Matrix.col(0.85, 0.75), // Class 1
  )

  val classificationTargets = List(
    Matrix.col(1.0, 0.0), // One-hot encoded
    Matrix.col(1.0, 0.0),
    Matrix.col(0.0, 1.0),
    Matrix.col(0.0, 1.0),
    Matrix.col(1.0, 0.0),
    Matrix.col(0.0, 1.0),
  )

  val classificationNetwork = new NeuralNetwork(
    List(2, 6, 4, 2),
    List("relu", "relu", "sigmoid"),
  )

  // Train for classification
  for (epoch <- 1 to 1000) {
    val avgLoss = TrainingUtils.trainMiniBatch(
      classificationNetwork,
      classificationInputs,
      classificationTargets,
      2,
      0.05,
    )

    if (epoch % 200 == 0) {
      println(f"Classification Epoch $epoch: Loss = $avgLoss%.6f")
    }
  }

  println("\nTesting Classification Network:")
  for ((input, expected) <- classificationInputs.zip(classificationTargets)) {
    val prediction     = classificationNetwork.predict(input)
    val inputVals      = f"[${input(1, 1)}%.1f, ${input(2, 1)}%.1f]"
    val predictedClass = if (prediction(1, 1) > prediction(2, 1)) 0 else 1
    val expectedClass  = if (expected(1, 1) > expected(2, 1)) 0 else 1

    println(f"Input: $inputVals -> Expected: Class $expectedClass, Predicted: Class $predictedClass")
  }

  println("\n🎉 Neural Network successfully trained!")
  println("\nNext steps:")
  println("- Add convolutional layers for image processing")
  println("- Implement different optimizers (Adam, RMSprop)")
  println("- Add batch normalization and dropout")
  println("- Integrate with your RTX 4060 for GPU acceleration")
}
