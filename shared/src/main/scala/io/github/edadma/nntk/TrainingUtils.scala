package io.github.edadma.nntk

import io.github.edadma.matrix.Matrix

object TrainingUtils {
  // Mini-batch training
  def trainMiniBatch(
      network: NeuralNetwork,
      inputs: List[Matrix[Double]],
      targets: List[Matrix[Double]],
      batchSize: Int = 32,
      learningRate: Double = 0.01,
  ): Double = {
    val batches   = inputs.zip(targets).grouped(batchSize).toList
    var totalLoss = 0.0

    for (batch <- batches) {
      var batchLoss = 0.0
      for ((input, target) <- batch) {
        batchLoss += network.train(input, target, learningRate)
      }
      totalLoss += batchLoss / batch.length
    }

    totalLoss / batches.length
  }

  // Learning rate scheduling
  def adaptiveLearningRate(initialRate: Double, epoch: Int, decay: Double = 0.95): Double = {
    initialRate * math.pow(decay, epoch / 10.0)
  }

  // Early stopping
  class EarlyStopping(val patience: Int = 10, val minDelta: Double = 1e-6) {
    private var bestLoss  = Double.MaxValue
    private var waitCount = 0

    def shouldStop(currentLoss: Double): Boolean = {
      if (currentLoss < bestLoss - minDelta) {
        bestLoss = currentLoss
        waitCount = 0
        false
      } else {
        waitCount += 1
        waitCount >= patience
      }
    }
  }
}
