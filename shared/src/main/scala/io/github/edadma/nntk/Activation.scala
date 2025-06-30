package io.github.edadma.nntk

object Activation {
  def sigmoid(x: Double): Double           = 1.0 / (1.0 + math.exp(-x))
  def sigmoidDerivative(x: Double): Double = {
    val s = sigmoid(x)
    s * (1.0 - s)
  }

  def relu(x: Double): Double           = math.max(0.0, x)
  def reluDerivative(x: Double): Double = if (x > 0) 1.0 else 0.0

  def tanh(x: Double): Double           = math.tanh(x)
  def tanhDerivative(x: Double): Double = {
    val t = tanh(x)
    1.0 - t * t
  }

  def leakyRelu(x: Double, alpha: Double = 0.01): Double =
    if (x > 0) x else alpha * x
  def leakyReluDerivative(x: Double, alpha: Double = 0.01): Double =
    if (x > 0) 1.0 else alpha
}
