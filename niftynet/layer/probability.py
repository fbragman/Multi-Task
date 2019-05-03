import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Dirichlet:
    """
    Dirichlet class that is reparameterized according to:
    Figurnov et al. Implicit reparamterization gradients, 2018
    """

    def __init__(self, mean, precision, batch_size=1, num_samples=1):
        """
        batch_size automatically set to 1 --> Dirichlet grouping equal for all batches
        Otherwise, set batch_size to batch_size
        :param mean:
        :param precision:
        :param num_samples:
        """
        self.batch_size = batch_size
        self.alpha = self.calculate_alpha(mean, precision)
        self.num_samples = num_samples

    def __call__(self):
        """
        Sample from a Dirichlet distribution
        :return:
        """
        # create distribution object
        dist = tfd.Dirichlet(self.alpha)
        # sample from the distribution
        samples = dist.sample(self.num_samples)

        if self.num_samples > 1:
            # Calculate expectation
            samples_mean = dist.mean(name='dirichlet_expectation')
        else:
            samples_mean = None

        return tf.squeeze(samples), tf.squeeze(samples_mean)

    def calculate_alpha(self, mean, precision):
        """
        Calculate Dirichlet concentration parameter from mean and precision parameterisation
        :param mean:
        :param precision:
        :return:
        """
        alpha = tf.multiply(mean, precision[:, tf.newaxis])
        if self.batch_size == 1:
            alpha = alpha[tf.newaxis, :]
        return alpha