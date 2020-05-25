import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        y_pred = nn.as_scalar(self.run(x))
        return 1 if y_pred >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            convergence = True
            for x, y in dataset.iterate_once(1):
                y = nn.as_scalar(y)
                y_pred = self.get_prediction(x)
                if y_pred != y:
                    convergence = False
                    self.w.update(x, y)
            if convergence:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 4
        self.learning_rate = 0.01
        # input layer
        self.w0 = nn.Parameter(1, 16)
        self.b0 = nn.Parameter(1, 16)
        # hidden layer
        self.w1 = nn.Parameter(16, 16)
        self.b1 = nn.Parameter(1, 16)
        # output layer
        self.w2 = nn.Parameter(16, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        x = nn.ReLU(x)
        x = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        x = nn.ReLU(x)
        x = nn.AddBias(nn.Linear(x, self.w2), self.b2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])
                self.w0.update(gradients[0], -self.learning_rate)
                self.b0.update(gradients[1], -self.learning_rate)
                self.w1.update(gradients[2], -self.learning_rate)
                self.b1.update(gradients[3], -self.learning_rate)
                self.w2.update(gradients[4], -self.learning_rate)
                self.b2.update(gradients[5], -self.learning_rate)
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) <= 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 200
        self.learning_rate = 0.1
        # input layer
        self.w0 = nn.Parameter(784, 128)
        self.b0 = nn.Parameter(1, 128)
        # hidden layer
        self.w1 = nn.Parameter(128, 32)
        self.b1 = nn.Parameter(1, 32)
        # output layer
        self.w2 = nn.Parameter(32, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        x = nn.ReLU(x)
        x = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        x = nn.ReLU(x)
        x = nn.AddBias(nn.Linear(x, self.w2), self.b2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])
                self.w0.update(gradients[0], -self.learning_rate)
                self.b0.update(gradients[1], -self.learning_rate)
                self.w1.update(gradients[2], -self.learning_rate)
                self.b1.update(gradients[3], -self.learning_rate)
                self.w2.update(gradients[4], -self.learning_rate)
                self.b2.update(gradients[5], -self.learning_rate)
            if dataset.get_validation_accuracy() >= 0.975:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 50
        self.learning_rate = 0.05
        self.w_input = nn.Parameter(self.num_chars, 128)
        self.w_hidden = nn.Parameter(128, 128)
        self.w_output = nn.Parameter(128, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.Linear(xs[0], self.w_input)
        for x in xs[1:]:
            z = nn.Add(nn.Linear(x, self.w_input), nn.ReLU(nn.Linear(z, self.w_hidden)))
        z = nn.Linear(z, self.w_output)
        return z

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for xs, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.w_input, self.w_hidden, self.w_output])
                self.w_input.update(gradients[0], -self.learning_rate)
                self.w_hidden.update(gradients[1], -self.learning_rate)
                self.w_output.update(gradients[2], -self.learning_rate)
            if dataset.get_validation_accuracy() >= 0.85:
                break
