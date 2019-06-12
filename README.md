# ML-neural-network
An implementation of a multi-class neural network.

In this exercise we was training our neural network on a dataset called “Fashion-MNIST”.

This dataset contains 10 different categories of clothing.

Our task was to train a classifier that classifies this data.


# Data
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.

Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel.

This pixel-value is an integer between 0 and 255.

# Labels
The possible labels are:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

# Mission
Our goal was to train a multi-class neural network for the Fashion-MNIST dataset.

Our network have one hidden layer with the ReLU activation function.

# Loss
Our model minimize the Negative Log Likelihood (NLL) loss function.

# Files
The program excepts to receive the data in the form of 3 files: 

(i) train x - will contain the training set examples;

(ii) train y - will contain the corresponding training set labels;

(iii) test x - will contain the test set examples.

# Model
We have train and validate our model using files (i)+(ii).

Finally, we output the model’s predictions on the examples in test x to a file named test y,

using the same format as in train y.
