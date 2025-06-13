CNN for Fashion MNIST Classification
Overview
This project implements Convolutional Neural Networks (CNN) to classify images from the Fashion MNIST dataset using PyTorch. The Fashion MNIST dataset consists of 28x28 grayscale images of 10 different clothing categories.

The goal is to train a deep learning model to accurately predict the class of the clothing item shown in each image.

Project Structure
CNN.py — Contains the CNN model architectures:

CNN — Basic CNN without Batch Normalization

CNN_batch — CNN with Batch Normalization layers for improved training stability.

train.py — Training and validation loops with accuracy and loss tracking.

data_loader.py — Dataset loading and transformation scripts using torchvision datasets.

utils.py — Utility functions for plotting decision regions and visualizing results.

How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install dependencies (preferably in a virtual environment):

bash
Copy
Edit
pip install torch torchvision matplotlib
Run the training script:

bash
Copy
Edit
python train.py
Training logs will show loss and accuracy per epoch. The final model can be evaluated on the test set.

Model Details
CNN Architecture:

Conv Layer 1: 16 filters, 5x5 kernel, padding=2, ReLU activation

MaxPooling Layer 1: 2x2

Conv Layer 2: 32 filters, 5x5 kernel, padding=2, ReLU activation

MaxPooling Layer 2: 2x2

Fully connected layer: outputs 10 classes.

CNN with BatchNorm: Same architecture but includes Batch Normalization after each convolutional and fully connected layer to improve convergence.

Results
Training loss and accuracy improve steadily over epochs.

Achieved accuracy on validation set after 5 epochs: add your results here

Future Work
Experiment with deeper networks.

Add dropout layers to prevent overfitting.

Implement learning rate schedulers.

Use data augmentation to improve generalization.

References
Fashion MNIST Dataset

PyTorch Documentation

Batch Normalization Paper

License
This project is licensed under the MIT License — see the LICENSE file for details.

