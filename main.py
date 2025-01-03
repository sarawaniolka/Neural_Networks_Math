import math
import random
import numpy as np  # For using argmax to find the index of the maximum value


# Core Neural Network Functions
def layer(X, W, b):
    """
    Computes the linear transformation for a single layer.
    Parameters:
        X (list): Input features.
        W (list of lists): Weights for the layer.
        b (list): Biases for the layer.
    Returns:
        list: Output values after the linear transformation.
    """
    outputs = []
    for i in range(len(W)):  # Iterate over the output neurons
        dot_product = sum(X[j] * W[i][j] for j in range(len(X)))  # Compute weighted sum
        outputs.append(dot_product + b[i])  # Add bias
    return outputs


def relu(Z):
    """
    ReLU activation function to introduce non-linearity.
    Parameters:
        Z (list): Linear outputs from a layer.
    Returns:
        list: Outputs after applying ReLU.
    """
    return [max(0, z) for z in Z]


def softmax(Z):
    """
    Softmax activation function to convert logits into probabilities.
    Parameters:
        Z (list): Outputs from the final layer.
    Returns:
        list: Probabilities for each class.
    """
    exp_values = [math.exp(z) for z in Z]
    total = sum(exp_values)
    return [ev / total for ev in exp_values]


def relu_derivative(Z):
    """
    Derivative of ReLU used for backpropagation.
    Parameters:
        Z (list): Linear outputs from a layer.
    Returns:
        list: Gradients for ReLU.
    """
    return [1 if z > 0 else 0 for z in Z]


def softmax_derivative(y_true, y_pred):
    """
    Derivative of softmax combined with loss.
    Parameters:
        y_true (list): True one-hot encoded labels.
        y_pred (list): Predicted probabilities.
    Returns:
        list: Gradients for the loss with respect to outputs.
    """
    return [y_pred[i] - y_true[i] for i in range(len(y_true))]


def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function.
    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    Returns:
        float: Computed MSE loss.
    """
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE loss function.
    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
    Returns:
        list: Gradients of the loss.
    """
    return [2 * (yp - yt) / len(y_true) for yp, yt in zip(y_pred, y_true)]


def network(X, layers):
    """
    Performs forward propagation through the neural network.
    Parameters:
        X (list): Input features.
        layers (list): List of tuples containing weights and biases for each layer.
    Returns:
        list: Final output of the network (predicted probabilities).
    """
    output = X
    for i, (W, b) in enumerate(layers):
        Z = layer(output, W, b)  # Linear transformation
        if i == len(layers) - 1:  # Output layer
            output = softmax(Z)  # Apply softmax for classification
        else:
            output = relu(Z)  # Apply ReLU for hidden layers
    return output


def backpropagation(X, y_true, layers):
    """
    Performs backpropagation to compute gradients.
    Parameters:
        X (list): Input features.
        y_true (list): True labels.
        layers (list): List of tuples containing weights and biases.
    Returns:
        tuple: Gradients for weights and biases.
    """
    layer_outputs = [X]  # Store activations of each layer
    pre_activation = []  # Store pre-activation (Z) values
    output = X
    for i, (W, b) in enumerate(layers):
        Z = layer(output, W, b)
        pre_activation.append(Z)
        if i == len(layers) - 1:
            output = softmax(Z)
        else:
            output = relu(Z)
        layer_outputs.append(output)

    dW_list = []
    db_list = []
    dA = softmax_derivative(
        y_true, layer_outputs[-1]
    )  # Compute gradient at output layer

    for i in reversed(range(len(layers))):
        W, b = layers[i]
        Z = pre_activation[i]
        A_prev = layer_outputs[i]

        if i == len(layers) - 1:  # Output layer
            dZ = dA  # No activation function derivative for softmax
        else:  # Hidden layers
            dZ = [dA[j] * relu_derivative(Z)[j] for j in range(len(Z))]

        dW = [[A_prev[k] * dZ[j] for k in range(len(A_prev))] for j in range(len(dZ))]
        db = dZ

        dW_list.insert(0, dW)
        db_list.insert(0, db)

        # Update dA for the next layer
        dA = [sum(W[j][k] * dZ[j] for j in range(len(W))) for k in range(len(W[0]))]
    return dW_list, db_list


def update_parameters(layers, dW_list, db_list, learning_rate):
    """
    Updates weights and biases using gradients.
    Parameters:
        layers (list): Current weights and biases.
        dW_list (list): Gradients for weights.
        db_list (list): Gradients for biases.
        learning_rate (float): Step size for parameter updates.
    Returns:
        list: Updated weights and biases.
    """
    updated_layers = []
    for i in range(len(layers)):
        W, b = layers[i]
        dW, db = dW_list[i], db_list[i]

        new_W = [
            [W[j][k] - learning_rate * dW[j][k] for k in range(len(W[j]))]
            for j in range(len(W))
        ]
        new_b = [b[j] - learning_rate * db[j] for j in range(len(b))]

        updated_layers.append((new_W, new_b))
    return updated_layers


def evaluate(X_test, y_test, layers):
    """
    Evaluates the model on the testing set.
    Parameters:
        X_test (list of lists): Testing inputs.
        y_test (list of lists): Testing targets.
        layers (list): Trained weights and biases.
    Returns:
        tuple: Average loss and accuracy on the testing set.
    """
    total_loss = 0
    correct_predictions = 0
    for X, y_true in zip(X_test, y_test):
        y_pred = network(X, layers)
        total_loss += mse(y_true, y_pred)

        # Convert predictions and labels to class indices
        true_label = np.argmax(y_true)
        predicted_label = np.argmax(y_pred)
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test) * 100
    return total_loss / len(X_test), accuracy


# Initialization Functions
def initialize_layers(layer_sizes):
    """
    Initializes weights and biases for each layer.
    Parameters:
        layer_sizes (list): Number of neurons in each layer.
    Returns:
        list: Initialized weights and biases.
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        W = [
            [random.uniform(-0.5, 0.5) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        b = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
        layers.append((W, b))
    return layers


# Training and Execution
def train_with_evaluation(
    X_train, y_train, X_test, y_test, layer_sizes, learning_rate, epochs
):
    """
    Trains the model and evaluates on test data.
    Parameters:
        X_train (list of lists): Training inputs.
        y_train (list of lists): Training targets.
        X_test (list of lists): Testing inputs.
        y_test (list of lists): Testing targets.
        layer_sizes (list): Network architecture (neurons per layer).
        learning_rate (float): Step size for gradient descent.
        epochs (int): Number of training iterations.
    Returns:
        tuple: Trained layers and visualization data.
    """
    layers = initialize_layers(layer_sizes)

    # For storing data for visualization
    training_loss_history = []
    testing_loss_history = []
    testing_accuracy_history = []
    gradients_history = []  # Store gradients for all epochs

    for epoch in range(epochs):
        total_training_loss = 0
        epoch_gradients = []  # Store gradients for all examples in the current epoch

        for X, y_true in zip(X_train, y_train):
            # Forward pass
            y_pred = network(X, layers)
            total_training_loss += mse(y_true, y_pred)

            # Backpropagation
            dW_list, db_list = backpropagation(X, y_true, layers)
            layers = update_parameters(layers, dW_list, db_list, learning_rate)

            # Save gradients for this training example
            epoch_gradients.append({"dW": dW_list, "db": db_list})

        # Save epoch-wise gradients
        gradients_history.append(epoch_gradients)

        # Calculate average losses and accuracy
        avg_training_loss = total_training_loss / len(X_train)
        testing_loss, testing_accuracy = evaluate(X_test, y_test, layers)

        # Save loss and accuracy for visualization
        training_loss_history.append(avg_training_loss)
        testing_loss_history.append(testing_loss)
        testing_accuracy_history.append(testing_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_training_loss:.6f}, "
            f"Testing Loss: {testing_loss:.6f}, Testing Accuracy: {testing_accuracy:.2f}%"
        )

    # Prepare visualization data
    visualization_data = {
        "layer_sizes": layer_sizes,
        "weights": [np.array(w).tolist() for w, _ in layers],
        "biases": [np.array(b).tolist() for _, b in layers],
        "training_loss_history": training_loss_history,
        "testing_loss_history": testing_loss_history,
        "testing_accuracy_history": testing_accuracy_history,
        "gradients_history": gradients_history,  # Gradients for all epochs
        "inputs": X_train[0].tolist(),  # Add the first training input as an example
        "labels": y_train[0].tolist(),
    }

    return layers, visualization_data


# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import json

    # Load and preprocess the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_onehot, test_size=0.2, random_state=42
    )

    # Define network architecture and training parameters
    layer_sizes = [4, 8, 6, 3]  # 4 input neurons, 2 hidden layers, 3 output neurons
    learning_rate = 0.01
    epochs = 100

    # Train the model and collect visualization data
    trained_layers, visualization_data = train_with_evaluation(
        X_train, y_train, X_test, y_test, layer_sizes, learning_rate, epochs
    )

    # Save visualization data to JSON
    with open("visualization_data.json", "w") as f:
        json.dump(visualization_data, f)
