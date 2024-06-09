import streamlit as st
import numpy as np

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Backpropagation Algorithm
def train(X, y, epochs, lr):
    input_neurons = X.shape[1]
    hidden_neurons = 2
    output_neurons = 1

    # Weight initialization
    wh = np.random.uniform(size=(input_neurons, hidden_neurons))
    bh = np.random.uniform(size=(1, hidden_neurons))
    wout = np.random.uniform(size=(hidden_neurons, output_neurons))
    bout = np.random.uniform(size=(1, output_neurons))

    for _ in range(epochs):
        # Forward propagation
        hidden_input = np.dot(X, wh) + bh
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, wout) + bout
        final_output = sigmoid(final_input)

        # Backward propagation
        error = y - final_output
        d_output = error * sigmoid_derivative(final_output)

        error_hidden = d_output.dot(wout.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        # Update weights and biases
        wout += hidden_output.T.dot(d_output) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr
        wh += X.T.dot(d_hidden) * lr
        bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

    return wh, bh, wout, bout

# Prediction function
def predict(X, wh, bh, wout, bout):
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, wout) + bout
    final_output = sigmoid(final_input)
    return final_output

# Streamlit app
def main():
    st.title("Backpropagation Algorithm")

    st.write("### Training Data")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    st.write("Inputs (X):", X)
    st.write("Outputs (y):", y)

    epochs = st.slider("Epochs", min_value=100, max_value=10000, step=100, value=1000)
    lr = st.slider("Learning Rate", min_value=0.01, max_value=1.0, step=0.01, value=0.1)

    if st.button("Train"):
        wh, bh, wout, bout = train(X, y, epochs, lr)
        st.write("Training completed.")

        st.write("### Test the Model")
        input_data = [st.number_input(f"Input {i+1}", value=0.0) for i in range(X.shape[1])]
        input_data = np.array(input_data).reshape(1, -1)
        prediction = predict(input_data, wh, bh, wout, bout)
        st.write(f"Prediction for {input_data}: {prediction[0][0]}")

if __name__ == "__main__":
    main()