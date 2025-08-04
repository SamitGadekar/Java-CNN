package layers;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * FullyConnectedLayer.java
 * Represents a fully connected (dense) layer in a neural network.
 * This layer performs a weighted sum of inputs, adds biases, and applies an activation function (ReLU).
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class FullyConnectedLayer extends Layer {

    private long SEED;
    private final double leak = 0.01; // Leaky ReLU value

    private double[] biases;
    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private double _learningRate;

    private double[] lastZ; // Stores pre-activation values (Z) for backpropagation
    private double[] lastX; // Stores last input values (X) for backpropagation

    private int _rows = 0; // Always 0
    private int _cols = 0; // Always 0

    /**
     * Constructs a fully connected layer with the specified parameters.
     * Initializes weights and biases randomly.
     *
     * @param _inLength     Number of input neurons
     * @param _outLength    Number of output neurons
     * @param SEED          Random seed for weight initialization
     * @param learningRate  Learning rate for gradient updates
     */
    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = learningRate;

        this._weights = new double[_inLength][_outLength];
        setRandomWeights();

        this.biases = new double[_outLength];
        setRandomBiases();
    }

    /**
     * Saves the FullyConnectedLayer's values to a specified file.
     * The format will be:
     * FullyConnectedLayer
     * inLength outLength learningRate SEED
     * biases...
     * weights...
     * ---END---
     *
     * @param filePath The path of the file to save the layer data.
     * @throws IOException If an I/O error occurs.
     */
    public void saveToFile(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write("FullyConnectedLayer");
            writer.newLine();

            writer.write(
                    this._inLength + " " +
                            this._outLength + " " +
                            this._learningRate + " " +
                            this.SEED
            );
            writer.newLine();

            for (double bias : this.biases) {
                writer.write(bias + " ");
            }
            writer.newLine();

            for (double[] row : this._weights) {
                for (double weight : row) {
                    writer.write(weight + " ");
                }
                writer.newLine();
            }

            writer.write("---END---");
            writer.newLine();
            writer.flush();
        }
    }

    /**
     * Constructs a FullyConnectedLayer object by loading parameters from a file.
     * Assumes the file follows the format specified in saveToFile().
     * Reads from a specific line number and updates the reference with the last read line.
     *
     * @param filePath The path of the file to load the layer from.
     * @param lineNumberRef A mutable reference to track the starting and ending line numbers.
     * @throws IOException If an I/O error occurs.
     */
    public FullyConnectedLayer(String filePath, int[] lineNumberRef) throws IOException {
        try (Scanner scanner = new Scanner(new File(filePath))) {
            int currentLine = 0;

            while (scanner.hasNextLine() && (currentLine < lineNumberRef[0])) {
                scanner.nextLine();
                currentLine++;
            }

            if (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (!line.equals("FullyConnectedLayer"))
                    throw new IOException("Unexpected layer type in file: \"" + line + "\"");
            }
            currentLine++;

            this._inLength = scanner.nextInt();
            this._outLength = scanner.nextInt();
            this._learningRate = scanner.nextDouble();
            this.SEED = scanner.nextLong();
            currentLine++;

            this.biases = new double[this._outLength];
            for (int i = 0; i < this._outLength; i++) {
                this.biases[i] = scanner.nextDouble();
            }
            scanner.nextLine();
            currentLine++;

            this._weights = new double[this._inLength][this._outLength];
            for (int i = 0; i < this._inLength; i++) {
                for (int j = 0; j < this._outLength; j++) {
                    this._weights[i][j] = scanner.nextDouble();
                }
                scanner.nextLine();
                currentLine++;
            }

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                currentLine++;
                if (line.equals("---END---")) {
                    break;
                }
            }

            lineNumberRef[0] = currentLine;
        }
    }

    public double getLearningRate() {
        return this._learningRate;
    }
    public void setLearningRate(double learningRate) {
        this._learningRate = learningRate;
    }

    /**
     * Performs a forward pass through the fully connected layer.
     * Computes Z = WX + B and applies the ReLU activation function.
     *
     * @param input Input vector of length _inLength
     * @return Output vector of length _outLength after activation
     */
    public double[] fullyConnectedForwardPass(double[] input) {
        lastX = input; // Store input for backpropagation
        double[] z = new double[_outLength]; // Pre-activation function output values
        double[] out = new double[_outLength]; // Post-activation function output values

        // Compute weighted sum Z = WX
        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                double value = input[i];
                value *= _weights[i][j];
                z[j] += input[i] * _weights[i][j];

            }
        }
        // Add bias and apply the ReLU activation
        for (int j = 0; j < _outLength; j++) {
            z[j] += biases[j];
            out[j] = reLu(z[j]);
        }

        lastZ = z; // Store pre-activation values for backpropagation
        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if (_nextLayer != null) {
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    /**
     * Performs backpropagation through this fully connected layer.
     * Updates weights and biases using the calculated gradients.
     *
     * @param dLdO Gradient of the loss with respect to the output (from next layer)
     */
    @Override
    public void backPropagation(double[] dLdO) {

        double[] dLdX = new double[_inLength];

        // w.r.t. = with respect to
        double dOdz; // Derivative ReLU of output w.r.t. pre-activation value
        double dzdw; // Derivative of pre-activation value w.r.t. weights
        double dLdw; // Derivative of loss w.r.t. weights
        double dzdx; // Derivative of pre-activation value w.r.t. inputs

        // Compute weight and bias updates
        for (int k = 0; k < _inLength; k++) {

            double dLdX_sum = 0;
            for (int j = 0; j < _outLength; j++) {

                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                dLdw = dLdO[j] * dOdz * dzdw;
                _weights[k][j] -= dLdw * _learningRate;

                dLdX_sum += dLdO[j] * dOdz * dzdx;
            }
            dLdX[k] = dLdX_sum;
        }
        for (int j = 0; j < _outLength; j++) {
            dOdz = derivativeReLu(lastZ[j]);
            double dLdb = dLdO[j] * dOdz;
            biases[j] -= dLdb * _learningRate;
        }

        // Propagate gradient to previous layer if it exists
        if (_previousLayer != null) {
            _previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        if (_rows == 0 || _cols == 0) {
            return 0; // Technically should always return 0.
        }
        return getOutputElements() / _rows / _cols;
    }

    @Override
    public int getOutputRows() {
        return _rows; // Should always return 0.
    }

    @Override
    public int getOutputCols() {
        return _cols; // Should always return 0.
    }

    @Override
    public int getOutputElements() {
        return _outLength; // Number of output neurons
    }

    public void setRandomWeights() {
        Random random = new Random(SEED);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = random.nextGaussian() * Math.sqrt(2.0 / _inLength); // Scaled for the standardized inputs to function
            }
        }
    }

    public void setRandomBiases() {
        Random random = new Random(SEED);
        for (int j = 0; j < _outLength; j++) {
            biases[j] = random.nextGaussian() * 0.01;  // Small random values for initialization
        }
    }

    /**
     * Applies the ReLU activation function.
     * If the input is negative, scales it by a small leak factor.
     *
     * @param input Pre-activation value
     * @return Activated value
     */
    public double reLu(double input) {
        return input > 0 ? input : input * leak;
    }
    /**
     * Computes the derivative of the ReLU activation function.
     * Helps during backpropagation.
     *
     * @param input Pre-activation value
     * @return Derivative of ReLU
     */
    public double derivativeReLu(double input) {
        return input > 0 ? 1 : leak;
    }

}
