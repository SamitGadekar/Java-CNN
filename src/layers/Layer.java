package layers;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Layer.java
 * An abstract base class representing a layer in a neural network.
 * Defines essential methods that all specific layer types must implement.
 * Supports both forward and backward propagation.
 * This class also provides utility functions for converting between
 * matrix-based and vector-based representations.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public abstract class Layer implements Serializable {

    // References to the next and previous layers in the network
    protected Layer _nextLayer;
    protected Layer _previousLayer;

    public Layer get_nextLayer() {
        return _nextLayer;
    }
    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }
    public Layer get_previousLayer() {
        return _previousLayer;
    }
    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }

    // Appends the layer's data to a file (can be read in from a specialized constructor for each layer)
    public abstract void saveToFile(String filePath) throws IOException;

    /**
     * Computes and returns the output of the layer given an input in matrix or vector form.
     */
    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    /**
     * Performs backpropagation through this layer, given the gradient of the loss function with respect to its output.
     */
    public abstract void backPropagation(List<double[][]> dLdO);
    public abstract void backPropagation(double[] dLdO);

    /**
     * Returns the related number of output values this layer produces.
     * Length: number of matrices in the matrix list (0 for a vector output)
     * Rows: number of rows in each matrix in the matrix list (0 for a vector output)
     * Cols: number of columns in each matrix in the matrix list (0 for a vector output)
     * Elements: the total number of values in the output
     */
    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();
    public abstract int getOutputElements();

    /**
     * Converts a list of 2D matrices into a 1D vector.
     * Used for flattening convolutional layer outputs into a fully connected layer.
     *
     * @param input A list of 2D matrices representing feature maps.
     * @return A flattened 1D array containing all elements of the matrices.
     */
    public double[] matrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;
        double[] vector = new double[length * rows * cols];

        int i = 0;
        for (int l = 0; l < length; l++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[i++] = input.get(l)[r][c];
                }
            }
        }
        return vector;
    }

    /**
     * Converts a 1D vector into a list of 2D matrices.
     * Used for reshaping flattened data back into spatial representation.
     *
     * @param input A 1D array containing image data.
     * @param length The number of feature maps (depth).
     * @param rows The number of rows in each matrix.
     * @param cols The number of columns in each matrix.
     * @return A list of 2D matrices reconstructed from the input vector.
     */
    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> out = new ArrayList<>();
        int i = 0;
        for (int l = 0; l < length; l++) {
            double[][] matrix = new double[rows][cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    matrix[r][c] = input[i++];
                }
            }
            out.add(matrix);
        }
        return out;
    }

}
