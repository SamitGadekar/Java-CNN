package layers;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import static data.MatrixUtility.*;

/**
 * ConvolutionLayer.java
 * Represents the convolution layer of a neural network.
 * Applies convolutional filters to input data to extract spatial features.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */
public class ConvolutionLayer extends Layer {

    private final long SEED;

    private List<double[][]> _filters;
    private final int _filterSize;
    private final int _stepSize;

    private final int _inLength;
    private final int _inRows;
    private final int _inCols;
    private double _learningRate;

    private List<double[][]> _lastInput;

    /**
     * Constructs a convolutional layer with the specified parameters.
     * Initializes filters randomly.
     *
     * @param _filterSize  Size of each convolutional filter (assumed square)
     * @param _stepSize    Stride length for moving the filter across the input
     * @param _inLength    Number of input channels (depth of the input volume)
     * @param _inRows      Number of rows in input data
     * @param _inCols      Number of columns in input data
     * @param SEED         Random seed for weight initialization
     * @param numFilters   Number of filters to be applied in this layer
     * @param learningRate Learning rate for weight updates during backpropagation
     */
    public ConvolutionLayer(int _filterSize, int _stepSize, int _inLength, int _inRows, int _inCols, long SEED, int numFilters, double learningRate) {
        this._filterSize = _filterSize;
        this._stepSize = _stepSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this.SEED = SEED;
        this._learningRate = learningRate;

        generateRandomFilters(numFilters);
    }

    /**
     * Saves the ConvolutionLayer's values to a specified file.
     * The format will be:
     * ConvolutionLayer
     * filterSize stepSize inLength inRows inCols numFilters learningRate SEED
     * filters row-wise (each filter on a new line)...
     * ---END---
     *
     * @param filePath The path of the file to save the layer data.
     * @throws IOException If an I/O error occurs.
     */
    public void saveToFile(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write("ConvolutionLayer");
            writer.newLine();

            writer.write(
                    this._inLength + " " +
                            this._inRows + " " +
                            this._inCols + " " +
                            this._filters.size() + " " +
                            this._filterSize + " " +
                            this._stepSize + " " +
                            this._learningRate + " " +
                            this.SEED
            );
            writer.newLine();

            for (double[][] filter : this._filters) {
                for (double[] row : filter) {
                    for (double weight : row) {
                        writer.write(weight + " ");
                    }
                }
                writer.newLine();
            }

            writer.write("---END---");
            writer.newLine();
            writer.flush();
        }
    }

    /**
     * Constructs a ConvolutionLayer object by loading parameters from a file.
     * Assumes the file follows the format specified in saveToFile().
     * Reads from a specific line number and updates the reference with the last read line.
     *
     * @param filePath The path of the file to load the layer from.
     * @param lineNumberRef A mutable reference to track the starting and ending line numbers.
     * @throws IOException If an I/O error occurs.
     */
    public ConvolutionLayer(String filePath, int[] lineNumberRef) throws IOException {
        try (Scanner scanner = new Scanner(new File(filePath))) {
            int currentLine = 0;

            while (scanner.hasNextLine() && (currentLine < lineNumberRef[0])) {
                scanner.nextLine();
                currentLine++;
            }

            if (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (!line.equals("ConvolutionLayer"))
                    throw new IOException("Unexpected layer type in file: \"" + line + "\"");
            }
            currentLine++;

            this._inLength = scanner.nextInt();
            this._inRows = scanner.nextInt();
            this._inCols = scanner.nextInt();
            int numFilters = scanner.nextInt();
            this._filterSize = scanner.nextInt();
            this._stepSize = scanner.nextInt();
            this._learningRate = scanner.nextDouble();
            this.SEED = scanner.nextLong();
            currentLine++;

            this._filters = new ArrayList<>();
            for (int f = 0; f < numFilters; f++) {
                double[][] filter = new double[this._filterSize][this._filterSize];
                for (int row = 0; row < this._filterSize; row++) {
                    for (int col = 0; col < this._filterSize; col++) {
                        filter[row][col] = scanner.nextDouble();
                    }
                }
                this._filters.add(filter);
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
     * Initializes the filters with random values using a Gaussian distribution.
     *
     * @param numFilters The number of filters to create
     */
    private void generateRandomFilters(int numFilters) {
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for (int n = 0; n < numFilters; n++) {

            double[][] newFilter = new double[_filterSize][_filterSize];
            for (int i = 0; i < _filterSize; i++) {
                for (int j = 0; j < _filterSize; j++) {
                    newFilter[i][j] = random.nextGaussian();
                    // newFilter[i][j] = random.nextGaussian() * Math.sqrt(2.0 / (_filterSize * _filterSize));
                }
            }
            filters.add(newFilter);
        }
        _filters = filters;
    }

    /**
     * Performs the forward pass by applying convolution with each filter and applying ReLU activation.
     *
     * @param list List of input matrices (one for each input channel)
     * @return List of output feature maps after applying filters and activation
     */
    public List<double[][]> convolutionForwardPass(List<double[][]> list) {
        _lastInput = list;

        List<double[][]> output = new ArrayList<>();
        for (double[][] eachInputMatrix : list) {
            for (double[][] filter : _filters) {
                double[][] convolved = convolve(eachInputMatrix, filter, _stepSize);
                output.add(matrixReLU(convolved));            }
        }
        return output;
    }

    /**
     * Applies leaky ReLU activation element-wise to the given matrix.
     *
     * @param input Input matrix
     * @return Output matrix after applying ReLU activation
     */
    private double[][] matrixReLU(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][j] = (input[i][j] > 0) ? input[i][j] : 0;
            }
        }
        return output;
    }

    /**
     * Applies the convolution operation between an input matrix and a filter.
     *
     * @param input    The input matrix
     * @param filter   The filter matrix
     * @param stepSize Stride length for the convolution
     * @return The convolved output matrix
     */
    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];
        int outRow = 0;
        for (int i = 0; i <= inRows - fRows; i += stepSize) {
            int outCol = 0;
            for (int j = 0; j <= inCols - fCols; j += stepSize) {
                // apply filter to this position
                double sum = 0;
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        sum += filter[x][y] * input[inputRowIndex][inputColIndex];
                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }
        return output;
    }

    /**
     * Expands a matrix by inserting zeroes between values based on step size.
     * Used in backpropagation to match the size of gradients to original input size.
     *
     * @param input The matrix to be expanded
     * @return The expanded matrix with spacing
     */
    public double[][] spaceArray(double[][] input) {
        if (_stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) * _stepSize + 1;
        int outCols = (input[0].length - 1) * _stepSize + 1;

        double[][] output = new double[outRows][outCols];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i * _stepSize][j * _stepSize] = input[i][j];
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = convolutionForwardPass(input);
        return _nextLayer.getOutput(output);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, _inLength, _inRows, _inCols);
        backPropagation(matrixInput);
    }

    /**
     * Performs backpropagation to update filters and propagate errors to the previous layer.
     *
     * @param dLdO The gradient of the loss with respect to the output of this layer.
     */
    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();

        for (int f = 0; f < _filters.size(); f++) {
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for (int i = 0; i < _lastInput.size(); i++) {

            double[][] errorForInput = new double[_inRows][_inCols];

            for (int f = 0; f < _filters.size(); f++) {

                double[][] currFilter = _filters.get(f);
                double[][] error = dLdO.get(i * _filters.size() + f);

                double[][] spacedError = spaceArray(error);

                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, _learningRate * -1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                // for backpropagation
                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = add(errorForInput, fullConvolve(currFilter, flippedError));
            }
            dLdOPreviousLayer.add(errorForInput);
        }

        for (int f = 0; f < _filters.size(); f++) {
            double[][] modified = add(_filters.get(f), filtersDelta.get(f));
            _filters.set(f, modified);
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dLdOPreviousLayer);
        }
    }

    /**
     * Flips a matrix horizontally (mirroring along the vertical axis).
     *
     * @param array The input matrix.
     * @return The horizontally flipped matrix.
     */
    public double[][] flipArrayHorizontal(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[rows - i - 1][j] = array[i][j];
            }
        }
        return output;
    }

    /**
     * Flips a matrix vertically (mirroring along the horizontal axis).
     *
     * @param array The input matrix.
     * @return The vertically flipped matrix.
     */
    public double[][] flipArrayVertical(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[][] output = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][cols - j - 1] = array[i][j];
            }
        }
        return output;
    }

    /**
     * Computes a full convolution between an input matrix and a filter.
     * Unlike standard convolution, full convolution expands the output size.
     *
     * @param input  The input matrix.
     * @param filter The filter matrix.
     * @return The result of the full convolution operation.
     */
    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outRows = input.length + filter.length + 1;
        int outCols = input[0].length + filter[0].length + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        for (int i = -fRows + 1; i < inRows; i ++) {

            int outCol = 0;
            for (int j = -fCols + 1; j < inCols; j ++) {

                double sum = 0;

                // apply filter to this position
                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols) {
                            sum += filter[x][y] * input[inputRowIndex][inputColIndex];
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }

    @Override
    public int getOutputLength() {
        return _filters.size() * _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _filterSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength() * getOutputRows() * getOutputCols();
    }

}
