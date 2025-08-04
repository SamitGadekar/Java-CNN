package layers;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Represents a Max Pooling layer in a neural network.
 * This layer reduces the spatial dimensions of the input using max pooling,
 * which helps in downsampling and extracting the most important features.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class MaxPoolLayer extends Layer {

    private int _stepSize;
    private int _windowSize;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxCol;

    /**
     * Constructs a max pool layer with the specified parameters.
     *
     * @param _stepSize   The stride for pooling.
     * @param _windowSize The size of the pooling window.
     * @param _inLength   The number of input feature maps.
     * @param _inRows     The height of the input.
     * @param _inCols     The width of the input.
     */
    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }

    /**
     * Saves the MaxPoolLayer's values to a specified file.
     * The format will be:
     * MaxPoolLayer
     * stepSize windowSize inLength inRows inCols
     * ---END---
     *
     * @param filePath The path of the file to save the layer data.
     * @throws IOException If an I/O error occurs.
     */
    public void saveToFile(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write("MaxPoolLayer");
            writer.newLine();

            writer.write(
                    this._inLength + " " +
                            this._inRows + " " +
                            this._inCols + " " +
                            this._windowSize + " " +
                            this._stepSize + " "
            );
            writer.newLine();

            writer.write("---END---");
            writer.newLine();
            writer.flush();
        }
    }

    /**
     * Constructs a MaxPoolLayer object by loading parameters from a file.
     * Assumes the file follows the format specified in saveToFile().
     * Reads from a specific line number and updates the reference with the last read line.
     *
     * @param filePath The path of the file to load the layer from.
     * @param lineNumberRef A mutable reference to track the starting and ending line numbers.
     * @throws IOException If an I/O error occurs.
     */
    public MaxPoolLayer(String filePath, int[] lineNumberRef) throws IOException {
        try (Scanner scanner = new Scanner(new File(filePath))) {
            int currentLine = 0;

            while (scanner.hasNextLine() && (currentLine < lineNumberRef[0])) {
                scanner.nextLine();
                currentLine++;
            }

            if (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (!line.equals("MaxPoolLayer"))
                    throw new IOException("Unexpected layer type in file: \"" + line + "\"");
            }
            currentLine++;

            this._inLength = scanner.nextInt();
            this._inRows = scanner.nextInt();
            this._inCols = scanner.nextInt();
            this._windowSize = scanner.nextInt();
            this._stepSize = scanner.nextInt();
            scanner.nextLine();
            currentLine++;

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

    /**
     * Performs a forward pass of max pooling on a batch of input feature maps.
     *
     * @param input List of 2D matrices representing input feature maps.
     * @return List of 2D matrices after max pooling.
     */
    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for (double[][] matrix : input) {
            output.add(pool(matrix));
        }

        return output;
    }

    /**
     * Applies max pooling to a single 2D input matrix.
     *
     * @param input The input matrix.
     * @return The pooled output matrix.
     */
    public double[][] pool(double[][] input) {
        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for (int r = 0; r < getOutputRows(); r += _stepSize) {
            for (int c = 0; c < getOutputCols(); c += _stepSize) {

                double max = 0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for (int x = 0; x < _windowSize; x++) {
                    for (int y = 0; y < _windowSize; y++) {
                        if (max < input[r + x][c + y]) {
                            max = input[r + x][c + y];
                            maxRows[r][c] = r + x;
                            maxCols[r][c] = c + y;
                        }
                    }
                }
                output[r][c] = max;
            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList);
    }

    /**
     * Performs backpropagation through the max pooling layer.
     *
     * @param dLdO The gradient of the loss with respect to the output.
     */
    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> dXdL = new ArrayList<>();

        int l = 0;
        for (double[][] array : dLdO) {
            double[][] error = new double[_inRows][_inCols];

            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if (max_i != -1 && max_j != -1) {
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }

            dXdL.add(error);
            l++;
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength * getOutputRows() * getOutputCols();
    }
}
