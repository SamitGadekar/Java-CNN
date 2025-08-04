package network;

import layers.*;

import java.util.ArrayList;
import java.util.List;

/**
 * A builder class to construct a customizable Neural Network with different layers.
 * Provides methods to add convolutional, max-pooling, and fully connected layers.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class NetworkBuilder {

    private NeuralNetwork net;
    private final int _inputRows;
    private final int _inputCols;
    private final double _scaleFactor;
    private final int progressResolution;
    private final long SEED;
    List<Layer> _layers;
    private double _learningDecay;

    /**
     * Constructs a NetworkBuilder instance with the given parameters.
     *
     * @param _inputRows         Number of rows in the input data.
     * @param _inputCols         Number of columns in the input data.
     * @param scaleFactor        Factor to scale input values.
     * @param progressResolution Determines how often training progress is displayed.
     * @param SEED               Seed for random number generation.
     */
    public NetworkBuilder(int _inputRows, int _inputCols, double scaleFactor, int progressResolution, long SEED) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = scaleFactor;
        this._learningDecay = 1.0;
        this.progressResolution = progressResolution;
        this.SEED = SEED;
        _layers = new ArrayList<>();
    }

    public void setLearningDecay(double learningDecay) {
        this._learningDecay = learningDecay;
    }
    public NeuralNetwork getNetwork() {
        return net;
    }

    /**
     * Builds and returns a Neural Network instance.
     * Ensures the last layer is a FullyConnectedLayer before creating the network.
     *
     * @return The constructed NeuralNetwork or null if the last layer is invalid.
     */
    public NeuralNetwork build() {
        if ( !(_layers.get(_layers.size() - 1) instanceof FullyConnectedLayer) ) {
            System.out.println("Cannot make neural network: last layer is not FullyConnectedLayer!");
            return null;
        }
        net = new NeuralNetwork(_layers, _scaleFactor, progressResolution);
        net.setLearningDecay(_learningDecay);
        return net;
    }

    /**
     * Adds a Convolutional Layer to the network with the specified parameters.
     *
     * @param numFilters  Number of filters in the convolutional layer.
     * @param filterSize  Size of each filter (kernel).
     * @param stepSize    Stride of the convolution.
     * @param learningRate Learning rate for this layer.
     */
    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate) {
        if (_layers.isEmpty()) {
            _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size() - 1);
            _layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    /**
     * Adds a Max-Pooling Layer to the network with the specified parameters.
     *
     * @param windowSize Size of the pooling window.
     * @param stepSize   Stride for pooling operation.
     */
    public void addMaxPoolLayer(int windowSize, int stepSize) {
        if (_layers.isEmpty()) {
            _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        } else {
            Layer prev = _layers.get(_layers.size() - 1);
            _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    /**
     * Adds a Fully Connected Layer to the network with the specified parameters.
     *
     * @param outLength    Number of neurons in the fully connected layer.
     * @param learningRate Learning rate for this layer.
     */
    public void addFullyConnectedLayer(int outLength, double learningRate) {
        if (_layers.isEmpty()) {
            _layers.add(new FullyConnectedLayer(_inputRows * _inputCols, outLength, SEED, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size() - 1);
            _layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, SEED, learningRate));
        }
    }

    /**
     * Deletes all layers of the current network builder.
     */
    public void clearAllLayers(){
        _layers.clear();
        _layers = new ArrayList<>();
    }
}
