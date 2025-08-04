package network;

import data.*;
import layers.*;
import static data.MatrixUtility.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Represents a simple feedforward neural network that can be trained and tested on image data.
 * The network consists of multiple layers and uses backpropagation for learning.
 * This implementation supports convolutional, fully connected, and pooling layers.
 * The network operates with a specified scale factor for input normalization and
 * provides progress updates during training.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class NeuralNetwork implements Serializable {

    private String networkFilePath;
    private String networkName = "myModel";
    List<Layer> _layers;
    double scaleFactor;
    int progressResolution;
    double learningDecay;

    /**
     * Constructs a neural network with a given set of layers.
     *
     * @param _layers            The list of layers forming the network.
     * @param scaleFactor        Factor used to scale input data.
     * @param progressResolution Determines how frequently progress is displayed.
     */
    public NeuralNetwork(List<Layer> _layers, double scaleFactor, int progressResolution) {
        this._layers = _layers;
        this.scaleFactor = scaleFactor;
        this.progressResolution = progressResolution;
        this.learningDecay = 1.0;
        linkLayers();
    }

    /**
     * Saves the neural network's metadata and all layers to a file.
     * The format will be:
     * NeuralNetwork
     * scaleFactor progressResolution learningDecay numLayers
     * (Layer data follows...)
     * ---END---
     *
     * @param filePath The path of the file to save the network.
     * @throws IOException If an I/O error occurs.
     */
    public void saveToFile(String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("NeuralNetwork");
            writer.newLine();

            writer.write(filePath);
            writer.newLine();

            writer.write(networkName);
            writer.newLine();

            writer.write(this.scaleFactor + " " + this.progressResolution + " " +
                    this.learningDecay + " " + _layers.size());
            writer.newLine();
            writer.flush();
        }

        for (Layer layer : _layers) {
            layer.saveToFile(filePath);
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write("NN---END---NN");
            writer.newLine();
            writer.flush();
        }
    }

    /**
     * Constructs a NeuralNetwork by loading its parameters and layers from a file.
     * Assumes the file follows the format specified in saveToFile().
     *
     * @param filePath The path of the file to load the network from.
     * @throws IOException If an I/O error occurs.
     */
    public NeuralNetwork(String filePath) throws IOException {
        _layers = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File(filePath))) {
            int[] lineNumberRef = {0};

            if (scanner.hasNextLine() && !scanner.nextLine().equals("NeuralNetwork")) {
                throw new IOException("Unexpected file format: Missing NeuralNetwork header.");
            }
            lineNumberRef[0]++;

            this.networkFilePath = scanner.nextLine();
            lineNumberRef[0]++;

            this.networkName = scanner.nextLine();
            lineNumberRef[0]++;

            this.scaleFactor = scanner.nextDouble();
            this.progressResolution = scanner.nextInt();
            this.learningDecay = scanner.nextDouble();
            int numLayers = scanner.nextInt();
            scanner.nextLine();

            lineNumberRef[0]++;

            int loadedLayers = 0;
            while (scanner.hasNextLine() && loadedLayers < numLayers) {
                String line = scanner.nextLine();

                if (line.equals("NN---END---NN")) {
                    break;
                }

                int prevLine = lineNumberRef[0];
                switch (line) {
                    case "FullyConnectedLayer":
                        _layers.add(new FullyConnectedLayer(filePath, lineNumberRef));
                        break;
                    case "ConvolutionLayer":
                        _layers.add(new ConvolutionLayer(filePath, lineNumberRef));
                        break;
                    case "MaxPoolLayer":
                        _layers.add(new MaxPoolLayer(filePath, lineNumberRef));
                        break;
                    default:
                        throw new IOException("Unknown layer type: " + line);
                }
                loadedLayers++;
                for (int i = prevLine; i < lineNumberRef[0] - 1; i++) {
                    scanner.nextLine();
                }
            }

            if (loadedLayers != numLayers) {
                throw new IOException("Mismatch in expected (" + numLayers + ") and loaded (" + loadedLayers + ") layer count.");
            }

            linkLayers();
        }
    }

    public String getNetworkFilePath() {
        return networkFilePath;
    }
    public void setNetworkName(String name) {
        this.networkName = name;
    }
    public String getNetworkName() {
        return this.networkName;
    }
    public void setLearningDecay(double learningDecay) {
        this.learningDecay = learningDecay;
    }

    /**
     * Links layers together by setting their previous and next connections.
     * This ensures correct forward propagation and backpropagation flow.
     */
    private void linkLayers() {
        if (_layers.size() <= 1) {
            return;
        }

        for (int i = 0; i < _layers.size(); i++) {
            if (i == 0) {
                _layers.get(i).set_nextLayer(_layers.get(i + 1));
            } else if (i == _layers.size() - 1) {
                _layers.get(i).set_previousLayer(_layers.get(i - 1));
            } else {
                _layers.get(i).set_previousLayer(_layers.get(i - 1));
                _layers.get(i).set_nextLayer(_layers.get(i + 1));
            }
        }
    }

    /**
     * Computes the error (difference between expected and actual output).
     *
     * @param networkOutput The output values produced by the neural network.
     * @param correctAnswer The correct classification label.
     * @return The error vector used for backpropagation.
     */
    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
    }

    /**
     * Evaluates the network's accuracy on a given dataset.
     *
     * @param images A list of images to test.
     * @return The accuracy as a percentage (0 to 1).
     */
    public double test(List<Image> images) {
        int correct = 0;
        for (Image img : images) {
            int guess = guess(img);

            if (guess == img.getLabel()) {
                correct++;
            }
        }

        return ((double) correct / (double) images.size());
    }

    /**
     * Evaluates the network's accuracy on a given dataset per number.
     *
     * @param images A list of images to test.
     * @return An array of accuracies for each number (0 to 9).
     */
    public double[] testPerType(List<Image> images) {
        int[] total = new int[10];
        int[] correct = new int[10];
        double[] accuracies = new double[10];

        for (int i = 0; i < 10; i++) {
            total[i] = 0;
            correct[i] = 0;
            accuracies[i] = 0.0;
        }

        for (Image img : images) {
            if (img.getLabel() < 0 || img.getLabel() > 9) continue;

            int guessNumber = guess(img);
            int correctNumber = img.getLabel();

            total[correctNumber]++;
            if (guessNumber == correctNumber) {
                correct[correctNumber]++;
            }
        }

        for (int i = 0; i < 10; i++) {
            accuracies[i] = (double) correct[i] / total[i];
        }
        return accuracies;
    }

    /**
     * Trains the neural network using the provided dataset.
     * Displays progress if `progressResolution` is set.
     *
     * @param images A list of training images.
     */
    public void train(List<Image> images) {

        if (progressResolution > 0) { System.out.print("|"); }

        int count = 0;
        for (Image img : images) {
            if ((progressResolution > 0) && (++count % (images.size()/progressResolution) == 0)) { System.out.print(">"); }

            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(img.getData(), (1.0/scaleFactor)));
            double[] out = _layers.get(0).getOutput(inList);
            double[] dLdO = getErrors(out, img.getLabel());
            _layers.get(_layers.size() - 1).backPropagation(dLdO);
        }
        if (progressResolution > 0) { System.out.print("|"); }

        for (Layer layer : _layers) {
            if (layer instanceof FullyConnectedLayer currLayer) {
                currLayer.setLearningRate(currLayer.getLearningRate() * learningDecay);
            } else if (layer instanceof ConvolutionLayer currLayer) {
                currLayer.setLearningRate(currLayer.getLearningRate() * learningDecay);
            }
        }
    }

    /**
     * Predicts the class label for a given image.
     *
     * @param image The input image.
     * @return The predicted class label.
     */
    public int guess(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1.0/scaleFactor)));

        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    /**
     * Returns the network's prediction values of each class bin.
     *
     * @param image The input image.
     * @return An array with the network's guess for each class.
     */
    public double[] getOutput(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), (1.0/scaleFactor)));

        double[] out = _layers.get(0).getOutput(inList);
        // return out;
        return softmax(out);
    }

    /**
     * Finds the index of the maximum value in an array.
     * This corresponds to the most confident class prediction.
     *
     * @param in The array of output values.
     * @return The index of the maximum value.
     */
    private int getMaxIndex(double[] in) {
        double max = 0;
        int index = 0;
        for (int i = 0; i < in.length; i++) {
            if (in[i] > max) {
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Computes the softmax function for a given array of values.
     * The softmax function converts raw scores into probabilities
     * by exponentiating each value and normalizing by the sum of all exponentiated values.
     *
     * @param input The array of input values.
     * @return A new array representing the softmax probabilities.
     */
    public static double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]);
            sum += output[i];
        }
        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }
        return output;
    }

}
