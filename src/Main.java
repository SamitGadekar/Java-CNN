import data.*;
import interactive.ModelBuilder;
import network.NetworkBuilder;
import network.NeuralNetwork;

import javax.swing.*;
import java.io.*;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import static java.util.Collections.shuffle;

public class Main {

    public static List<Image> imagesTrain;
    public static List<Image> imagesTest;

    // THESE VARIABLES ARE FOR EXPERIMENTING WITH CODING THE MODELS (THEY ARE NOT USED IN THE GUI COMPONENTS)
    // GO TO ModelBuilder.java FOR ACCESSING THOSE RELEVANT VARIABLES
    private static final long SEED = 123; // To initialize the randomized starting weights
    private static final double learnRate = 0.01; // The multiplier for how fast the network converges
    private static final int progressResolution = 100; // The resolution of the loading bar for training the network
    private static final double scaleFactor = 10; // Scales the input values down by this factor
    private static final double learningDecay = 0.97; // Multiplies the learning rate every epoch by this amount

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ModelBuilder().createAndShowGUI());
    }

    /**
     * Constructs a convolutional neural network using a predefined architecture.
     * The network consists of convolution, pooling, and fully connected layers.
     *
     * @return A fully initialized neural network to be trained.
     */
    public static NeuralNetwork makeNet() {
        // Initialize the builder
        NetworkBuilder builder = new NetworkBuilder(28, 28, scaleFactor, progressResolution, SEED);

        /*
        You may add as many layers as you want in any order you'd like.
        In most cases, the builder will figure out the connections,
        however, if a fully connected layer connects to a convolutional layer,
        the number of outputs of the fully connected layer must be a multiple
        of the (_inputRows x _inputCols) that the fully connected layer is expecting.

        .addConvolutionLayer(numFilters, filterSize, stride, learningRate)
        .addMaxPoolLayer(windowSize, stride)
        .addFullyConnectedLayer(outputLength, learningRate)

        Example:
        builder.addConvolutionLayer(8, 5, 1, 0.01);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.01);
        */
        builder.addConvolutionLayer(8, 5, 1, learnRate);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, learnRate);

        builder.setLearningDecay(learningDecay);

        // Build and return the network
        return builder.build();
    }

    /**
     * Converts a given rate value into a percentage string formatted as "_.__%".
     *
     * @param rate The decimal value of the rate (e.g., 0.85 for 85%).
     * @return A formatted string representation of the rate percentage.
     */
    public static String rtp(double rate) {
        return String.format("%.2f", rate * 100) + "%";
    }

    /**
     * Loads training and testing data from CSV files.
     * This method reads and stores MNIST dataset images for training and testing.
     */
    public static void loadData() {
        imagesTrain = new DataReader(28, 28).readData("data/mnist_train.csv");
        System.out.println("Training Data Loaded: " + imagesTrain.size());
        imagesTest = new DataReader(28, 28).readData("data/mnist_test.csv");
        System.out.println("Testing Data Loaded: " + imagesTest.size());
    }
    /**
     * Standardizes the training and testing image datasets.
     * Normalization ensures consistent input values for the neural network.
     */
    public static void standardizeData() {
        DataReader.standardizeNormal(imagesTrain);
        DataReader.standardizeNormal(imagesTest);
    }

    /**
     * Trains the given neural network for a specified number of epochs.
     *
     * @param net The neural network to train.
     * @param numEpochs The number of training iterations.
     * @return The trained neural network.
     */
    public static NeuralNetwork trainNetwork(NeuralNetwork net, int numEpochs) {
        // trains for those epochs and returns test rate

        boolean printProg = progressResolution > 0;

        if (printProg) {
            System.out.print("|");
            for (int i = 0; i < progressResolution; i++) System.out.print("-");
            System.out.println("|");
        }

        for (int i = 1; i <= numEpochs; i++) {
            shuffle(imagesTrain);
            net.train(imagesTrain);

            System.out.println(" : Epoch " + i + " : " + rtp(net.test(imagesTest)));
        }

        return net;
    }
    /**
     * Trains the neural network until the target accuracy is reached or failure count is exceeded.
     *
     * @param net The neural network to train.
     * @param targetRate The target accuracy rate.
     * @param maxFailureCount The maximum consecutive failures allowed before stopping.
     * @return The best-performing trained neural network.
     */
    public static NeuralNetwork trainNetwork(NeuralNetwork net, double targetRate, int maxFailureCount) {
        boolean printProg = progressResolution > 0;

        if (printProg) {
            System.out.print("|");
            for (int i = 0; i < progressResolution; i++) System.out.print("-");
            System.out.println("|");
        }

        ArrayList<Double> rateList = new ArrayList<>();
        rateList.add(net.test(imagesTest));

        int epochCount = 0;
        int bestEpochCount = 0;
        int failCount = 0;
        String modelFilePath = newFilePath();

        while (rateList.getLast() < targetRate) {
            shuffle(imagesTrain);
            net.train(imagesTrain);
            epochCount++;
            double newRate = net.test(imagesTest);

            if (newRate <= rateList.getLast()) {
                failCount++;
                System.out.println(" Dis " + failCount + " : Epoch " + epochCount + " : " + rtp(newRate));
            } else {
                rateList.add(newRate);
                bestEpochCount = epochCount;

                try {
                    net.saveToFile(modelFilePath);
                    System.out.println(" Saved : Epoch " + epochCount + " : " + rtp(newRate));
                } catch (IOException e) {
                    System.out.println(" ERROR : Epoch " + epochCount + " : " + rtp(newRate));
                    e.printStackTrace();
                }
            }

            if (failCount >= maxFailureCount) break;
        }
        System.out.println("MODEL EPOCH " + bestEpochCount + " CHOSEN");

        NeuralNetwork bestNet = null;
        try {
            bestNet = new NeuralNetwork(modelFilePath);
            if (printProg) System.out.println("Model loaded successfully (saved in file \"" + modelFilePath + "\")");
        } catch (IOException e) {
            System.out.println("ERROR: could not load model (saved in file \"" + modelFilePath + "\")");
            e.printStackTrace();
        }
        return bestNet;
    }
    /**
     * Trains the best previous neural network until the target accuracy is reached or failure count is exceeded.
     *
     * @param net The neural network to train.
     * @param targetRate The target accuracy rate.
     * @param maxFailureCount The maximum consecutive failures allowed before stopping.
     * @return The best-performing trained neural network.
     */
    public static NeuralNetwork trainOnBestNetwork(NeuralNetwork net, double targetRate, int maxFailureCount) {
        boolean printProg = progressResolution > 0;

        if (printProg) {
            System.out.print("|");
            for (int i = 0; i < progressResolution; i++) System.out.print("-");
            System.out.println("|");
        }

        ArrayList<Double> rateList = new ArrayList<>();
        rateList.add(net.test(imagesTest));

        int epochCount = 0;
        int bestEpochCount = 0;
        int failCount = 0;
        String modelFilePath = newFilePath();

        while (rateList.getLast() < targetRate) {
            shuffle(imagesTrain);
            net.train(imagesTrain);
            epochCount++;
            double newRate = net.test(imagesTest);

            if (newRate <= rateList.getLast()) {
                failCount++;
                System.out.println(" Dis " + failCount + " : Epoch " + epochCount + " : " + rtp(newRate));

                try {
                    net = new NeuralNetwork(modelFilePath);
                } catch (IOException e) {
                    System.out.println("ERROR: could not load best model yet (saved in file \"" + modelFilePath + "\")");
                    e.printStackTrace();
                }

            } else {
                rateList.add(newRate);
                bestEpochCount = epochCount;

                try {
                    net.saveToFile(modelFilePath);
                    System.out.println(" Saved : Epoch " + epochCount + " : " + rtp(newRate));
                } catch (IOException e) {
                    System.out.println(" ERROR : Epoch " + epochCount + " : " + rtp(newRate));
                    e.printStackTrace();
                }
            }

            if (failCount >= maxFailureCount) break;
        }
        System.out.println("MODEL EPOCH " + bestEpochCount + " CHOSEN");

        NeuralNetwork bestNet = null;
        try {
            bestNet = new NeuralNetwork(modelFilePath);
            if (printProg) System.out.println("Model loaded successfully (saved in file \"" + modelFilePath + "\")");
        } catch (IOException e) {
            System.out.println("ERROR: could not load model (saved in file \"" + modelFilePath + "\")");
            e.printStackTrace();
        }
        return bestNet;
    }


    /**
     * Generates a unique file path for saving neural network models.
     *
     * @return The unique file path string.
     */
    private static String newFilePath() {
        String basePath = "models/model_";
        int count = 0;

        String filePath;
        do {
            filePath = basePath + count + ".txt";
            count++;
        } while (new File(filePath).exists());

        return filePath;
    }

    /**
     * Deletes a file at the specified file path.
     *
     * @param filePath The file path to be deleted.
     */
    private static boolean deleteFile(String filePath) {
        File file = new File(filePath);
        if (file.exists()) return file.delete();
        return false;
    }

    /**
     * Gets all neural network models in the specified folder path.
     * If the folder does not exist or is empty, appropriate messages are displayed.
     *
     * @param directoryPath The path of the directory to list files from.
     * @return An arraylist of the neural networks from each file in the directory
     */
    public static List<NeuralNetwork> getModelsInDirectory(String directoryPath) {
        File folder = new File(directoryPath);
        if (!(folder.exists() && folder.isDirectory())) {
            System.out.println("Directory does not exist (\"" + directoryPath + "\").");
            return new ArrayList<>();
        }
        File[] files = folder.listFiles(); // Get all files and directories
        if (files == null) {
            System.out.println("Directory is empty (\"" + directoryPath + "\").");
            return new ArrayList<>();
        }

        if (directoryPath.charAt(directoryPath.length() - 1) != '/') {
            directoryPath += '/';
        }

        List<NeuralNetwork> netList = new ArrayList<>();
        Arrays.sort(files, (f1, f2) -> f1.getName().compareToIgnoreCase(f2.getName()));
        for (File file : files) {
            String modelFilePath = directoryPath + file.getName();
            try {
                NeuralNetwork net = new NeuralNetwork(modelFilePath);
                netList.add(net);
            } catch (IOException e) {
                System.out.println("ERROR: could not load model (saved in file \"" + modelFilePath + "\")");
                e.printStackTrace();
            }
        }
        return netList;
    }



}
