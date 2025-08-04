package interactive;

import data.Image;
import data.DataReader;
import network.NeuralNetwork;
import network.NetworkBuilder;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static java.util.Collections.shuffle;

public class ModelBuilder {
    private static final int WIDTH = 1300;
    private static final int HEIGHT = 500;

    private static final int GRID_ROWS = 28, GRID_COLS = 28;
    private static final double LEARNING_RATE = 0.01;
    private static final double LEARNING_DECAY = 0.97;
    private static final long SEED = 123;

    private List<Image> imagesTrain, imagesTest;
    private NetworkBuilder builder = new NetworkBuilder(GRID_ROWS, GRID_COLS, 10, 100, SEED);
    private NeuralNetwork currentModel = null;
    private JTextArea logArea;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ModelBuilder().createAndShowGUI());
    }

    public void createAndShowGUI() {
        JFrame frame = new JFrame("Neural Network Model Trainer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(WIDTH, HEIGHT);
        frame.setLayout(new BorderLayout());

        // Text log area
        logArea = new JTextArea();
        logArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        logArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(logArea);
        frame.add(scrollPane, BorderLayout.CENTER);

        // Control panel with all command buttons
        JPanel controlPanel = new JPanel(new GridLayout(0, 2, 10, 10));

        // Load data
        JButton loadDataButton = new JButton("Load & Standardize Data");
        controlPanel.add(loadDataButton);

        JPanel loadInputs = new JPanel(new GridLayout(2, 2, 0, -5));
        loadInputs.add(new JLabel("train filepath"));
        loadInputs.add(new JLabel("test filepath"));
        JTextField inputLoadTrain = new JTextField("data/mnist_train.csv", 5);
        loadInputs.add(inputLoadTrain);
        JTextField inputLoadTest = new JTextField("data/mnist_test.csv", 5);
        loadInputs.add(inputLoadTest);
        controlPanel.add(loadInputs);

        loadDataButton.addActionListener(e -> loadData(e, "data/mnist_train.csv", "data/mnist_test.csv"));


        // Add convolutional layer (filters, window size, stride)
        JButton addConvLayerButton = new JButton("Add Convolutional Layer");
        controlPanel.add(addConvLayerButton);

        JPanel convInputs = new JPanel(new GridLayout(2, 3, 10, -5));
        convInputs.add(new JLabel("filters"));
        convInputs.add(new JLabel("window size"));
        convInputs.add(new JLabel("stride"));
        JTextField inputConvFilters = new JTextField("8", 2);
        convInputs.add(inputConvFilters);
        JTextField inputConvWindowSize = new JTextField("5", 2);
        convInputs.add(inputConvWindowSize);
        JTextField inputConvStride = new JTextField("1", 2);
        convInputs.add(inputConvStride);
        controlPanel.add(convInputs);

        addConvLayerButton.addActionListener(e -> {
            try {
                int filters = Integer.parseInt(inputConvFilters.getText());
                int size = Integer.parseInt(inputConvWindowSize.getText());
                int stride = Integer.parseInt(inputConvStride.getText());

                if (filters <= 0 || size <= 0 || stride <= 0) {
                    JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
                }
                else {
                    builder.addConvolutionLayer(filters, size, stride, LEARNING_RATE);
                    log("Convolution Layer Added (num filters: " + filters + ", size: " + size + ", stride: " + stride + ")");
                }
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
            }
        });


        // Add max pool layer (window size | stride)
        JButton addPoolLayerButton = new JButton("Add Max Pool Layer");
        controlPanel.add(addPoolLayerButton);

        JPanel poolInputs = new JPanel(new GridLayout(2, 2, 10, -5));
        poolInputs.add(new JLabel("window size"));
        poolInputs.add(new JLabel("stride"));
        JTextField inputPoolWindowSize = new JTextField("3", 5);
        poolInputs.add(inputPoolWindowSize);
        JTextField inputPoolStride = new JTextField("2", 5);
        poolInputs.add(inputPoolStride);
        controlPanel.add(poolInputs);

        addPoolLayerButton.addActionListener(e -> {
            try {
                int window = Integer.parseInt(inputPoolWindowSize.getText());
                int stride = Integer.parseInt(inputPoolStride.getText());

                if (window <= 0 || stride <= 0) {
                    JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
                }
                else {
                    builder.addMaxPoolLayer(window, stride);
                    log("Max Pool Layer Added (window: " + window + ", stride: " + stride + ")");
                }
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
            }
        });


        // Add fully connected layer (output size)
        JButton addFCLayerButton = new JButton("Add Fully Connected Layer");
        controlPanel.add(addFCLayerButton);

        JPanel fcInputs = new JPanel(new GridLayout(2, 1, 10, -5));
        fcInputs.add(new JLabel("output size"));
        JTextField inputFCOutputSize = new JTextField("10", 5);
        fcInputs.add(inputFCOutputSize);
        controlPanel.add(fcInputs);

        addFCLayerButton.addActionListener(e -> {
            try {
                int output = Integer.parseInt(inputFCOutputSize.getText());

                if (output <= 0) {
                    JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
                }
                else {
                    builder.addFullyConnectedLayer(output, LEARNING_RATE);
                    log("Fully Connected Layer Added (output size: " + output + ")");
                }
            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
            }
        });


        // Build model or Clear layers (name)
        JPanel buildOrClearPanel = new JPanel(new GridLayout(1, 2, 10, 10));

        JButton buildModelButton = new JButton("Build Model");
        JButton clearLayersButton = new JButton("Clear Layers");
        buildOrClearPanel.add(buildModelButton);
        buildOrClearPanel.add(clearLayersButton);
        controlPanel.add(buildOrClearPanel);

        JPanel buildInputs = new JPanel(new GridLayout(2, 1, 10, -5));
        buildInputs.add(new JLabel("model name"));
        JTextField inputBuildModelName = new JTextField("myModel", 5);
        buildInputs.add(inputBuildModelName);
        controlPanel.add(buildInputs);

        buildModelButton.addActionListener(e -> {
            try {
                currentModel = builder.build();
                currentModel.setNetworkName(inputBuildModelName.getText());
                log("Model successfully built! (name: " + currentModel.getNetworkName() + ")");
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Could not build model.");
            }
        });
        clearLayersButton.addActionListener(e -> {
            builder.clearAllLayers();
            log("CLEARED LAYERS OF BUILDER.\n");
        });


        // Train (epochs)
        JButton trainEpochsButton = new JButton("Train for N Epochs");
        controlPanel.add(trainEpochsButton);

        JPanel trainEpochInputs = new JPanel(new GridLayout(2, 1, 10, -5));
        trainEpochInputs.add(new JLabel("number of epochs"));
        JTextField inputTrainEpochs = new JTextField("3");
        trainEpochInputs.add(inputTrainEpochs);
        controlPanel.add(trainEpochInputs);

        trainEpochsButton.addActionListener(e -> {
            try {
                int numEpochs = Integer.parseInt(inputTrainEpochs.getText());

                if (numEpochs <= 0) {
                    JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
                }
                else {
                    log("Attempting training " + currentModel.getNetworkName() + " for " + numEpochs + " epochs...");
                    trainEpochs(numEpochs);
                    log("Training Complete.");
                }

            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter positive numeric values.");
            }
        });


        // Train (accuracy)
        JButton trainUntilAccuracyButton = new JButton("Train Until Accuracy");
        controlPanel.add(trainUntilAccuracyButton);

        JPanel trainAccuracyInputs = new JPanel(new GridLayout(2, 1, 10, -5));
        trainAccuracyInputs.add(new JLabel("target accuracy"));
        JTextField inputTrainAccuracy = new JTextField("0.95");
        trainAccuracyInputs.add(inputTrainAccuracy);
        controlPanel.add(trainAccuracyInputs);

        trainUntilAccuracyButton.addActionListener(e -> {
            try {
                double targetAccuracy = Double.parseDouble(inputTrainAccuracy.getText());

                if (targetAccuracy < 0 || targetAccuracy > 1) {
                    JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter decimal of [0, 1].");
                }
                else {
                    log("Attempting training " + currentModel.getNetworkName() + " until accuracy of " + targetAccuracy + " achieved...");
                    trainUntilTargetAccuracy(targetAccuracy);
                    log("Training Complete.");
                }

            } catch (NumberFormatException ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Invalid input. Please enter decimal of [0, 1].");
            }
        });

        // Save model
        JButton saveButton = new JButton ("Save Current Model");
        controlPanel.add(saveButton);

        saveButton.addActionListener(e -> {
            String modelPath = generateModelPath();

            try {
                currentModel.saveToFile(modelPath);
                log("Model successfully saved! (name: " + currentModel.getNetworkName() + ", path: " + modelPath + ")");
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(null, "ERROR: Could not save model.");
            }
        });


        // Test all models
        JButton testAllButton = new JButton("Test All Saved Models");
        controlPanel.add(testAllButton);

        testAllButton.addActionListener(this::testAllSavedModels);


        // Test chosen model (filepath)
        JButton testChosenModel = new JButton ("Test Chosen Model");
        controlPanel.add(testChosenModel);

        JPanel chooseInputs = new JPanel(new GridLayout(2, 1, 10, -5));
        chooseInputs.add(new JLabel("choose model"));
        JComboBox<String> inputModelDropdown = new JComboBox<>();
        inputModelDropdown.setFont(new Font("Monospaced", Font.PLAIN, 12));
        File folder = new File("models/");
        File[] files = folder.listFiles((dir, name) -> name.endsWith(".txt"));
        if (files != null) {
            Arrays.sort(files, Comparator.comparing(File::getName));
            String[] networkNames = new String[files.length];

            int maxFilePathSize = 0;
            int maxNetworkNameSize = 0;
            for (int i = 0; i < files.length; i++) {
                File file = files[i];
                try {
                    NeuralNetwork net = new NeuralNetwork(file.getPath());
                    if (file.getPath().length() > maxFilePathSize) {
                        maxFilePathSize = file.getPath().length();
                    }
                    networkNames[i] = net.getNetworkName();
                    if (net.getNetworkName().length() > maxNetworkNameSize) {
                        maxNetworkNameSize = net.getNetworkName().length();
                    }
                } catch (IOException ex) {
                    networkNames[i] = null;
                    log("File " + file.getPath() + " is unreadable.");
                }
            }

            for (int i = 0; i < networkNames.length; i++) {
                if (networkNames[i] != null) {
                    String formatter = "%" + maxNetworkNameSize + "s (%" + maxFilePathSize + "s)";
                    String selectionOption = String.format(formatter, networkNames[i], files[i].getPath());
                    inputModelDropdown.addItem(selectionOption);
                }
            }
            // inputModelDropdown.addItem(net.getNetworkName() + " (" + file.getPath() + ")");
        }
        chooseInputs.add(inputModelDropdown);
        controlPanel.add(chooseInputs);

        testChosenModel.addActionListener(e -> {
            String selectedString = inputModelDropdown.getSelectedItem().toString();
            selectedString = selectedString.substring(selectedString.indexOf('(') + 1, selectedString.indexOf(')'));
            testChosenModel(selectedString.strip());
        });


        frame.add(controlPanel, BorderLayout.WEST);

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private void log(String text) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(text + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }

    private void loadData(ActionEvent e, String trainFilepath, String testFilepath) {
        imagesTrain = new DataReader(GRID_ROWS, GRID_COLS).readData(trainFilepath); // "data/mnist_train.csv"
        imagesTest = new DataReader(GRID_ROWS, GRID_COLS).readData(testFilepath); // "data/mnist_test.csv"

        DataReader.standardizeNormal(imagesTrain);
        DataReader.standardizeNormal(imagesTest);

        log("Training and test data loaded and standardized. (training size: " + imagesTrain.size() + ", testing size: " + imagesTest.size() + ")");
    }

    private void trainEpochs(int numEpochs) {
        new Thread(() -> {
            if (imagesTrain == null || imagesTest == null) {
                log("Please load data first.");
                return;
            }

            try {
                for (int i = 1; i <= numEpochs; i++) {
                    shuffle(imagesTrain);
                    currentModel.train(imagesTrain);
                    double accuracy = currentModel.test(imagesTest);
                    log("Epoch " + i + ": Accuracy = " + String.format("%.2f%%", accuracy * 100));
                }


                String modelPath = generateModelPath();
                currentModel.saveToFile(modelPath);
                log("Current model saved to: " + modelPath);
            } catch (Exception ex) {
                log("Error: " + ex.getMessage() + "\n");
            }
        }).start();
    }

    private void trainUntilTargetAccuracy(double targetAccuracy) {
        new Thread(() -> {
            if (imagesTrain == null || imagesTest == null) {
                log("Please load data first.");
                return;
            }

            try {
                int maxFails = 5;

                int failCount = 0;
                int epoch = 0;
                double bestAcc = 0;
                String modelPath = generateModelPath();

                log("Generated model path: " + modelPath);

                while (failCount < maxFails) {
                    epoch++;
                    shuffle(imagesTrain);
                    currentModel.train(imagesTrain);
                    double acc = currentModel.test(imagesTest);

                    if (acc > bestAcc) {
                        bestAcc = acc;
                        currentModel.saveToFile(modelPath);
                        log("Saving Epoch " + epoch + " | Accuracy = " + String.format("%.2f%%", acc * 100));
                    } else {
                        failCount++;
                        log("Fail-" + failCount + " Epoch " + epoch + " | Accuracy = " + String.format("%.2f%%", acc * 100) + " (no improvement)");
                    }

                    if (bestAcc >= targetAccuracy) break;
                }

                log("Best model saved to: " + modelPath);

                if (failCount >= maxFails) {
                    log("(note, current model has been over-trained... current model is not the same as best model)");
                }
                else {
                    log("(current model is the same as best model)");
                }

            } catch (Exception ex) {
                log("Error: " + ex.getMessage() + "\n");
            }
        }).start();
    }

    private void testAllSavedModels(ActionEvent e) {
        new Thread(() -> {
            if (imagesTest == null) {
                log("Please load data first.");
                return;
            }

            File folder = new File("models/");
            File[] files = folder.listFiles((dir, name) -> name.endsWith(".txt"));

            if (files == null || files.length == 0) {
                log("No models found in /models.");
                return;
            }

            Arrays.sort(files, Comparator.comparing(File::getName));

            for (File file : files) {
                try {
                    NeuralNetwork net = new NeuralNetwork(file.getPath());
                    double acc = net.test(imagesTest);

                    log(String.format("%15s (%20s) - Accuracy: %7s", net.getNetworkName(), net.getNetworkFilePath(), String.format("%.2f%%", acc * 100)));
                } catch (IOException ex) {
                    log("Failed to load file " + file.getName());
                }
            }
        }).start();
    }

    private void testChosenModel(String modelPath) {
        try {
            NeuralNetwork nn = new NeuralNetwork(modelPath);
            new DigitRecognizer(nn, 28);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private String generateModelPath() {
        int i = 0;
        while (new File("models/model_" + i + ".txt").exists()) i++;
        return "models/model_" + i + ".txt";
    }
}
