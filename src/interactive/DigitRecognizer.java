package interactive;

import data.Image;
import data.DataReader;
import network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

/**
 * DigitRecognizer.java
 * An interactive GUI for drawing digits and recognizing them using a neural network.
 *
 * @author Samit Gadekar
 * @version April 2, 2025
 */
public class DigitRecognizer {
    private static final int PIXEL_SIZE = 20;  // Size of each cell in pixels
    private static final double RADIUS = 3; // Radius of drawing cells

    private final int GRID_SIZE;  // n x n grid for digit input
    private final double[][] pixelGrid; // Stores active pixel states
    private double ogPixel = 0.0; // Stores the state of the pixel clicked to maintain consistent drawing
    private final NeuralNetwork neuralNetwork; // Neural network for classification
    private final JPanel sidePanel; // Side panel for displaying prediction bars
    private final JProgressBar[] predictionBars; // Progress bars for digit probabilities
    private final int[] darkModeMultipliers = {1, -1};

    /**
     * Main method to launch the GUI.
     */
    public static void main(String[] args) {
        try {
            NeuralNetwork nn = new NeuralNetwork("models/model_5.txt");
            new DigitRecognizer(nn, 28);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Constructor initializes the GUI and sets up event listeners.
     *
     * @param nn The trained neural network for digit recognition.
     */
    public DigitRecognizer(NeuralNetwork nn, int gridSize) {
        this.neuralNetwork = nn;
        this.GRID_SIZE = gridSize;
        this.pixelGrid = new double[GRID_SIZE][GRID_SIZE];

        JFrame frame = new JFrame("Neural Network Number Recognizer");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

        // Canvas for drawing
        PixelCanvas canvas = new PixelCanvas();
        frame.add(canvas, BorderLayout.CENTER);

        // Side panel for guess probabilities
        sidePanel = new JPanel();
        sidePanel.setLayout(new GridLayout(10, 1));
        predictionBars = new JProgressBar[10];

        for (int i = 0; i < 10; i++) {
            JPanel barPanel = new JPanel(new BorderLayout());
            JLabel label = new JLabel(i + ": ");
            JProgressBar bar = new JProgressBar(0, 100);
            bar.setStringPainted(true);
            predictionBars[i] = bar;
            barPanel.add(label, BorderLayout.WEST);
            barPanel.add(bar, BorderLayout.CENTER);
            sidePanel.add(barPanel);
        }

        frame.add(sidePanel, BorderLayout.EAST);

        // Bottom panel with buttons
        JPanel buttonPanel = new JPanel();

        JButton selectModelButton = new JButton("Select Model");
        JButton drawButton = new JButton("Draw");
        JButton eraseButton = new JButton("Erase");
        JButton clearButton = new JButton("Clear");
        JButton lightDarkButton = new JButton("Light/Dark");

        selectModelButton.addActionListener(e -> {

        });
        drawButton.addActionListener(e -> {
            ogPixel = 0.0;
        });
        eraseButton.addActionListener(e -> {
            ogPixel = 1;
        });
        clearButton.addActionListener(e -> {
            clearGrid();
            canvas.repaint();
        });
        lightDarkButton.addActionListener(e -> {
            if (darkModeMultipliers[0] == 1) {
                darkModeMultipliers[0] = 0;
                darkModeMultipliers[1] = 1;
            } else {
                darkModeMultipliers[0] = 1;
                darkModeMultipliers[1] = -1;
            }
            canvas.repaint();
        });

        buttonPanel.add(selectModelButton);
        buttonPanel.add(drawButton);
        buttonPanel.add(eraseButton);
        buttonPanel.add(clearButton);
        buttonPanel.add(lightDarkButton);
        frame.add(buttonPanel, BorderLayout.SOUTH);

        frame.setSize(GRID_SIZE * PIXEL_SIZE + 100, GRID_SIZE * PIXEL_SIZE + 20);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }


    /**
     * Runs the neural network to show the prediction of the drawn digit.
     */
    private void predictDigit() {
        double[][] input = new double[GRID_SIZE][GRID_SIZE];

        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                input[row][col] = pixelGrid[row][col];
            }
        }

        // Convert input to Image object and standardizes
        Image inputImage = new Image(input, -1);
        List<Image> images = List.of(inputImage);
        DataReader.standardizeNormal(images);

        // Get prediction from neural network
        double[] output = neuralNetwork.getOutput(images.get(0));

        // Find the most probable digit
        int maxIndex = getMaxIndex(output);

        // Update side panel bars
        for (int i = 0; i < 10; i++) {
            predictionBars[i].setValue((int) (output[i] * 100));
            predictionBars[i].setString(String.format("%.2f%%", output[i] * 100));

            // Highlight the most probable digit
            if (i == maxIndex) {
                predictionBars[i].setBackground(Color.RED); // Highlight in red
                predictionBars[i].setFont(new Font("Roboto", Font.BOLD, 15)); // Bold font
            } else {
                predictionBars[i].setBackground(Color.BLACK); // Reset others to black
                predictionBars[i].setFont(new Font("Arial", Font.PLAIN, 12)); // Normal font
            }
        }
    }

    /**
     * Clears the grid, resetting all pixels to OFF.
     */
    private void clearGrid() {
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                pixelGrid[row][col] = 0;
            }
        }
    }

    /**
     * Finds the index of the maximum value in an array.
     *
     * @param output The array of neural network output probabilities.
     * @return The index of the highest value, representing the predicted digit.
     */
    private int getMaxIndex(double[] output) {
        int maxIndex = 0;
        double maxVal = output[0];

        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double[][] exportGrid(PixelCanvas canvas) {
        double[][] returnedGrid = new double[GRID_SIZE][GRID_SIZE];
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                returnedGrid[i][j] = pixelGrid[i][j];
                pixelGrid[i][j] = 0;
            }
        }
        canvas.repaint();
        //printGrid(returnedGrid);
        return returnedGrid;
    }
    private void printGrid(int[][] g) {
        System.out.println("2D Array of Pixels:");
        for (int[] intArray : g) {
            for (int i: intArray) {
                System.out.print(i + " ");
            }
            System.out.println();
        }
    }
    private class PixelCanvas extends JPanel {
        public PixelCanvas() {
            setPreferredSize(new Dimension(GRID_SIZE * PIXEL_SIZE, GRID_SIZE * PIXEL_SIZE));
            addMouseListener(new MouseAdapter() {
                @Override
                public void mousePressed(MouseEvent e) {
                    togglePixel(e.getX(), e.getY());
                }
            });

            addMouseMotionListener(new MouseAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    toggleNextPixel(e.getX(), e.getY());
                }
            });
        }

        private void togglePixel(int x, int y) {
            int col = x / PIXEL_SIZE;
            int row = y / PIXEL_SIZE;
            if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
                if (ogPixel == 0) {
                    brush(row, col, x, y);
                } else {
                    erase(row, col, x, y);
                }
                repaint();
                /*
                double temp = pixelGrid[row][col];
                if (pixelGrid[row][col] == 0) {
                    brush(row, col, x, y);
                    repaint();
                    return temp;
                } else {
                    erase(row, col, x, y);
                    repaint();
                    return temp;
                }
                */
            }
            // return 0.0;
        }
        private void toggleNextPixel(int x, int y) {
            int col = x / PIXEL_SIZE;
            int row = y / PIXEL_SIZE;
            if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
                if (ogPixel == 0) {
                    brush(row, col, x, y);
                } else {
                    erase(row, col, x, y);
                }
                repaint();
            }
        }

        /**
         * Applies a circular brush effect around the given pixel.
         */
        private void brush(int row, int col, int x, int y) {
            double centerX = (double) x / PIXEL_SIZE;
            double centerY = (double) y / PIXEL_SIZE;
            for (int i = (int) Math.floor(-RADIUS); i <= Math.ceil(RADIUS); i++) {
                for (int j = (int) Math.floor(-RADIUS); j <= Math.ceil(RADIUS); j++) {
                    int newRow = row + i;
                    int newCol = col + j;
                    if (newRow >= 0 && newRow < GRID_SIZE && newCol >= 0 && newCol < GRID_SIZE) {
                        double newColDelta = newCol - centerX;
                        double newRowDelta = newRow - centerY;
                        double distance = Math.sqrt(newRowDelta * newRowDelta + newColDelta * newColDelta);
                        if (distance <= RADIUS) { // Ensures a circular effect
                            double intensity = RADIUS / Math.pow(Math.E, Math.pow(distance, 2));
                            if (intensity > 1) intensity = 1;
                            if (intensity < 0) intensity = 0;
                            if (pixelGrid[newRow][newCol] < intensity) pixelGrid[newRow][newCol] = intensity;
                        }
                    }
                }
            }
            repaint();
        }
        private void erase(int row, int col, int x, int y) {
            double centerX = (double) x / PIXEL_SIZE;
            double centerY = (double) y / PIXEL_SIZE;
            for (int i = (int) Math.floor(-RADIUS); i <= Math.ceil(RADIUS); i++) {
                for (int j = (int) Math.floor(-RADIUS); j <= Math.ceil(RADIUS); j++) {
                    int newRow = row + i;
                    int newCol = col + j;
                    if (newRow >= 0 && newRow < GRID_SIZE && newCol >= 0 && newCol < GRID_SIZE) {
                        double newColDelta = newCol - centerX;
                        double newRowDelta = newRow - centerY;
                        double distance = Math.sqrt(newRowDelta * newRowDelta + newColDelta * newColDelta);
                        if (distance <= RADIUS * 1.25) {
                            double intensity = RADIUS / Math.pow(Math.E, Math.pow(distance, 0.5));
                            if (intensity > 1) intensity = 1;
                            if (intensity < 0) intensity = 0;
                            intensity = 1 - intensity;
                            if (pixelGrid[newRow][newCol] > intensity) pixelGrid[newRow][newCol] = intensity;
                        }
                    }
                }
            }
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            for (int row = 0; row < GRID_SIZE; row++) {
                for (int col = 0; col < GRID_SIZE; col++) {

                    if (pixelGrid[row][col] > 1) pixelGrid[row][col] = 1;
                    if (pixelGrid[row][col] < 0) pixelGrid[row][col] = 0;

                    // Convert double value to grayscale (0 = black, 255 = white)
                    int grayValue = (int) (255 * (darkModeMultipliers[0] + darkModeMultipliers[1] * pixelGrid[row][col]));
                    g.setColor(new Color(grayValue, grayValue, grayValue));
                    g.fillRect(col * PIXEL_SIZE, row * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);

                    // g.setColor(Color.GRAY);
                    // g.drawRect(col * PIXEL_SIZE, row * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
                }
            }
            predictDigit();
        }
    }

}
