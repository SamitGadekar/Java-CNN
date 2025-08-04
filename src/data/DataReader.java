package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * DataReader.java
 * Handles reading and processing image data from a file.
 * Provides methods to normalize and standardize image pixel values.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class DataReader {

    private final int rows;
    private final int cols;

    /**
     * Constructor to initialize the DataReader with image dimensions.
     *
     * @param rows Number of rows in each image.
     * @param cols Number of columns in each image.
     */
    public DataReader(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Reads image data from a CSV file and converts it into a list of Image objects.
     * Each row in the CSV file represents one image, with the first value being the label
     * and the remaining values being pixel intensities.
     *
     * @param path The file path to read data from.
     * @return A list of Image objects containing pixel data and labels.
     */
    public List<Image> readData(String path) {

        List<Image> images = new ArrayList<>();

        try (BufferedReader dataReader = new BufferedReader(new FileReader(path))) {
            String line;

            while ((line = dataReader.readLine()) != null) {
                String[] lineItems = line.split(",");

                int label = Integer.parseInt(lineItems[0]);

                double[][] data = new double[rows][cols];
                int i = 1;
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        data[row][col] = Double.parseDouble(lineItems[i++]);
                    }
                }

                images.add(new Image(data, label));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return images;
    }

    /**
     * Standardizes image pixel values to have a mean of 0 and a standard deviation of 1.
     * Uses the formula: standardized_pixel = (pixel - mean) / std_dev.
     * This ensures pixel values are normally distributed.
     *
     * @param images List of Image objects to standardize.
     */
    public static double[] standardizeNormal(List<Image> images) {
        double sum = 0;
        double sumSq = 0;
        int totalPixels = 0;

        // Calculate mean and standard deviation
        for (Image img : images) {
            double[][] data = img.getData();
            for (double[] row : data) {
                for (double pixel : row) {
                    sum += pixel;
                    sumSq += pixel * pixel;
                    totalPixels++;
                }
            }
        }

        double mean = sum / totalPixels;
        double variance = (sumSq / totalPixels) - (mean * mean);
        double std = Math.sqrt(variance) + 1e-8; // prevents div by 0

        // Standardize each image
        for (Image img : images) {
            double[][] data = img.getData();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = (data[i][j] - mean) / std;
                }
            }
        }

        return new double[]{mean, std};
    }

    /**
     * Standardizes image pixel values to have a mean of 0.5 and a standard deviation of 0.5.
     * Uses the formula: standardized_pixel = 0.5 + 0.5 * ((pixel - mean) / std_dev).
     * This keeps values within the range [0,1] while preserving normality.
     *
     * @param images List of Image objects to standardize.
     */
    public static void standardizeNormalPositive(List<Image> images) {
        double sum = 0;
        double sumSq = 0;
        int totalPixels = 0;

        // Calculate mean and standard deviation
        for (Image img : images) {
            double[][] data = img.getData();
            for (double[] row : data) {
                for (double pixel : row) {
                    sum += pixel;
                    sumSq += pixel * pixel;
                    totalPixels++;
                }
            }
        }

        double mean = sum / totalPixels;
        double variance = (sumSq / totalPixels) - (mean * mean);
        double std = Math.sqrt(variance) + 1e-8; // Prevents division by zero

        // Standardize each image with mean = 0.5 and std = 0.5
        for (Image img : images) {
            double[][] data = img.getData();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = 0.5 + 0.5 * ((data[i][j] - mean) / std);
                }
            }
        }
    }

    /**
     * Normalizes image pixel values to the range [0,1] using min-max scaling.
     * Uses the formula: normalized_pixel = (pixel - min) / (max - min).
     * Ensures that the smallest value becomes 0 and the largest value becomes 1.
     *
     * @param images List of Image objects to normalize.
     */
    public static void standardizeZeroToOne(List<Image> images) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        // Find min and max pixel values
        for (Image img : images) {
            double[][] data = img.getData();
            for (double[] row : data) {
                for (double pixel : row) {
                    if (pixel < min) min = pixel;
                    if (pixel > max) max = pixel;
                }
            }
        }

        double range = max - min;

        // Normalize each pixel to [0,1]
        for (Image img : images) {
            double[][] data = img.getData();
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    data[i][j] = (range < 1e-8) ? 1e-8 : (data[i][j] - min) / range;
                }
            }
        }
    }

}

