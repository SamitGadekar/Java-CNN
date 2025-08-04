package data;

import java.io.Serializable;

/**
 * Image.java
 * Represents an image with pixel data and a label.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class Image implements Serializable {

    private double[][] data;
    private int label;

    /**
     * Constructs an Image object with pixel data and a corresponding label.
     *
     * @param data 2D array representing the pixel values of the image.
     * @param label Integer representing the category or class of the image.
     */
    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }
    public int getLabel() {
        return label;
    }

    /**
     * Converts the Image object into a formatted string.
     * The output format is:
     * label,
     * pixel_value1-1, pixel_value1-2, ...
     * ...
     * pixel_valueN-1, pixel_valueN-2, ...
     *
     * @return A string representation of the image.
     */
    public String toString() {
        StringBuilder s = new StringBuilder(label + ",");
        for (double[] row : data) {
            s.append("\n");
            for (double pixel : row) {
                s.append(pixel).append(",");
            }
        }
        return s.toString();
    }
}
