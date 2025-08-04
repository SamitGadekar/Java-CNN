package data;

/**
 * MatrixUtility.java
 * A utility class providing common matrix operations such as addition and multiplication.
 * Supports element-wise operations for both 1D and 2D arrays.
 *
 * @author Samit Gadekar
 * @version January 1, 2025
 */

public class MatrixUtility {

    /**
     * Performs element-wise addition of two matrices.
     *
     * @param matA First matrix (2D array).
     * @param matB Second matrix (2D array).
     * @return A new matrix representing the element-wise sum of 'matA' and 'matB'.
     */
    public static double[][] add(double[][] matA, double[][] matB) {
        double[][] out = new double[matA.length][matA[0].length];
        for (int i = 0; i < matA.length; i++) {
            for (int j = 0; j < matA[0].length; j++) {
                out[i][j] = matA[i][j] + matB[i][j];
            }
        }
        return out;
    }

    /**
     * Performs element-wise addition of two vectors.
     *
     * @param vecA First vector (1D array).
     * @param vecB Second vector (1D array).
     * @return A new vector representing the element-wise sum of 'vecA' and 'vecB'.
     */
    public static double[] add(double[] vecA, double[] vecB) {
        double[] out = new double[vecA.length];
        for (int i = 0; i < vecA.length; i++) {
            out[i] = vecA[i] + vecB[i];
        }
        return out;
    }

    /**
     * Performs element-wise multiplication of two matrices.
     * This is NOT traditional matrix multiplication but Hadamard (element-wise) multiplication.
     *
     * @param matA First matrix (2D array).
     * @param matB Second matrix (2D array).
     * @return A new matrix where each element is the product of the corresponding elements in 'matA' and 'matB'.
     */
    public static double[][] multiply(double[][] matA, double[][] matB) {
        double[][] out = new double[matA.length][matA[0].length];
        for (int i = 0; i < matA.length; i++) {
            for (int j = 0; j < matA[0].length; j++) {
                out[i][j] = matA[i][j] * matB[i][j];
            }
        }
        return out;
    }

    /**
     * Performs scalar multiplication on a matrix.
     * Each element in the matrix is multiplied by the given scalar.
     *
     * @param matA The input matrix (2D array).
     * @param scalar The scalar value to multiply each element by.
     * @return A new matrix where each element of 'matA' is multiplied by 'scalar'.
     */
    public static double[][] multiply(double[][] matA, double scalar) {
        double[][] out = new double[matA.length][matA[0].length];
        for (int i = 0; i < matA.length; i++) {
            for (int j = 0; j < matA[0].length; j++) {
                out[i][j] = matA[i][j] * scalar;
            }
        }
        return out;
    }

    /**
     * Performs scalar multiplication on a vector.
     * Each element in the vector is multiplied by the given scalar.
     *
     * @param vecA The input vector (1D array).
     * @param scalar The scalar value to multiply each element by.
     * @return A new vector where each element of 'vecA' is multiplied by 'scalar'.
     */
    public static double[] multiply(double[] vecA, double scalar) {
        double[] out = new double[vecA.length];
        for (int i = 0; i < vecA.length; i++) {
            out[i] = vecA[i] * scalar;
        }
        return out;
    }
}
