# Java-CNN

## Overview

Java-CNN is a convolutional neural network (CNN) implemented in Java, built as an extension and learning exercise inspired by the [CNN\_Tutorial](https://github.com/evarae/CNN_Tutorial) by evarae. This project aims to provide a deeper understanding of CNNs by coding them from scratch, following a structured approach to neural network design and optimization.

## Inspiration

This project is based on the [Convolutional Neural Network Tutorial](https://www.youtube.com/watch?v=3MMonOWGe0M&list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN) by evarae, which walks through the development of a CNN in Java. While following the tutorial, I have implemented additional improvements, optimizations, and experiments to enhance performance and generalization.

## Features

* Handcrafted convolutional, fully connected, and max pool layers without external deep learning libraries
* Backpropagation implementation for learning
* Training and evaluation on the MNIST dataset for recognizing handwritten numerical digits
* Functionality for saving and loading trained networks to and from files
* GUI for creating, testing, and interacting with models in real time

## Requirements

* Java 8+

## Setup and Execution

1. Clone the repository
```bash
git clone https://github.com/SamitGadekar/Java-CNN.git
cd Java-CNN
```
2. Project structure
```
/data
  |-- DataReader.java
  |-- Image.java
  |-- MatrixUtility.java
/layers
  |-- Layer.java
  |-- ConvolutionLayer.java
  |-- MaxPoolLayer.java
  |-- FullyConnectedLayer.java
/network
  |-- NeuralNetwork.java
  |-- NetworkBuilder.java
/interactive
  |-- ModelBuilder.java
  |-- DigitRecognizer.java
Main.java
```
3. Dependencies  
This project uses only standard Java libraries, so no external dependencies or package managers are required. (```java.util```, ```java.io```, ```java.awt```, ```javax.swing```)
4. Running the program
You can run the project using an IDE (like IntelliJ IDEA or Eclipse) or from the terminal.
  
**Run from IDE**
- Open the project folder in your IDE.
- Make sure all folders (```/data```, ```/layers```, ```/network```, ```/interactive```) are marked as source directories.
- Run ```Main.java```.

**Run from Terminal**
- Assuming your ```.java``` files are inside a ```src``` folder:
```bash
javac -d out src/Main.java src/*.java src/data/*.java src/layers/*.java src/network/*.java src/interactive/*.java
java -cp out Main
```
You can adjust the paths above if your ```.java``` files are not inside a ```src``` directory.
5. Once you have opened and run the project, experiment with creating new models and test them out in real time by drawing with your mouse.

## Reference Tutorial

The original tutorial repository can be found [here](https://github.com/evarae/CNN_Tutorial). The video series explaining the implementation in detail starts [here](https://youtu.be/3MMonOWGe0M).

## Contributing

If you'd like to contribute, feel free to submit issues or pull requests to enhance this project further!
