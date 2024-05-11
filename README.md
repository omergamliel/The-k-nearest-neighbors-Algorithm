# The K-Nearest Neighbors Algorithm

![K-Nearest Neighbors](https://miro.medium.com/v2/resize:fit:505/0*2_qzcm2gSe9l67aI.png)

## Introduction
The K-Nearest Neighbors (KNN) algorithm is a simple, yet powerful machine learning technique used for both classification and regression. It belongs to the supervised learning domain and works by finding the most similar instances (neighbors) in the training data to a new instance and predicting its label by majority vote or averaging.

## About This Project
This project demonstrates the implementation of the KNN algorithm on the Iris dataset, a widely used multivariate dataset introduced by Sir Ronald Fisher in 1936. The Iris dataset consists of 50 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor) with four features describing the width and the length of the sepals and petals.

## Features of the Project
- **Data Preprocessing**: Normalizing the dataset using min-max scaling.
- **Distance Calculation**: Compute the Euclidean distance between each point.
- **KNN Algorithm**: Implement the KNN algorithm from scratch.
- **Accuracy Measurement**: Calculate the classification accuracy of the model.

## Requirements
- Python 3.x
- NumPy
- Scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/The-k-nearest-neighbors-Algorithm.git

## How to Run
To run the project, execute the following command in the terminal:
```bash

## Code Explanation
- **Data Loading and Preparation**:
  - `datasets.load_iris()` loads the Iris dataset.
  - Data is shuffled using `np.random.permutation` to randomize the order of samples.
  - The dataset is split into a training set (80%) and a test set (20%).

- **Feature Scaling**:
  - `max_min_Scalling` function scales the dataset features to a 0-1 range using Min-Max scaling, which helps in improving the performance of the kNN algorithm.

- **Distance Calculation**:
  - `euclidean_distance` function calculates the Euclidean distance between each test sample and all training samples.

- **Prediction**:
  - The `predict` function sorts the distances to find the nearest neighbors (based on the value of `k`), determines the most common class among these neighbors, and returns the predicted class.

- **Accuracy Calculation**:
  - The `main_knn` function calculates the overall classification accuracy by comparing the predicted labels with the true labels of the test set.

This script demonstrates a simple implementation of the kNN algorithm and shows how to assess its accuracy with a specific value of `k`.
