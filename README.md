# ML_DL_Labs

End-to-end Machine Learning and Deep Learning implementations built in Google Colab using **TensorFlow/Keras**, **Scikit-Learn**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**. This repository showcases regularized Convolutional Neural Networks for computer vision alongside supervised Linear Regression pipelines.

---

## Models & Performance Summary

| Category | Project | Dataset | Architecture / Algorithm | Key Details & Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Deep Learning** | **CIFAR-10 Image Classifier** | CIFAR-10 (60k RGB images, 10 classes) | Deep CNN (`Conv2D` 32→64→128→256 + `BatchNorm` + `Dropout`) | **654,858** Trainable Params |
| **Deep Learning** | **MNIST Digit Recognizer** | MNIST (70k grayscale digits, 0–9) | Neural Network / CNN Classifier | **97.73%** Test Accuracy (`0.0943` Loss) |
| **Machine Learning** | **House Price Predictor** | Synthetic / Generated Housing Dataset | Linear Regression | Continuous Price Modeling |
| **Machine Learning** | **Student Result Predictor** | Custom Study Hours, sleep , attendance,etc vs. Score Dataset | Multi_Varient Linear Regression | Score Trend Prediction |

---

## Project Breakdown

### 1. Deep Learning & Computer Vision
* **CIFAR-10 Image Classification (`cifar10_cnn.ipynb`)**
  * Engineered a regularized Sequential CNN (`maaz`) in TensorFlow/Keras featuring four `Conv2D` blocks (`32`, `64`, `128`, and `256` filters), paired with `BatchNormalization`, `MaxPooling2D`, and `Dropout` layers to prevent overfitting[cite: 4, 5].
  * Flattened feature maps (`1,024` units) into a `256`-unit Dense classifier head (`654,858` trainable parameters / `1,311,190` total parameters).
  * Visualized training/validation accuracy and loss curves with Matplotlib.
* **MNIST Handwritten Digit Classification (`mnist_classification.ipynb`)**
  * Built and trained a classification pipeline on $28 \times 28$ grayscale handwritten digits (`0–9`), achieving **97.73% test accuracy** and **0.0943 test loss**.
  * Evaluated generalization on unseen test data and plotted confusion matrices using Seaborn.

### 2. Supervised Machine Learning (Linear Regression)
* **House Price Prediction (`house_price_linear_regression.ipynb`)**
  * Implemented a pure Linear Regression pipeline using Scikit-Learn on generated numerical housing data to model continuous property prices.
  * Visualized regression fit lines, residual distributions, and feature correlations with Matplotlib and Seaborn.
* **Student Score Prediction (`student_score_linear_regression.ipynb`)**
  * Built a custom Linear Regression model predicting student exam scores based on daily study hours.
  * Analyzed the linear relationship between study duration and academic performance with exploratory scatter and best-fit plots.

---

## Tech Stack

* **Deep Learning:** TensorFlow, Keras
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebooks (`.ipynb`)

---

## How to Run

1. **Run in Google Colab (Recommended):**
   * Open any `.ipynb` notebook in this repository and launch it directly in [Google Colab](https://colab.research.google.com/) (tested with T4 GPU runtime).
2. **Run Locally:**
   ```bash
   git clone [https://github.com/MuhammadAI-Expert/ML_DL_Labs.git](https://github.com/MuhammadAI-Expert/ML_DL_Labs.git)
   cd ML_DL_Labs
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
   jupyter notebook
