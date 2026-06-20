# 📡🏙️ Deep Learning - Signal Quality Prediction & Street View Digit Recognition

This project applies **Artificial Neural Networks (ANN)** to solve two real-world classification problems across two distinct domains: Electronics & Telecommunication and Autonomous Vehicles.

---

### Project Objective

To design and evaluate deep learning models that can accurately classify multi-class targets from structured and image-based data, while exploring optimization techniques such as Batch Normalization, Dropout, and Kernel Initialization to improve model performance.

---

### Dataset Summary

**Part 1 - Electronics & Telecommunication**
- Dataset of signal test records from a communications equipment manufacturing company
- Each row represents a signal test with 11 measurable parameters
- Target variable: **Signal Strength / Quality** (multi-class: classes 3-8)
- Key challenges: class imbalance and duplicate parameter records

**Part 2 - Google Street View House Number (SVHN)**
- Real-world image dataset sourced from Google Street View photographs
- Images are grayscale, centred around a single digit with visual distractors
- Data split: 42,000 train / 18,000 test / 60,000 validation samples
- Target variable: **Digit class** (10 classes: 0-9)
- Key challenges: lighting variations, shadows, occlusions, and motion blur

---

### Approach

**Part 1 - Signal Quality**
- Exploratory Data Analysis (EDA) on signal parameters
- Handling duplicate records via group-mean imputation
- StandardScaler normalization and OneHotEncoding for multi-class labels
- 70:30 stratified train-test split
- Multiple ANN architectures evaluated with varying regularization techniques

**Part 2 - Street View Digit Recognition**
- Loading and preprocessing `.h5` format image data
- Pixel normalization (dividing by 255) and label encoding via `to_categorical`
- Flattened grayscale images (32×32 = 1024 features) fed into a fully connected ANN
- Model evaluated using classification report (precision, recall, F1-score)

---

### Deep Learning Models

**Part 1 - Signal Quality Prediction**

| Model | Architecture | Key Technique | Test Accuracy |
|---|---|---|---|
| Baseline ANN | 128 → 64 → 32 → 6 | ReLU + Softmax | ~60% |
| Model 1 | 64 → 32 → 16 → 6 | Kernel Initializer | ~58% |
| Model 2 | 64 → 32 → 16 → 6 | Batch Normalization | ~58% |
| Model 3 | 64 → 32 → 16 → 6 | Dropout (0.5) + BatchNorm | ~56% |

**Part 2 - Street View Digit Recognition**

| Model | Architecture | Key Technique | Test Accuracy |
|---|---|---|---|
| Baseline ANN | 256 → 128 → 64 → 32 → 10 | ReLU + Softmax | **81%** |
| Enhanced ANN | 256 → 128 → 64 → 32 → 10 | BatchNorm + He Normal Init | Overfitting observed |

> The **baseline ANN with 3 hidden layers** achieved the best test accuracy of **81%**, with training and validation losses closely converging. The Batch Normalization variant overfit on the training data and was not selected as the final model.

---

### Tools Used

- Python, NumPy, Pandas, Matplotlib, Seaborn
- Data Handling: `h5py`, `sklearn.preprocessing`
- Deep Learning: TensorFlow / Keras (`Sequential`, `Dense`, `BatchNormalization`, `Dropout`)
- Optimizers: Adam (learning rate 0.001)
- Evaluation: `classification_report` (Precision, Recall, F1-Score)
- Environment: Google Colab

---

### 🔍 Use Case

**Part 1** enables communications equipment companies to proactively monitor and classify signal quality using measurable parameters, supporting predictive maintenance and quality assurance pipelines.

**Part 2** demonstrates how neural networks can automate address localization from street-level imagery — a core component in building accurate, scalable digital maps for autonomous navigation systems.
