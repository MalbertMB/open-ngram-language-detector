# Language Detection using Open N-Grams and Naive Bayes

A Python implementation of a **Multinomial Naive Bayes** classifier built from scratch, utilizing **Open N-Grams** for feature extraction. This project explores how non-contiguous character sequences can provide a robust representation of words for language identification tasks, approximating cognitive tolerance to internal character disorder.

---

## Project Overview

This repository contains a NumPy-based implementation of a probabilistic text classifier for language detection. It combines custom open n-gram feature engineering with a Multinomial Naive Bayes model trained using maximum likelihood estimation with Laplace smoothing. The system is designed to be lightweight, interpretable, and resilient to noisy input text.

---

## Features

- **Custom Feature Extraction**: Implements open n-grams (ordered, non-contiguous character subsets) to represent lexical structure.  
- **From-Scratch Model**: Pure NumPy implementation of the Multinomial Naive Bayes algorithm.  
- **Laplace Smoothing**: Mitigates zero-frequency issues in sparse feature spaces.  
- **Noise Robustness**: Maintains classification performance under minor typos or character perturbations due to open n-gram structure.  
- **Modular Design**: Separates feature engineering, model logic, and execution pipeline.  

---

## Directory Structure

```text
language-detection/
├── data/
│   └── dataset.csv        # Input dataset (Text, language)
├── src/
│   ├── features.py        # Open n-gram generation logic
│   └── model.py           # MultinomialNB class implementation
├── main.py                # Training and evaluation pipeline
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd language-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Ensure your dataset is located at:

```
data/dataset.csv
```

The CSV file must contain at least two columns:

- `Text` — the input sentence or document  
- `language` — the corresponding language label  

---

## Usage

To train the model and evaluate classification accuracy:

```bash
python main.py
```

The script performs feature extraction, vectorization (e.g., TF-IDF), model training, and evaluation.

---

## Execution Flow

### 1. Data Loading

- Loads `dataset.csv`.
- Extracts text and language labels.

### 2. Feature Engineering

- Generates open n-grams from input text.
- Converts text into numerical feature vectors.

### 3. Model Training

- Computes class priors.
- Estimates conditional probabilities using Laplace smoothing.

### 4. Evaluation

- Predicts language labels on validation data.
- Reports overall classification accuracy.

---

## Methodology

### Open N-Grams

Unlike standard n-grams (which consist of contiguous characters), open n-grams capture ordered characters that may be separated by other characters within a word. This representation encodes structural information while tolerating internal noise.

Example for the word `HELLO`:

- Standard Bigrams: `HE`, `EL`, `LL`, `LO`  
- Open Bigrams (subset): `HL`, `HO`, `EO`, etc.  

This approach captures a word’s structural "skeleton," improving resilience to typographical variation.

---

### Probabilistic Model

The classifier estimates the posterior probability of a language class $c$ given a document $d$:


$$
P(c \mid d) \propto P(c) \prod_{i=1}^{V} P(w_i \mid c)^{f_i}
$$


Where:

- $P(c)$ is the prior probability of class $c$.  
- $P(w_i \mid c)$ is the conditional probability of feature $w_i$ given class $c$.  
- $f_i$ is the frequency of feature $w_i$ in document $d$.  
- $V$ is the vocabulary size.  

Laplace smoothing is applied:


$$
P(w_i \mid c) =
\frac{N_{ic} + \alpha}{\sum_{j=1}^{V} N_{jc} + \alpha V}
$$

Where:

- $N_{ic}$ is the count of feature $w_i$ in class $c$.  
- $\alpha$ is the smoothing parameter (typically 1).  

---

## Results

The model achieves approximately **95% classification accuracy** on standard language identification datasets, demonstrating that open n-grams provide an effective and computationally efficient feature representation for multilingual text classification.

---

## License

This project is open-source and available under the **MIT License**.
