# Text-Classification-Using-TensorFlow
CNN-based sentiment analysis and text classification model using TensorFlow and the IMDB movie review dataset.

# 🎬 IMDB Sentiment Classification with CNN

A text classification model that predicts whether a movie review is **positive or negative**, trained on the IMDB dataset using a 1D Convolutional Neural Network in TensorFlow/Keras.

---

## 📌 Overview

The model reads a movie review (up to 200 words), converts it to word embeddings, and runs it through a Conv1D layer to extract local text patterns. A single sigmoid output gives a score between 0 (negative) and 1 (positive).

It trains in under 2 minutes on a CPU and hits ~88% test accuracy in 5 epochs.

---

## 🗂️ Project Structure

```
Text_classification.py      # All-in-one script: data loading, model, training, evaluation, plots
```

---

## 🧠 Model Architecture

```
Input (sequence of 200 integer token IDs)
    │
    ▼
Embedding(vocab=10000, dim=32)     # Maps each word to a 32-dim vector
    │
    ▼
Conv1D(32 filters, kernel=3, ReLU) # Detects 3-gram patterns across the sequence
    │
    ▼
GlobalMaxPooling1D()               # Takes the strongest signal from each filter
    │
    ▼
Dense(10, ReLU)                    # Small fully connected layer
    │
    ▼
Dense(1, Sigmoid)                  # Output: probability of positive sentiment
```

---

## 📦 Requirements

### Python Version
- Python 3.8+

### Dependencies

```bash
pip install tensorflow matplotlib
```

| Package       | Purpose                              |
|---------------|--------------------------------------|
| `tensorflow`  | Model building, training, evaluation |
| `matplotlib`  | Plotting accuracy and loss curves    |

The IMDB dataset downloads automatically via `tf.keras.datasets.imdb` on first run (~17 MB).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/text-classification-cnn.git
cd Text-Classification-Using-TensorFlow
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt`:
> ```
> tensorflow
> matplotlib
> ```

### 3. Run

```bash
python Text_classification.py
```

The script will:
- Download and pad the IMDB dataset
- Train the CNN for 5 epochs
- Print test loss and accuracy
- Plot training vs. validation accuracy
- Plot training vs. validation loss

---

## 📊 Dataset

- **Source**: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) via `tf.keras.datasets.imdb`
- 25,000 training reviews, 25,000 test reviews
- Labels: `0` = negative, `1` = positive
- **Preprocessing**:
  - Vocabulary capped at the top 10,000 most frequent words
  - Sequences padded (or truncated) to exactly 200 tokens with pre-padding

---

## ⚙️ Configuration

| Parameter      | Value   | Description                              |
|----------------|---------|------------------------------------------|
| `vocab_size`   | `10000` | Top N words kept from the corpus         |
| `max_length`   | `200`   | Token sequence length after padding      |
| `embedding_dim`| `32`    | Word vector size                         |
| `filters`      | `32`    | Number of Conv1D filters                 |
| `kernel_size`  | `3`     | N-gram window the Conv1D scans           |
| `epochs`       | `5`     | Training epochs                          |
| `batch_size`   | `128`   | Samples per gradient update              |
| `val_split`    | `0.2`   | 20% of training data used for validation |
| Optimizer      | `Adam`  | Default learning rate (0.001)            |
| Loss           | `binary_crossentropy` | Standard for binary classification |

---

## 📈 Expected Results

| Metric              | Approximate Value |
|---------------------|-------------------|
| Test Accuracy       | ~87–89%           |
| Test Loss           | ~0.28–0.33        |

Results vary slightly between runs due to random weight initialization and shuffling.

---

## 📉 Training Plots

Two plots appear after training:

1. **Model Accuracy** — Train and validation accuracy per epoch. A widening gap between the two signals overfitting.
2. **Model Loss** — Train and validation binary cross-entropy per epoch.

---

## 🔧 Extending the Project

A few directions worth exploring:

- **LSTM or GRU** — Replace `Conv1D + GlobalMaxPooling` with a recurrent layer to capture longer-range dependencies in the review text.
- **Pre-trained embeddings** — Swap the random `Embedding` layer for GloVe or FastText vectors to start with richer word representations.
- **Larger vocab / longer sequences** — Bump `vocab_size` to 20,000 or `max_length` to 500 if reviews are being truncated heavily.
- **Dropout regularization** — Add `layers.Dropout(0.5)` before the final Dense layer to reduce overfitting.
- **Custom review input** — Run your own text through the model after training by encoding it with the IMDB word index:
  ```python
  word_index = imdb.get_word_index()
  # encode your review, pad to max_length, then call model.predict()
  ```

---

## ☁️ Running on Google Colab

Upload the file and run:

```python
!python Text_classification.py
```

Or paste directly into a notebook cell. GPU runtime cuts training time from ~2 minutes to under 30 seconds.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push and open a Pull Request

---

## 🙏 Acknowledgements

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) — Andrew Maas et al., Stanford AI Lab
- [TensorFlow / Keras](https://www.tensorflow.org/) — Model framework

