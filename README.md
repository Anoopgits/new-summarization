New Summarization using LSTM and T5 Transformer

This project focuses on **new summarization**— the process of automatically generating concise and meaningful summaries from long pieces of text.
It demonstrates **two approaches**:

1. **Seq2Seq model (LSTM-based)**
2. **Transformer-based model (T5 Small)**

The goal is to compare the performance of a traditional sequence-to-sequence LSTM model and a modern Transformer model using TensorFlow.

---

##  Project Overview

###  Extractive vs. Abstractive Summarization

* **Extractive** → Selects key sentences from the text.
* **Abstractive** → Generates new sentences to summarize the context.

This project focuses on **Abstractive Summarization**.

---

##  Workflow Steps

1. **Dataset Gathering**

   * Used `news_summary.csv` and `news_summary_more.csv` from Kaggle.

2. **Data Preprocessing**

   * Lowercasing
   * Removing punctuation, URLs, and special characters
   * Tokenization
   * Adding special tokens (`_START_` and `_END_`)

3. **Tokenization & Padding**

   * Used Keras `Tokenizer` to convert text and summary into sequences.
   * Applied padding to ensure uniform input lengths.

4. **Model Building**

   * **Approach 1:** LSTM-based Seq2Seq Encoder-Decoder Model
   * **Approach 2:** Transformer-based T5 model (`t5-small`) fine-tuned on the dataset.

5. **Training**

   * Used early stopping to prevent overfitting.
   * Partial fine-tuning of the T5 model to reduce GPU usage.

6. **Evaluation**

   * Compared training and validation losses.
   * Visualized performance with loss curves.

7. **Inference**

   * Provided a function to generate summaries from any input text.
   * Model outputs a concise and meaningful summary.

8. **Model Saving**

   * Saved trained models in TensorFlow format (`.h5`) and Hugging Face format (`.bin` and `config.json`).

---

##  Technologies Used

| Category   | Tools                                            |
| ---------- | ------------------------------------------------ |
| Language   | Python                                           |
| Frameworks | TensorFlow, Keras, Hugging Face Transformers     |
| Libraries  | NumPy, Pandas, Matplotlib, Scikit-learn, Seaborn |
| Models     | LSTM Encoder-Decoder, T5-Small                   |
| Dataset    | Kaggle News Summary Dataset                      |

---

##  Installation

```bash
# Clone this repository
git clone https://github.com/your-username/text-summarization.git
cd text-summarization

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Requirements

Add these to your `requirements.txt` file:

```
tensorflow==2.15.0
transformers==4.33.2
datasets==2.14.5
evaluate
rouge-score
numpy
pandas
matplotlib
scikit-learn
seaborn
fsspec==2025.3.0
gcsfs==2025.3.0
```

---

##  Model Architecture

### ** LSTM Model**

* Encoder: Embedding + LSTM
* Decoder: Embedding + LSTM + Dense (Softmax)
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy

### ** T5 Transformer**

* Pretrained Model: `t5-small`
* Task Prefix: `"summarize:"`
* Fine-tuning on cleaned dataset
* Supports both **full** and **partial** fine-tuning

---


##  Results & Observations

| Model                  | Training Speed | Accuracy  | GPU Usage |
| ---------------------- | -------------- | --------- | --------- |
| LSTM Seq2Seq           | Slow           | Good      | High      |
| T5 (Full Fine-Tune)    | Moderate       | Excellent | High      |
| T5 (Partial Fine-Tune) | Fast           | Very Good | Low       |

---

##  Future Improvements

* Add ROUGE / BLEU metric evaluation
* Implement **live news summarization** using APIs
* Support **multilingual summarization**



