# Contributor Verification: Binary Classification via Robust Deep Neural Networks

## 1. Executive Summary
This project addresses the **Contributor Verification** challenge, a supervised binary classification task focused on determining the legitimacy of article contributors. By integrating textual embeddings with structured metadata, we developed a system to predict whether a candidate is a genuine contributor to a specific work. 

The final solution, a highly regularized Deep Neural Network (DNN), achieved a **Private Leaderboard Rank of 11th out of 25**, demonstrating superior generalization capabilities on unseen data.

---

## 2. Problem Definition & Context
In the digital publishing landscape, maintaining archival integrity and preventing attribution fraud are critical. Contributor verification is the process of confirming authorship using writing signals and contextual data.

### Project Goals:
* **Binary Classification:** Classify pairs of (Article, Candidate) as `1` (Verified Contributor) or `0` (Non-contributor).
* **Robustness:** Minimize the "Generalization Gap" between training performance and real-world test performance.
* **Feature Fusion:** Effectively combine sparse categorical metadata with high-dimensional textual embeddings.



---

## 3. Dataset & Feature Engineering
The model was trained on a dataset of **25,000 articles** with a separate **2,000-article test set**.

* **Textual Data:** Processed as tokenized sequences and converted into dense embeddings.
* **Metadata:** Features such as `article_venue` and `article_year` were treated as categorical inputs.
* **Balanced Sampling:** To ensure the model learned stable contribution patterns, a balanced dataset of **88,340 samples** was generated for the final training phase, preventing bias toward any single class.

---

## 4. Architectural Evolution
We systematically explored three different modeling strategies to identify the most stable solution:

### A. Baseline Experiment (`DNN_Baseline_Experiment.ipynb`)
A standard feed-forward neural network was used to establish a performance floor. This helped identify the primary signals in the categorical and textual data.

### B. Ensemble Approach (`Ensemble_DNN_XGBoost.ipynb`)
We attempted to blend the deep feature extraction of a DNN with the gradient-boosted decision trees of **XGBoost**. While the ensemble captured diverse patterns, the simple probability averaging scheme proved less effective than the specialized regularization of a standalone deep network.

### C. Final Robust DNN (`Final_Model_Robust_DNN.ipynb`)
The selected model utilized the **Keras Functional API** to create a multi-input architecture.
* **Architecture:** Integrated **Global Average Pooling** for text embeddings to extract global context.
* **Regularization:** The core of the 11th-place success was the implementation of **L2 weight regularization** and specific **Dropout (0.3/0.2)** layers.
* **Outcome:** This model prioritized stability, resulting in only a **-2.43%** drop from public to private scores, whereas more complex models suffered from severe overfitting.



---

## 5. Performance Results
| Metric | Score |
| :--- | :--- |
| **Public Leaderboard Accuracy** | 0.87500 |
| **Private Leaderboard Accuracy** | **0.86071** |
| **Final Competition Rank** | **11th / 25** |

---

## 6. Repository Structure
* `notebooks/`: Contains the complete experimental pipeline (Baseline, Ensemble, and Final Robust Model).
* `reports/`: Detailed technical project report covering ablation studies and methodology.
* `outputs/`: Final `submission.csv` containing the model's predictions.
* `requirements.txt`: Comprehensive list of Python dependencies.

---

## 7. Installation & Usage

### Setup Environment
This project requires **Python 3.8+**. It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

