# 🧠 Scalable NLP Sentiment Analysis & Review Clustering with PySpark

Completed as part of the **STAT483: Big Data Analytics** course at the University of Bahrain.

This project applies distributed computing, Natural Language Processing (NLP), supervised machine learning, and unsupervised clustering techniques to analyze millions of Amazon product reviews using Apache Spark and PySpark.

---

# 👥 Team

- **Ebrahim Juma Shakak Alsawan**
- **Sadeq Jaafar Ali Deyab**
- **Salman Wael Salman**
- **Ali Sameer Ali Alzenji**
- **Ahmed Sadiq Ali Alsameea**

---

# 🎯 Project Objective

The objective of this project was to extract meaningful insights from large-scale Amazon review data using distributed machine learning and NLP techniques.

The project focused on:
- Sentiment classification
- Review clustering
- Text preprocessing
- Large-scale feature engineering
- Distributed NLP pipelines
- Pattern discovery from customer reviews

The analysis combined supervised and unsupervised machine learning workflows to evaluate customer sentiment and uncover hidden product-review structures at scale.

---

# 📊 Dataset Overview

## Dataset Source
Amazon Reviews Dataset from Kaggle

## Dataset Link
https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews

## Dataset Scale
- **Total Reviews:** 3.8 Million
- **Training Dataset:** 3.6 Million Reviews
- **Test Dataset:** 400,000 Reviews

## Sentiment Structure
- Ratings 1–2 → Negative Reviews
- Ratings 4–5 → Positive Reviews

The dataset provides large-scale customer review text data suitable for distributed Natural Language Processing and machine learning analysis.

---

# 📖 Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `reviewerID` | str | Unique identifier for the reviewer/customer |
| `asin` | str | Amazon Standard Identification Number (ASIN) representing the product |
| `reviewerName` | str | Name or username of the reviewer |
| `helpful` | array[int] | Helpfulness rating represented as `[helpful_votes, total_votes]` |
| `reviewText` | str | Full text review written by the customer |
| `overall` | int | Product rating score given by the customer (1–5 stars) |
| `summary` | str | Short summary/title of the review |
| `unixReviewTime` | int | Review timestamp stored in Unix time format |
| `reviewTime` | str | Human-readable review date |

---

# ⚙️ Engineered NLP Features

| Feature | Description |
|---|---|
| `body_tokens` | Tokenized review text |
| `body_clean` | Review text after stop-word removal |
| `raw_features` | CountVectorizer numerical representation |
| `features` | TF-IDF feature vectors used for machine learning |
| `prediction` | Predicted sentiment class |
| `cluster` | Assigned K-means cluster label |

---

# 🧠 Business Problem

Large e-commerce platforms generate millions of customer reviews daily, making manual review analysis impossible at scale.

This project aimed to solve several analytical challenges:

- Automatically classify customer sentiment
- Identify hidden product-review groupings
- Discover review language patterns
- Build scalable NLP pipelines using distributed computing
- Analyze customer satisfaction trends efficiently

The project demonstrates how big data technologies can transform unstructured text into actionable business intelligence.

---

# ⚙️ Distributed Computing Highlights

- Processed 3.8 million Amazon reviews using Apache Spark
- Leveraged distributed PySpark pipelines for scalable NLP preprocessing
- Applied Spark MLlib for machine learning at scale
- Implemented memory-efficient TF-IDF vectorization workflows
- Used parallelized transformations and distributed computations

---

# 🧹 NLP Preprocessing Workflow

The project implemented a complete distributed NLP preprocessing pipeline.

## Preprocessing Steps

```python
✔ Lowercase text normalization
✔ Regex tokenization
✔ Stop-word removal
✔ CountVectorizer transformation
✔ TF-IDF feature scaling
✔ Sparse vector generation
✔ Distributed feature engineering
```

---

# 🛠 Technology Stack

## Big Data Processing
- Apache Spark
- PySpark
- Spark MLlib

## Machine Learning
- Logistic Regression
- K-Means Clustering
- TF-IDF Vectorization
- Bag-of-Words Modeling

## NLP Processing
- RegexTokenizer
- StopWordsRemover
- CountVectorizer
- IDF Transformation

## Visualization & Analysis
- Pandas
- Matplotlib
- Seaborn
- WordCloud

---

# 📈 Analysis Workflow

## 1. Data Preprocessing

```python
# Lowercase normalization
train_df = train_df.withColumn("title_lower", lower(col("title")))
                   .withColumn("body_lower", lower(col("body")))

# Tokenization and stop-word removal
tokenizer = RegexTokenizer(inputCol="body_lower", outputCol="body_tokens")
stop_remover = StopWordsRemover(inputCol="body_tokens", outputCol="body_clean")
```

---

## 2. Feature Engineering & Sentiment Classification

```python
# TF-IDF Pipeline
cv = CountVectorizer(inputCol="body_clean", outputCol="raw_features", vocabSize=5000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[cv, idf, lr])
model = pipeline.fit(train_df)
```

---

## 3. Clustering Analysis

```python
# K-Means clustering
kmeans = KMeans(
    k=12,
    initMode="k-means||",
    maxIter=20,
    featuresCol="body_bow"
)

clustering_model = kmeans.fit(vectorized_data)
```

---

# 📊 Sentiment Classification Results

| Metric | Score |
|--------|-------|
| Accuracy | 85.57% |
| Precision | 84.89% |
| Recall | 86.54% |
| F1-Score | 85.71% |
| False Positive Rate | 15.40% |

---

# 📌 Classification Insights

- Logistic Regression achieved strong sentiment classification performance on millions of reviews.
- Positive reviews frequently used emotional and quality-focused language.
- Negative reviews focused more heavily on product issues and transactional concerns.
- TF-IDF feature engineering significantly improved sentiment detection quality.

---

# 📷 Classification Visuals

<p align="center">
  <img src="results/confusion_matrix.png" width="45%">
  <img src="results/word_clouds.png" width="45%">
</p>

---

# 📈 Clustering Analysis

## Clustering Results

- Optimal clusters identified using the elbow method: **K = 12**
- Reviews naturally grouped by product category and vocabulary patterns.
- K-means clustering successfully separated major product themes.

---

# 📌 Clustering Insights

## Product-Based Clusters Identified

| Cluster | Dominant Vocabulary |
|---|---|
| Cluster 2 | book, read, good |
| Cluster 4 | album, cd, songs, music |
| Cluster 7 | movie, film, good |

### Key Findings
- Product-specific vocabulary strongly influenced clustering behavior.
- Review similarity naturally formed category-based groupings.
- Cluster distribution revealed both generalized and niche review categories.

---

# 📷 Clustering Visuals

<p align="center">
  <img src="results/elbow_method.png" width="45%">
  <img src="results/cluster_distribution.png" width="45%">
</p>

---

# 🔍 Key Insights

## The Language of Satisfaction

Satisfied customers commonly used:
- emotional language
- evaluative descriptors
- quality-oriented vocabulary

Examples:
- great
- amazing
- perfect
- excellent

---

## The Language of Dissatisfaction

Negative reviews focused more heavily on:
- transactional concerns
- product defects
- pricing/value complaints

Examples:
- money
- bought
- problem
- return

---

## Scalability Achievement

This project successfully demonstrated:
- distributed machine learning
- scalable NLP processing
- large-scale text analytics
- memory-efficient Spark workflows

on a dataset containing millions of reviews.

---

# 📁 Project Structure

```bash
.
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── src/
│   ├── preprocessing.py
│   ├── classification.py
│   ├── clustering.py
│   └── evaluation.py
├── results/
│   ├── confusion_matrix.png
│   ├── word_clouds.png
│   ├── cluster_distribution.png
│   └── elbow_method.png
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Install Dependencies

```bash
pip install pyspark pandas matplotlib seaborn wordcloud scikit-learn
```

---

## Run the Analysis

```bash
# Preprocessing
python src/preprocessing.py

# Classification
python src/classification.py

# Clustering
python src/clustering.py

# Evaluation
python src/evaluation.py
```

---

# 🔮 Future Enhancements

- Deep Learning integration using LSTM/BERT
- Real-time streaming analysis using Kafka
- Multi-class sentiment classification
- Aspect-based sentiment analysis
- Advanced transformer-based NLP pipelines

---

# 📊 Key Skills Demonstrated

- Big Data Analytics
- Distributed Computing
- Natural Language Processing (NLP)
- Machine Learning
- Sentiment Classification
- K-Means Clustering
- PySpark
- Spark MLlib
- Feature Engineering
- TF-IDF Vectorization
- Text Mining
- Scalable Data Processing

---

# 🎓 Course Information

**Course:** STAT483 – Big Data Analytics  
**Institution:** University of Bahrain  
**Project Type:** Big Data NLP & Machine Learning Project

---

# 👤 Authors

### Ebrahim Juma Shakak Alsawan
- LinkedIn: https://www.linkedin.com/in/ebrahim-alsawan-a6977a2b9/

### Team Members
- Sadeq Jaafar Ali Deyab
- Salman Wael Salman
- Ali Sameer Ali Alzenji
- Ahmed Sadiq Ali Alsameea
