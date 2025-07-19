# 🛒 Amazon Product Reviews Analysis & Classification

This project was completed as part of the **STAT483: Big Data Analytics** course, employing PySpark and machine learning techniques to analyze Amazon product reviews for sentiment classification and content clustering.

## 👥 Team

- **Ebrahim Juma Shakak Alsawan**
- **Sadeq Jaafar Ali Deyab**
- **Salman Wael Salman**
- **Ali Sameer Ali Alzenji**
- **Ahmed Sadiq Ali Alsameea**

## 🎯 Objective

To extract meaningful insights from Amazon product reviews using distributed computing and machine learning techniques, implementing both unsupervised clustering and supervised classification to analyze textual review data at scale.

## 📊 Dataset Overview

**Source**: [Amazon Reviews Polarity Dataset](https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews)

**Size**: 3.8 million reviews total
- **Training Data**: 3.6 million samples (1.8M positive + 1.8M negative)
- **Test Data**: 400,000 samples (200K per class)
- **Rating Scale**: 1-2 (negative) and 4-5 (positive) star reviews
- **Perfect Class Balance**: Equal distribution eliminates bias concerns

## 🔍 Key Findings

### Sentiment Classification Results
- **Accuracy**: 85.57%
- **F1-Score**: 85.71%
- **Precision**: 84.89%
- **Recall**: 86.54%
- **False Positive Rate**: 15.40%

### Word Pattern Analysis
**Most Common Words Overall**: "book", "one", "like", "good", "great"

**Positive Reviews Vocabulary**:
- Emotional language: "great", "love", "best", "wonderful"
- Quality descriptors: "excellent", "amazing", "perfect"

**Negative Reviews Vocabulary**:
- Transactional focus: "money", "bought", "product", "better"
- Practical concerns: "time", "get", "would"

### Clustering Insights
- **Optimal Clusters**: K=12 using elbow method
- **Product-Based Grouping**: Reviews naturally clustered by product categories
  - **Cluster 2**: Books ("book", "read", "good")
  - **Cluster 4**: Music ("album", "cd", "songs", "music")  
  - **Cluster 7**: Movies ("movie", "film", "good")
- **Highly Skewed Distribution**: Majority in general cluster, specialized clusters for niche categories

## 🛠 Technology Stack

### Big Data Processing
- **Apache Spark**: Distributed computing framework
- **PySpark**: Python API for Spark operations
- **Spark MLlib**: Machine learning algorithms at scale

### Machine Learning Techniques
- **Supervised Learning**: Logistic Regression for sentiment classification
- **Unsupervised Learning**: K-means clustering for content grouping
- **Feature Engineering**: TF-IDF vectorization, Bag-of-Words

### Text Processing Pipeline
- **Tokenization**: RegexTokenizer for word extraction
- **Stop Words Removal**: Common word filtering
- **Count Vectorization**: Text-to-numeric conversion (vocab size: 5000)
- **IDF Transformation**: Term importance scaling

## 📈 Analysis Workflow

### 1. Data Preprocessing
```python
# Lowercase conversion for consistent tokenization
train_df = train_df.withColumn("title_lower", lower(col("title")))
                   .withColumn("body_lower", lower(col("body")))

# Tokenization and stop word removal
tokenizer = RegexTokenizer(inputCol="body_lower", outputCol="body_tokens")
stop_remover = StopWordsRemover(inputCol="body_tokens", outputCol="body_clean")
```

### 2. Feature Extraction & Classification
```python
# TF-IDF Pipeline for sentiment classification
cv = CountVectorizer(inputCol="body_clean", outputCol="raw_features", vocabSize=5000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[cv, idf, lr])
model = pipeline.fit(train_df)
```

### 3. Clustering Analysis
```python
# K-means clustering with elbow method optimization
kmeans = KMeans(k=12, initMode="k-means||", maxIter=20, featuresCol="body_bow")
clustering_model = kmeans.fit(vectorized_data)
```

## 🔗 Key Insights

### The Language of Satisfaction
Our analysis reveals distinct vocabulary patterns:
- **Satisfied customers** use emotional, evaluative language focusing on quality and experience
- **Dissatisfied customers** employ transactional language emphasizing product attributes and value concerns

### Product Category Recognition
Unsupervised clustering successfully identified product categories without labels:
- Reviews naturally segregated into books, movies, music, and general product clusters
- Product-specific vocabulary serves as strong similarity indicator

### Scalability Achievement
Successfully processed 3.8 million reviews using distributed computing:
- Efficient memory management and parallelization
- Real-world big data application demonstration

## 📁 Project Structure

```
├── data/
│   ├── train.csv                 # Training dataset (3.6M reviews)
│   └── test.csv                  # Test dataset (400K reviews)
├── notebooks/
│   ├── data_exploration.ipynb    # EDA and visualization
│   ├── preprocessing.ipynb       # Data cleaning pipeline  
│   ├── classification.ipynb      # Supervised learning analysis
│   └── clustering.ipynb          # Unsupervised learning analysis
├── src/
│   ├── preprocessing.py          # Text preprocessing utilities
│   ├── classification.py         # Logistic regression implementation
│   ├── clustering.py             # K-means clustering implementation
│   └── evaluation.py             # Model evaluation metrics
├── results/
│   ├── confusion_matrix.png      # Classification performance
│   ├── word_clouds.png          # Sentiment-specific vocabularies
│   ├── cluster_distribution.png  # Cluster size visualization
│   └── elbow_method.png         # Optimal k determination
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install pyspark pandas matplotlib seaborn wordcloud scikit-learn
```

### Running the Analysis
```bash
# 1. Start Spark session and load data
python src/preprocessing.py

# 2. Run sentiment classification
python src/classification.py

# 3. Perform clustering analysis  
python src/clustering.py

# 4. Generate evaluation metrics
python src/evaluation.py
```

## 📊 Performance Metrics

### Classification Performance
| Metric | Score |
|--------|-------|
| Accuracy | 85.57% |
| Precision | 84.89% |
| Recall | 86.54% |
| F1-Score | 85.71% |

### Confusion Matrix Results
- **True Negatives**: 169,192
- **True Positives**: 173,085  
- **False Positives**: 30,802
- **False Negatives**: 26,912

## 🔮 Future Enhancements

- **Deep Learning Integration**: LSTM/BERT models for improved accuracy
- **Real-time Processing**: Kafka streaming for live review analysis
- **Multi-class Classification**: Fine-grained sentiment categories (1-5 stars)
- **Aspect-based Analysis**: Product feature-specific sentiment extraction

## 📌 Acknowledgements

Special thanks to the course instructor for guidance in big data analytics applications and distributed computing concepts.

*Submitted as part of STAT483: Big Data Analytics*
---

**Data Source**: [Amazon Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews)
