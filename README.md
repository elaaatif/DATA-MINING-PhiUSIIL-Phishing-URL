# PhiUSIIL Phishing URL Dataset Analysis

## Introduction

This repository contains a data mining project focused on analyzing the PhiUSIIL Phishing URL Dataset. The dataset comprises a substantial collection of legitimate and phishing URLs, extracted from both webpage source code and URL features.

## Dataset Description

The PhiUSIIL Phishing URL Dataset consists of:

- **Legitimate URLs**: 134,850 instances
- **Phishing URLs**: 100,945 instances
- **Features**: Extracted from the source code of the webpage and URL. Includes features such as CharContinuationRate, URLTitleMatchScore, URLCharProb, and TLDLegitimateProb.

For more details, refer to the [dataset documentation](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

## Analysis Overview

The analysis workflow consists of the following steps:

1. **Data Cleaning and Preprocessing**: 
   - Reading the dataset
   - Dropping irrelevant columns (e.g., 'FILENAME', 'URL', 'Domain', 'Title')
   - Handling categorical variables
   - Removing highly correlated features
   - Profiling the dataset for exploratory analysis ([Profiling Report](https://elaaatif.github.io/DATA-MINING-PhiUSIIL-Phishing-URL/))

2. **Random Forest Algorithm**: 
   - Training a Random Forest classifier
   - Evaluating performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve
   - Visualizing a decision tree from the forest and feature importances

3. **Comparison with Other Algorithms**:
   - Implementing K-Nearest Neighbors (KNN), Naive Bayes, and Decision Trees
   - Training classifiers, evaluating performance, and visualizing results
   - Comparing accuracy, precision, recall, and F1-score across algorithms

4. **Additional Analysis**:
   - Conducting a Chi-square test for feature importance assessment

## Usage

To replicate the analysis, follow these steps:

1. Clone the repository:
      ```bash
   git clone https://github.com/elaaatif/DATA-MINING-PhiUSIIL-Phishing-URL
   ```
2. Install the required packages:
     ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```
3. Run the Jupyter Notebook :
      ```bash
   python Phishing URL (Website) - DATA MINING PROJECT.IPYNB
   ```
This will execute the data preprocessing steps, train and evaluate the Random Forest classifier, and compare the results with other algorithms.

## Interpretation of Results

The analysis demonstrates the effectiveness of different machine learning algorithms in classifying legitimate and phishing URLs. Here are some key findings:

- **Random Forest**: Achieved high accuracy and balanced precision and recall values.
- **K-Nearest Neighbors (KNN)**: Also performed well, with high accuracy and balanced precision and recall values.
- **Naive Bayes**: Demonstrated strong performance, albeit with slightly lower accuracy compared to Random Forest and KNN.
- **Decision Trees**: Achieved perfect accuracy, precision, recall, and F1-score, indicating optimal performance on the dataset.

The comparison highlights the strengths and weaknesses of each algorithm, providing valuable insights for selecting the most suitable approach for phishing URL detection tasks.
##
The following Image show the F1-Score ,Precision & Recall for the K-Nearest Neighbors (KNN), Naive Bayes, and Decision Trees in comparison with Random Forest 

![image](https://github.com/elaaatif/DATA-MINING-PhiUSIIL-Phishing-URL/assets/122917261/4f60a071-4e6b-487f-ac8e-85d8c195f58a)

