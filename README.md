# Churn-Customer-Prediction
# 👤 About Me 

-Name :Melody Mousavi 

-Company:SedraPro

---

# 📁 Project Description 

This  repository explores customer churn prediction using a churn_modelling.csv bank dataset, employing both traditional machine learning algorithms such as Logistic Regression and Decision Tree and Xgboost and Deep Neural Network(DNN). The main objective of the project is to compare and analyze the effectiveness of these models for instructional purposes.

# 🛠️ Tools & Technologies 

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib, Seaborn

---

# 🚩 Project Workflow

General Steps of the Project:
1.	Importing and Initial Exploration of the Dataset     
o	a. Importing the initial data from the CSV file  
o	b. Exploring the general information of the DataFrame and identifying missing values using tables and Heatmap visualization
2.	Outlier Detection and Removal
o	a. Identifying numerical and non-numerical columns
o	b. Calculating the Interquartile Range (IQR) to detect outliers
o	c. Removing outliers using the Z-score method
o	d. Displaying and comparing statistical information of the original and cleaned data
4.	Demographic Feature Analysis: Gender and Geographical Location
5.	Statistical Analysis
o	a. Investigating the relationship between features such as gender and geographical location with churn rate
o	b. Plotting frequency charts for categorical columns
6.	Data Preprocessing and Preparation
o	a. Converting non-numerical variables into model-usable data (One-Hot Encoding)
o	b. Normalizing data (using Min-Max and Standardization/Z-score) for better modeling
7.	Comparing Feature Distributions Before and After Normalization
o	a. Plotting histograms to show the distribution of numerical features before and after normalization
8.	Splitting the Data for Training and Testing
o	a. Dividing the data into training (80%) and testing (20%) sets
9.	Training and Evaluating Classic Machine Learning Models
o	a. Logistic Regression
o	b. Decision Tree
o	c. XGBoost Classifier
o	d. Gradient Boosting Classifier
Each of these models will be trained on the training set and their results will be evaluated on the test set using metrics such as Accuracy, Confusion Matrix, and Classification Report.
10.	Comparing Model Performances Based on Accuracy and Evaluation Metrics
11.	Design and Training of a Multi-Layer Neural Network (MLP)
•	a. With the aim of increasing prediction accuracy compared to traditional models
12.	Analysis of Results and Feature Interpretation
•	a. Investigating which features have the most impact on customer churn
•	b. Comparing the accuracy of the models and selecting the optimal model
13.	Complete Documentation of the Project
    


