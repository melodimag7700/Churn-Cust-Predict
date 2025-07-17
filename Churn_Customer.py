
# Import necessary libraries
 # Import pandas library for data handling | کتابخانه پانداس برای کار با داده‌ها
import pandas as pd
# Read the CSV file into a DataFrame | خواندن فایل CSV و ساخت دیتا فریم
df=pd.read_csv('D:\\churnPrediction\\Churn_Modelling.csv')
 # Print first 5 rows to check data | نمایش ۵ سطر اول برای بررسی داده
print(df.head())

##INFORMATION
# Show information about columns and data types | اطلاعات کلی درباره دیتافریم و نوع داده‌ها را نمایش می‌دهد
df.info()

#-----------------------------------------------------------
##Missing Values
#----------------------------------------------------------
#showing missing values
# Check for missing values (True where missing) | بررسی داده‌های گمشده (در صورت وجود True)
missing_values = df.isnull()
print("Missing Values:\n",missing_values)
#Import nessary libraries for visualization
# Import seaborn for visualization | کتابخانه سیبورن برای مصورسازی
import seaborn as sns
# Import matplotlib for plotting | کتابخانه matplotlib برای رسم نمودارس
import matplotlib.pyplot as plt
# Set size of the heatmap | تنظیم سایز تصویر هیت‌مپ
plt.figure(figsize=(15,7))
# Draw heatmap for missing values | رسم هیت‌مپ برای داده‌های گمشده
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
plt.title('Missing Values')
plt.show()

#----------------------------------------------------------
##OutLiers
#----------------------------------------------------------
#Import nessesary libraries for Outliers
import numpy as np
#Check which columns are numeric
# Check if each column is numeric | بررسی عددی بودن هر ستون
isnumeric=df.apply(lambda x:pd.api.types.is_number(x))
print("Numeric Columns:",isnumeric)
# Check if all columns are numeric | بررسی این‌که آیا همه ستون‌ها عددی هستند یا نه
allnum=isnumeric.all()
if allnum:
    print("all columns are numerical")
else:
    #if any column is not numerical
    print("There is non-numerical column/columns")
    ## Find the non-numeric columns | پیدا کردن نام ستون‌های غیرعددی
    non_num=df.columns[~isnumeric]
#LList of non-numerical columns
    print("nun-numerical columns:",non_num )

#----------------------------------------------------------
#Analysis based on geographic region and gender
#----------------------------------------------------------
#import necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Descriptive statistics for gender
# List of columns to analyze | لیست ستون‌های مورد بررسی
columnsName=['Geography', 'Gender']
# Group by gender, calculate mean and count | گروه‌بندی بر اساس جنسیت و محاسبه میانگین و تعداد
genderSts = df.groupby('Gender')['Exited'].agg(['mean','count']).reset_index()
# Rename columns | تغییر نام ستون‌ها
genderSts.columns=['Gender','Exit Rate','Count']
print("statistics gender  \n",genderSts)
# Group by geography, calculate mean and count | گروه‌بندی بر اساس کشور و محاسبه میانگین و تعداد
geoSts=df.groupby('Geography')['Exited'].agg(['mean','count']).reset_index()
# Rename columns | تغییر نام ستون‌ها
geoSts.columns =['Geography','Exit Rate','Count']
print("statistics gerogerahy \n",geoSts)

 # Set figure size | تنظیم اندازه نمودار
plt.figure(figsize=(15,7))

#bar chart
# Barplot for gender exit rate | نمودار ستونی نرخ ریزش بر اساس جنسیت
sns.barplot(x='Gender', y='Exit Rate', data=genderSts)
plt.title('Exit Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Mean Exit Rate')
plt.show()

# Barplot for geography exit rate | نمودار ستونی نرخ ریزش بر اساس کشور
sns.barplot(x='Geography', y='Exit Rate', data=geoSts)
plt.title('Exit Rate by Geography')
plt.xlabel('Geography')
plt.ylabel('Mean Exit Rate')
plt.show()

# ----------------------------------------------------------
#delete features
import pandas as pd
df=pd.read_csv('D:\\churnPrediction\\Churn_Modelling.csv')
# Drop column 'Surname' as it's not needed | حذف ستون Surname چون در مدل‌سازی کاربرد ندارد
df.drop(columns=['Surname'], inplace=True)
df.to_csv('D:\\churnPrediction\\Churn_Modelling_upd.csv')

print(df)

#-----------------------------------------------------------
#Calculate IQR
import pandas as pd
# Select only numerical columns | فقط ستون‌های عددی را انتخاب می‌کند
num_columns = df.select_dtypes(include=['float64', 'int64'])
# Dictionary to store outlier indices | لیست ایندکس سطرهای پرت برای هر ستون
outlier={}
for cln in num_columns:
    # Calculate Q1 (25th percentile) | محاسبه چارک اول
    Q1=df[cln].quantile(0.25)
    # Calculate Q3 (75th percentile) | محاسبه چارک سوم
    Q3=df[cln].quantile(0.75)
    # Calculate IQR | محاسبه فاصله بین چارک‌ها
    IQR = Q3-Q1

 #define lower and upper bounds
 # Lower bound for outliers | حد پایین برای پرت‌ها
lower_b = Q1 - 1.5 * IQR
# Upper bound for outliers | حد بالا برای پرت‌ها
upper_b = Q3 -1.5 * IQR
#find outlier for each column

# Find outlier indices | پیدا کردن ایندکس پرت‌ها
outlier_idx = df[(df[cln] < lower_b) | (df[cln] > upper_b)].index
outlier[cln] = outlier_idx.tolist()

for col,num in outlier.items():
    print(f"Outliers in {col}: {num}")

#-----------------------------------------------------------
#evaluation of outlier
import pandas as pd
num_col = df.select_dtypes(include=['float64', 'int64']).columns
#calculate mean
mean_val = df[num_col].mean()
std_val = df[num_col].std()
summary = pd.DataFrame({'Mean': mean_val, 'Std': std_val})
print(summary)

#-----------------------------------------------------------
#Remove outlier and calculate z-score
import pandas as pd
import numpy as np
# Import scipy for statistical functions | وارد کردن سای‌پای برای توابع آماری
from scipy import stats
# Get numerical columns | گرفتن ستون‌های عددی
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 0:
    # Compute absolute Z-score for each value | محاسبه Z-score برای هر مقدار
    z_scores=np.abs(stats.zscore(df[num_cols]))

    # Define Z-score threshold for outlier | حد آستانه Z-score برای پرت (معمولاً ۳)
    threshold = 3
    # Find where Z-score exceeds threshold | پیدا کردن داده‌های پرت 
    outlier = z_scores >threshold
    # Identify rows with outliers | سطرهایی که حداقل یک ستون عددی‌شان پرت است
    rows_ = np.any(outlier,axis=1)
    # Create new DataFrame without outlier rows | حذف داده‌های پرت و ذخیره در دیتافریم جدید
    df_cleaned = df[~rows_]
    print("Number of rows removed:", np.sum(rows_))
else:
     print("not available numerical columns")

#------------------------------------------------------
#calculate stast
def calculate_stats(df):
  
    num_col = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_col)>0:
        mean_val=df[num_col].mean()
        sts_val =df[num_col].std()
        stats_data = {
                'Mean': mean_val,
                'Standard Deviation': std_val
            }
        stats_tbl= pd.DataFrame(stats_data)
        print(stats_tbl)

    else:
        print("no numerical columns")

orginal = calculate_stats(df)
cleaned= calculate_stats(df_cleaned)
print(orginal)
print(cleaned)

#-----------------------------------------------------------
#emove outliers from the DataFrame using the Z-score

import pandas as pd
import numpy as np
from scipy import stats

def remove_outlier(df):
    num_col = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_col) > 0:

        zscore = np.abs(stats.zscore(df[num_col]))
        threshold = 3
        outlier = (zscore > threshold)
        rows_ = np.any(outlier,axis=1)
        df_cleaned = df[~rows_]
        print(df_cleaned)
        return df_cleaned


df_cleaned = remove_outlier(df)

df_cleaned.to_csv('D:\\churnPrediction\\Cleaned.csv')

print(df_cleaned)

#-----------------------------------------------------------
#select categorical columns
# Find categorical columns | یافتن ستون‌های غیرعددی (Categorical)
cat_col = df.select_dtypes(include=['object']).columns
# Unique values for each categorical column | مقادیر یکتا در هر ستون کاتگوریکال
uq_values = []
for col in cat_col:
     # Unique values for each categorical column | مقادیر یکتا در هر ستون کاتگوریکال
     uq_val = df[col].unique()
     # Store name and unique values | ذخیره نام و مقدارها
     uq_values.append((col,uq_val))
     # Display as DataFrame | نمایش به صورت دیتافریم
     unique_values_df = pd.DataFrame(uq_values)
 # Show all rows of DataFrame | نمایش کامل ردیف‌های دیتافریم
pd.set_option('display.max_rows', None)  
print(unique_values_df)

#-----------------------------------------------------------
#show diagram and calculate unique values for each categorical
import pandas as pd
import matplotlib.pyplot as plt

cat_cols = df.select_dtypes(include=['object']).columns
for i,col in enumerate(cat_cols):
      print(f"Frequency of values in column '{col}':")
      val_cnt= df[col].value_counts()
      print(val_cnt)
      val_cnt.plot(kind='bar',figsize=(10,5),title=f'Frequency of values in column: {col}')
      plt.tight_layout()
      plt.show()

#-----------------------------------------------------------
# use on-hot encoding for categorical columns
import pandas as pd
# Select categorical columns | ستون‌هایی که نوع‌شان object است
cat_cols1=df_cleaned.select_dtypes(include=['object']).columns.tolist()
# Apply one-hot encoding and drop first to avoid dummy trap | اعمال وان-هات انکودینگ و حذف اولین مقدار از هر ستونس
df_encoded = pd.get_dummies(df_cleaned, columns=cat_cols1, drop_first=True)
print("\nFinal dataset after One-Hot Encoding:")

print(df_encoded.head())
#------------------------------------------------------------------
#Normalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import MinMaxScaler for normalization | ایمپورت MinMaxScaler برای نرمال‌سازی
from sklearn.preprocessing import MinMaxScaler

# فرض بر اینکه df_encoded تعریف شده باشد

# Make a copy before normalization | ساخت یک نسخه قبل از نرمال‌سازی
df_before = df_encoded.copy()

# نرمال‌سازی مین-مکس
sc = MinMaxScaler()
# Find all numerical columns | یافتن همه ستون‌های عددی
num_col2 = df_encoded.select_dtypes(include=[np.number]).columns
# Apply min-max scaling | اعمال نرمال‌سازی مین-مکس
df_encoded[num_col2] = sc.fit_transform(df_encoded[num_col2])

# تعداد ستون‌های عددی
num_col3 = len(num_col2)
rows_1 = num_col3
cols = 2

# ساخت figure و axs با فاصله زیاد بین سطرها
fig, axs = plt.subplots(rows_1, cols, figsize=(12, 6 * num_col3), sharex=False, sharey=False)

# عنوان کل
fig.suptitle('Comparison of Data Before and After Normalization', fontsize=14, y=1.03)

# Plot histograms before and after normalization
for i, column in enumerate(num_col2):
    # قبل از نرمال سازی
    ax_before = axs[i, 0]
    df_before[column].hist(bins=40, color='blue', ax=ax_before)
    ax_before.set_title(f'Before: {column}', fontsize=12)
    ax_before.set_ylabel('Count', fontsize=11)
    #ax_before.tick_params(axis='x', labelsize=10)
    #ax_before.grid(False)

    # بعد از نرمال سازی
    ax_after = axs[i, 1]
    df_encoded[column].hist(bins=40, color='green', ax=ax_after)
    ax_after.set_title(f'After: {column}', fontsize=12)
    ax_after.set_ylabel('Count', fontsize=11)
    #ax_after.tick_params(axis='x', labelsize=10)
    #ax_after.grid(False)


# plt.tight_layout(pad=5.0, h_pad=4.5, w_pad=2.5, rect=[0, 0, 1, 0.97])
# plt.subplots_adjust(top=0.95, hspace=1.0, wspace=0.35)
plt.subplots_adjust(top=0.95, hspace=0.9, wspace=0.3)

plt.tight_layout(pad=2.5, h_pad=4, w_pad=1)
plt.show()

#**********************************************************
#normaliztion using z-score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

sc_z = StandardScaler()
#use copy of the original dataset

df_standard= df_encoded.copy()  # Make a copy of the original dataset
df_standard[num_col2] = sc_z.fit_transform(df_encoded[num_col2])

# Number of numeric columns
num_cols_z= len(num_col2)
#create rows based on features num
rows_z = num_cols_z
cols_z = 2
#show figure size based on number of row
fig, axs = plt.subplots(rows_z, cols, figsize=(12, 3 * num_cols_z))
#set Subtitle
fig.suptitle('Comparison of Data Before and After Normalization-Z score', fontsize=16, y=2.50)
# Plot histograms  before and after normalization
for i, column in enumerate(num_col2):
        # Plot before normalization
    ax_before_z = axs[i, 0]
    df_encoded[column].hist(bins=60, color='blue', ax=ax_before_z)
    ax_before_z.set_title(f'Before Standardization: {column}', fontsize=10)

    # Plot after standardization
    ax_after_z = axs[i, 1]
    df_standard[column].hist(bins=60, color='green', ax=ax_after_z)
    ax_after_z.set_title(f'After Standardization: {column}', fontsize=10)

plt.subplots_adjust(top=0.95, hspace=0.4)
plt.show()

#*******************************************8
#merge min&max and Z-score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
num_col_m = df_encoded.select_dtypes(include=[np.number]).columns
#Start:MinMaxScaler
min_max_sc = MinMaxScaler()
df_min_max_sc = df_encoded.copy()
df_min_max_sc[num_col_m ] = min_max_sc.fit_transform(df_encoded[num_col_m ])
#end:MinMaxScaler
#Start:Z-Score
zScore = StandardScaler()
df_standardized_f = df_min_max_sc.copy()
df_standardized_f[num_col_m] = zScore.fit_transform(df_min_max_sc[num_col_m])
#end:z-score

num_col_f = len(num_col_m)
rows_f = num_col_f
col_f = 2
fig, axs = plt.subplots(rows_f, col_f, figsize=(12, 3 * num_col_f))
fig.suptitle('Comparison of Data After Min-Max Scaling and After Standardization', fontsize=16, y=2.50)
for i, column in enumerate(num_col_m):
    ax_min_max = axs[i, 0]
    df_min_max_sc[column].hist(bins=60, color='Yellow', ax=ax_min_max)
    ax_min_max.set_title(f'After Min-Max Scaling: {column}', fontsize=9)

    ax_standardized = axs[i, 1]
    df_standard[column].hist(bins=60, color='orange', ax=ax_standardized)
    ax_standardized.set_title(f'After Standardization: {column}', fontsize=9)

plt.subplots_adjust(top=0.95, hspace=0.4)
plt.show()

print(df_standard)

#-------------------------------------------------------------------
#training 880% data and validation 20%
import pandas as pd
# Import train_test_split for splitting the data | ایمپورت تابع تقسیم داده به آموزش و تست
from sklearn.model_selection import train_test_split
# Split the DataFrame into features (X) and target variable (Y) based on exited feature
# Features: Other columns except target | متغیرهای ورودی (بدون ستون هدف)
X = df_encoded.drop('Exited', axis=1)
# Target: Churn column | متغیر هدف یعنی ریزش
y = df_encoded['Exited']
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
print(f"Training data (X_train): {X_train.shape}")
print(f"Test data (X_test): {X_test.shape}")
print(f"Training target (y_train): {y_train.shape}")
print(f"Test target (y_test): {y_test.shape}")

#------------------------------------------------------
#logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Create logistic regression model | ساخت مدل رگرسیون لجستیک
log_reg_mdl = LogisticRegression()
# Train model with training data | آموزش مدل با داده آموزش
log_reg_mdl.fit(X_train, y_train)
# Predict churn for test data | پیش‌بینی مقادیر ریزش برای داده تست
y_pred = log_reg_mdl.predict(X_test)
# Calculate accuracy score | محاسبه دقت مدل
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Rate:"+ str(accuracy))

 # Print classification metrics | گزارش معیارهای دسته‌بندی مدل
print("\nClassification Report:")
print( classification_report(y_test, y_pred))
 # Create the confusion matrix | ساخت ماتریس آشفتگی
conf_mtx = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mtx, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()

#------------------------------------------------
#Decision Tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Import DecisionTree tools | وارد کردن ابزارهای درخت تصمیم
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Create decision tree model with max depth | ساخت مدل درخت تصمیم با عمق حداکثر 9s
tree_mdl = DecisionTreeClassifier(random_state=38, max_depth=9)
# Fit the tree on training data | آموزش مدل با داده‌های آموزش
tree_mdl.fit(X_train, y_train)
# Predict test labels | پیش‌بینی با مدل روی داده تست
predict = tree_mdl.predict(X_test)
print("Confusion Matrix:")
# Compute confusion matrix | محاسبه ماتریس آشفتگی
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(20, 15))
tree.plot_tree(tree_mdl, filled=True, feature_names=X_train.columns, class_names=['False', 'True'])
plt.title('Decision Tree')
plt.show()

#-----------------------------------------------

#xgboost model
# Import XGBoost classifier | وارد کردن مدل XGBoost
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#Create the XGBoost classifier model with specific parameters
# ساخت مدل XGBoost با تنظیمات خاص (بدون label encoder و معیار ارزیابی mlogloss)
xgb_mdl = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
#Train the XGBoost model on the training data
# آموزش مدل XGBoost با داده‌های آموزش (train)
xgb_mdl.fit(X_train, y_train)
# Predict the target values for the test data
# پیش‌بینی مقادیر خروجی برای داده‌های تست (test)
y_pred_xgb = xgb_mdl.predict(X_test)
# Compute accuracy of the model predictions
# محاسبه میزان دقت مدل
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy:"+ str(accuracy_xgb))
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
conf_mtx_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(conf_mtx_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.show()

#----------------------------------------------
#Gradient Boosting Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Create a Gradient Boosting model with 100 trees, learning rate 0.2, max_depth 5
# ساخت مدل گرادیانت بوستینگ با ۱۰۰ درخت، نرخ یادگیری ۰.۲ و عمق ۵
gb_mdl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=50)
# Create a Gradient Boosting model with 100 trees, learning rate 0.2, max_depth 5
# ساخت مدل گرادیانت بوستینگ با ۱۰۰ درخت، نرخ یادگیری ۰.۲ و عمق ۵
gb_mdl.fit(X_train, y_train)
# Predict labels of the test data using the trained model
# پیش‌بینی برچسب‌های داده تست توسط مدل آموزش دیده
gb_predict = gb_mdl.predict(X_test)
# Compute the confusion matrix between true and predicted labels
# محاسبه ماتریس آشفتگی بین برچسب‌های واقعی و پیش‌بینی شده ***(توجه: اینجا باید gb_predict باشه!)*
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)
# Create a classification report (precision, recall, F1-score) for model results
# تولید گزارش دسته‌بندی شامل precision، recall و F1 برای مدل *** (باز هم باید gb_predict باشه!) *
gb_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(gb_report)
# Calculate the accuracy of predictions
# محاسبه دقت مدل بر اساس پیش‌بینی‌ها *** اینجا هم باید gb_predict باشه! ***
accuracy_gb = accuracy_score(y_test, y_pred)
print("Gradient Boosting Accuracy :", str(accuracy_gb))
# Get feature importances from the trained model
# دریافت اهمیت ویژگی‌ها از مدل آموزش دیده
feature_importance = gb_mdl.feature_importances_
# Create a pandas Series with importance scores and feature names
# ساخت سری pandas با نمره اهمیت هر ویژگی و نام‌های ویژگی‌ها
imp = pd.Series(feature_importance, index=X.columns)
# Sort the importances descending (important features on top)
# ترتیب نزولی اهمیت ویژگی‌ها
imp = imp.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Score')
plt.show()

#------------------------------------------------------------
#evaluation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
# Define features X by dropping the target column
# تعریف ویژگی‌ها با حذف ستون هدف ('Exited') از داده‌ها
X = df_encoded.drop('Exited', axis=1)
# Define target y as the 'Exited' column
# تعریف متغیر هدف (y) با قرار دادن ستون Exited
y = df_encoded['Exited']
# Split the dataset into training and testing sets (80% train, 20% test)
# تقسیم داده به آموزش و تست (۸۰٪ آموزش، ۲۰٪ تست)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
# Define a function to evaluate models and collect results
# تعریف تابع برای ارزیابی مدل‌ها و جمع‌آوری نتایج
def evaluate_mdl(model):
    # Empty list to store each model's results
    # ایجاد لیست خالی برای ذخیره نتایج هر مدل
    resVals = []
    for mdl, model_name in model:
        mdl.fit(X_train, y_train)
        predict_mdl = mdl.predict(X_test)
        accuracy_mdl = accuracy_score(y_test, y_pred)
        rpt_classification = classification_report(y_test, y_pred, output_dict=True)
        # Append model name, accuracy, precision, recall, F1 to results list
        # اضافه کردن نام مدل، دقت، دقت وزنی، recall و F1 به لیست نتایج
        resVals.append({ 'Model': model_name,'Accuracy': accuracy_mdl,
            'Precision': rpt_classification['weighted avg']['precision'],
            'Recall': rpt_classification['weighted avg']['recall'],
            'F1 Score': rpt_classification['weighted avg']['f1-score'] })
    return pd.DataFrame(resVals)

#list of Models
models = [
    (LogisticRegression(max_iter=200), "Logistic Regression"),
    (XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), "XGBoost"),
    (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), "Gradient Boosting"),
    (DecisionTreeClassifier(random_state=42, max_depth=8), "Decision Tree")]
res_df = evaluate_mdl(models)

result = res_df.sort_values(by='Accuracy', ascending=False)
print(result)
# Bar plot for all key metrics for each model
# رسم نمودار ستونی برای نمایش دقت، precision، recall و F1 هر مدل
res_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison Using Accuracy')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()
#--------------------------------------------------
#Deep Neural network
# Import TensorFlow library for deep learning
# ایمپورت کتابخانه تنسورفلو برای یادگیری عمیق
import tensorflow as tf
#Import Sequential model API from Keras (high-level API on top of TensorFlow)
# ایمپورت مدل ترتیبی Keras که ساده‌ترین حالت ساخت مدل در Keras است
from tensorflow.keras.models import Sequential
# Import Dense (fully connected) and Input layers
# ایمپورت لایه Dense (کامل متصل) و لایه Input
from tensorflow.keras.layers import Dense, Input
# Number of input features (for example, 13 features in the dataset)
# تعیین تعداد ویژگی‌های ورودی مدل (مثلاً ۱۳ ویژگی در دیتاست)
inp_shape = 13


# Create a Sequential model object (stacking layers one after another)
# تعریف مدل ترتیبی (Sequential) برای افزودن لایه‌ها به صورت پشت سر هم
model = Sequential()

# Add an input layer with shape (13,)
# افزودن لایه ورودی با شکل (13،) با تعداد ویژگی‌های تعیین شده
model.add(Input(shape=(inp_shape,)))
# Add a dense hidden layer with 64 neurons and ReLU activation
# افزودن اولین لایه مخفی با ۶۴ نرون و تابع فعال‌سازی ReLU
model.add(Dense(64, activation='relu'))
# Add a second hidden layer with 128 neurons and ReLU activation
# افزودن دومین لایه مخفی با ۱۲۸ نرون و تابع فعال‌سازی ReLU (لایه عمیق‌تر)
model.add(Dense(128, activation='relu'))
# Add a third hidden layer with 64 neurons, again using ReLU
# افزودن سومین لایه مخفی با ۶۴ نرون و تابع فعال‌سازی ReLU (قرینه لایه اول)
model.add(Dense(64, activation='relu'))
# Add output layer with 1 neuron and sigmoid activation (for binary classification)
# افزودن لایه خروجی با ۱ نرون و تابع فعال‌سازی سیگموید (مناسب برای دسته‌بندی دودویی)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# - Optimizer: 'adam' is an adaptive learning rate optimization algorithm
# - Loss function: 'binary_crossentropy' is suitable for binary classification
# - Metrics: 'accuracy' to monitor model accuracy during training and evaluation
# Compile the model with specified optimizer, loss and metrics
# کامپایل مدل با بهینه‌ساز، تابع هزینه و معیار ارزیابی مشخص شده
model.compile(optimizer='adam'# Optimizer algorithm used for updating weights
    # الگوریتم آدام جهت بهینه‌سازی وزن‌ها
              , loss='binary_crossentropy'# Loss function for binary classification
    # تابع هزینه بر مبنای انتروپی متقاطع دودویی (مناسب دسته‌بندی دوتایی)
              , metrics=['accuracy']# Which metrics to show during training
    # معیار ارزیابی مدل که accuracy انتخاب شده است
              )
# Print the summary of the model (architecture, parameters)
# نمایش خلاصه ساختار مدل و تعداد پارامترهای قابل آموزش
model.summary()
#-----------------------------------------------------------
#training and evaluation of nerual network model
# Train the DNN model on the training data for 18 epochs with batch size 28,
# and evaluate validation metrics on the test set after each epoch.
# آموزش مدل شبکه عصبی روی داده‌های آموزش برای ۱۸ دوره (epoch) با اندازه دسته ۲۸،
# و ارزیابی عملکرد مدل روی داده‌های تست بعد از هر دوره (validation_data).
history = model.fit(X_train, y_train, epochs=18, batch_size=28, validation_data=(X_test, y_test))

# Evaluate the model on the test data (X_test, y_test)
# Evaluate the model on the test set, returning loss and accuracy.
# ارزیابی مدل روی داده‌های تست (X_test, y_test)، مقادیر خطا (loss) و دقت (accuracy) را برمی‌گرداند
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the test accuracy after evaluation
print(f"Test Accuracy: {test_accuracy}")