# Internship Tasks – Machine Learning  

This repository contains the tasks I completed during my **2-week remote internship on Machine Learning**.  
The internship required a minimum of **3 tasks**, but I successfully completed **all 8 tasks** to gain broader practical experience.  

---

## 📂 Repository Structure
Each folder corresponds to one task and contains the code, datasets (if applicable), and results.
```
├── Task1_StudentScorePrediction/
├── Task2_CustomerSegmentation/
├── Task3_ForestCoverClassification/
├── Task4_LoanApprovalPrediction/
├── Task5_MovieRecommendation/
├── Task6_MusicGenreClassification/
├── Task7_SalesForecasting/
└── Task8_TrafficSignRecognition/
```

---

## ✅ Completed Tasks

### **Task 1: Student Score Prediction**
- Built a model to predict students' exam scores based on study hours and related factors.  
- Performed data cleaning, visualization, and train-test split.  
- Trained a **Linear Regression model**, visualized predictions, and evaluated performance.  
- **Bonus:** Explored polynomial regression and experimented with feature combinations.  

---

### **Task 2: Customer Segmentation**
- Clustered customers into segments based on **income** and **spending score** using the Mall Customer dataset.  
- Applied **K-Means clustering**, determined the optimal number of clusters, and visualized results.  
- Performed scaling and exploratory analysis.  
- **Bonus:** Tested alternative algorithms like **DBSCAN** and analyzed spending patterns per cluster.  

---

### **Task 3: Forest Cover Type Classification**
- Predicted types of forest cover using **environmental and cartographic features** from the UCI Covertype dataset.  
- Cleaned and preprocessed categorical data.  
- Trained and evaluated **multi-class classification models**.  
- Visualized confusion matrices and feature importance.  
- **Bonus:** Compared models such as **Random Forest** and **XGBoost** with hyperparameter tuning.  

---

### **Task 4: Loan Approval Prediction**
- Built a model to predict **loan application approvals**.  
- Preprocessed data: handled missing values and encoded categorical features.  
- Focused on **imbalanced data challenges**, using metrics such as precision, recall, and F1-score.  
- **Bonus:** Implemented **SMOTE** for balancing and compared **logistic regression** vs. decision tree models.  

---

### **Task 5: Movie Recommendation System**
- Developed a **user-based collaborative filtering** system using the MovieLens dataset.  
- Constructed a user-item matrix to compute similarity scores.  
- Recommended unseen, top-rated movies for users.  
- Evaluated recommendation quality using **Precision@K**.  
- **Bonus:** Experimented with **item-based filtering** and **matrix factorization (SVD)**.  

---

### **Task 6: Music Genre Classification**
- Classified songs into genres using the **GTZAN dataset**.  
- Extracted features (e.g., **MFCCs**) with Librosa and tested both tabular and image-based approaches.  
- Implemented **multi-class models** using Scikit-learn and CNNs for spectrogram images.  
- **Bonus:** Applied **transfer learning** for CNN models.  

---

### **Task 7: Sales Forecasting**
- Forecasted **future sales** using Walmart’s historical sales data.  
- Engineered **time-based features** such as lags, rolling averages, and seasonal decomposition.  
- Trained regression models and visualized predicted vs. actual sales over time.  
- **Bonus:** Tested **XGBoost/LightGBM** with time-aware validation.  

---

### **Task 8: Traffic Sign Recognition**
- Built a **CNN model** for traffic sign classification using the GTSRB dataset.  
- Preprocessed images (resizing, normalization).  
- Trained and evaluated deep learning models with accuracy and confusion matrices.  
- **Bonus:** Compared performance of a **custom CNN** vs. pre-trained models (e.g., MobileNet) and used **data augmentation**.  

---

## 🛠️ Tools & Libraries
- **Python**  
- **NumPy, Pandas, Matplotlib, Seaborn**  
- **Scikit-learn, XGBoost, LightGBM**  
- **Librosa** (for audio features)  
- **TensorFlow/Keras** (for CNN models)  
- **OpenCV** (for image preprocessing)  

---

## 🚀 Highlights
- Completed **all 8 tasks** in just 2 weeks.  
- Worked with **real-world datasets** (student performance, mall customers, UCI Covertype, MovieLens, GTZAN, Walmart sales, GTSRB).  
- Covered a wide range of ML topics: **regression, classification, clustering, recommendation, forecasting, and deep learning**.  

---

## 📖 How to Use
1. Clone the repository:  
   ```bash
   git clone https://github.com/username/repo-name.git
2. Navigate into any task folder.

3. Upload or provide access to the dataset:

   - Option A: Upload directly to Colab during runtime
     ```bash
     from google.colab import files
     uploaded = files.upload()
   - Option B: Mount Google Drive and access the dataset from there
     ```bash
     from google.colab import drive
     drive.mount('/content/drive')

3. Open the Jupyter Notebook or Python script to explore the code and results.

## 🙌 Acknowledgment

Thanks to Elevvo for the opportunity to apply and expand my skills through this internship.
