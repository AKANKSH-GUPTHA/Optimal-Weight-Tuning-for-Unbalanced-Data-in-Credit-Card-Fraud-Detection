# 💳 Optimal Weight-Tuning for Unbalanced Data in Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using advanced machine learning models such as Random Forest and CatBoost, with Bayesian weight-tuning and SMOTE sampling to handle highly imbalanced datasets.  
A Flask web application is provided for real-time fraud prediction with a simple, user-friendly interface.

---

## 🚀 How to Run the Project

1️⃣ Clone the repository
git lfs install
git clone https://github.com/AKANKSH-GUPTHA/Optimal-Weight-Tuning-for-Unbalanced-Data-in-Credit-Card-Fraud-Detection.git
cd Optimal-Weight-Tuning-for-Unbalanced-Data-in-Credit-Card-Fraud-Detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Flask app
python app.py

Then open your browser and go to:  
http://127.0.0.1:5000

---

## 📊 Dataset

• Name: Credit Card Fraud Detection Dataset  
• Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
• Description: Contains transactions made by European cardholders in September 2013. The dataset is highly unbalanced — only 0.172% of transactions are fraudulent.  
• Size: ~150 MB (creditcard.csv)  
• Note: The dataset is stored with Git LFS, so cloning requires Git LFS enabled.

---

## 🧠 Model Development

### Algorithms Used
- Random Forest Classifier  
- CatBoost Classifier  
- XGBoost & LightGBM (for comparison)  
- Ensemble techniques (Voting and Stacking)

### Techniques Applied
- SMOTE sampling to balance fraud vs. non-fraud data  
- Bayesian hyperparameter tuning for optimal model performance  
- Class-weight optimization to focus on the minority (fraudulent) class  
- Feature scaling and PCA for dimensionality reduction

### Model Performance Metrics
- Precision, Recall, and F1-Score  
- AUC-ROC curve for overall evaluation  
- Emphasis on high recall (catching most frauds)

---

## 🌐 Flask Web Application

The web interface allows users to:
- Input transaction details  
- Predict whether a transaction is Fraudulent or Legitimate  
- View results in real time

### Features
- Secure Login / Signup using SQLite (signup.db)  
- Model loaded dynamically from model_rf.sav  
- Clean, minimal front-end built with HTML and CSS

---

## 🧩 Folder Structure
Optimal-Weight-Tuning-for-Unbalanced-Data-in-Credit-Card-Fraud-Detection/
│
├── app.py                # Flask app
├── Notebook.ipynb        # Model training & analysis
├── model_rf.sav          # Trained RandomForest model
├── creditcard.csv        # Dataset (Git LFS)
├── signup.db             # SQLite database for login system
├── templates/            # HTML templates
├── static/               # CSS, JS assets
├── flowchart.txt         # Workflow explanation
├── testcase.txt          # Sample test data
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

---

## ⚙️ Tech Stack
- Python  
- Flask  
- Scikit-learn  
- CatBoost  
- Pandas, NumPy  
- SQLite  
- HTML, CSS  
- Git & GitHub

---

## 💡 Key Learnings
- Handling highly imbalanced datasets in real-world fraud detection  
- Using Bayesian optimization for hyperparameter tuning  
- Building a deployable machine learning pipeline with Flask  
- Managing large datasets via Git LFS

---

## 🧑‍💻 Author
Akanksh Modadugu  
Email: akankshmodadugu12345@gmail.com  
GitHub: https://github.com/AKANKSH-GUPTHA

---

## 🏁 License
This project is for educational and research purposes only.  
Dataset © Kaggle – Credit Card Fraud Detection.

---

## ✨ Example Snippet (Fraud Prediction Route)
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    prediction = model.predict(final_input)[0]
    if prediction == 1:
        return render_template('result.html', pred='🚨 Fraudulent Transaction Detected!')
    else:
        return render_template('result.html', pred='✅ Legitimate Transaction')

---

## ✅ Summary
✔️ Dataset handled with Git LFS  
✔️ Machine Learning models trained & optimized  
✔️ Flask app for real-time prediction  
✔️ Complete end-to-end fraud detection pipeline
