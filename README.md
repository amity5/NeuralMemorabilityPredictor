# **Predicting Image Memorability from Neural Responses**  

##  **Overview**  
This project explores how neural activity in **visual cortex regions** predicts the memorability of images using **fMRI (humans)** and **electrode (monkeys)** data. We analyze **response magnitudes** and apply **machine learning models** to assess how well activity in visual brain regions correlates with memorability scores.  

---

## **Research Question**  
**How can neural activity in visual cortex regions predict the memorability of visual stimuli?**  

---

## **Data Used**  

### **THINGS fMRI Dataset** (Human Data)  
- **Participants:** 3 humans (2 female, 1 male; mean age: 25.33 years)  
- **Stimuli:** 9,840 images  
- **Regions of Interest (ROIs):** FFA, LOC, PPA, EBA  

### **THINGS TVSD Dataset** (Monkey Data)  
- **Participants:** 2 monkeys  
- **Stimuli:** 22,248 images  
- **Regions of Interest (ROIs):** V1, V4, IT  

---

## **Approach**  

### **1️ Data Preparation & Preprocessing**  
- Loaded **voxel response data**, **memorability scores**, and **semantic category metadata**  
- Aggregated voxel data and computed **ROI-level response magnitudes**  
- Merged datasets by aligning **stimuli identifiers**  

### **2️ Response Magnitude-Based Regression**  
- Calculated **mean BOLD response magnitude** for each ROI  
- Correlated **responses with memorability scores** using Pearson correlation  
- Used these features as input to a **linear regression model**  

### **3️ Machine Learning Approach**  
- **Predicting Memorability Scores from Activity Patterns**  
- Experimenting with different models:  
  - **Support Vector Regression (SVR)**  
  - **Support Vector Classifier (SVC)**  
  - **Multi-Layer Perceptron (MLP)**  

---

## **Project Goal**  
- Investigate how neural activity in **visual cortex areas** (FFA, LOC, PPA, EBA, V1, V4, IT) encodes image memorability  
- Compare **ROI-level response magnitudes** and **pattern-based machine learning** to predict memorability  
- Bridge findings between human **fMRI** and **monkey electrode recordings**  

---

## **📂 Repository Structure**  
```
📁 Predicting-Image-Memorability  
│── 📂 data/            # Preprocessed data files (not included in repo)  
│── 📂 models/          # Machine learning models (SVR, SVC, MLP)  
│── 📂 notebooks/       # Jupyter notebooks for analysis  
│── 📂 results/         # Output plots, model evaluations  
│── preprocess.py       # Data preprocessing script  
│── train_model.py      # Training machine learning models  
│── README.md           # Project documentation  
```

---

## **Dependencies**  
To run this project, install required libraries:  
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

---

## **Usage**  

1️⃣ **Preprocess Data:**  
```bash
python preprocess.py
```  

2️⃣ **Train Models:**  
```bash
python train_model.py
```  

---

## **Contact**  
For any questions or contributions, feel free to open an issue or reach out! 🚀  
---
Email - amityelin@gmail.com
