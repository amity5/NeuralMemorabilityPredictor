# **Predicting Image Memorability from Neural Responses**

## **Overview**

This project explores how neural activity in **visual cortex regions** predicts the memorability of images using **fMRI (humans)** and **electrode (monkeys)** data. We analyze **response magnitudes** and apply **machine learning models** to assess how well activity in visual brain regions correlates with memorability scores.

---

## **Research Question**

**How can neural activity in visual cortex regions predict the memorability of visual stimuli?**

---

## **Data Used**

### **THINGS fMRI Dataset** (Human Data)

- **Participants:** 3 humans (2 female, 1 male; mean age: 25.33 years)
- **Stimuli:** 9,840 images
- **Regions of Interest (ROIs):**
  -Visual: FFA, LOC, PPA, EBA, Inferotemporal Cortex

  -Memory: Hippocampus, Entorhinal Cortex, Parahippocampal Areas (PHA1–3), Retrosplenial Cortex, Perirhinal Cortex, Presubiculum, Prosubiculum, Subgenual Cortex, Orbitofrontal Cortex, Anterior/Medial/Dorsolateral Prefrontal Cortex

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
📁 NeuralMemorabilityPredictor/
│
├── 📁 THINGS-fMRI-dataset-analysis/
│   ├── SVC_&_MLP_fMRI_THINGS.ipynb            # ML models for human fMRI data
│   ├── THINGS fMRI preprocess and data.ipynb  # Preprocessing pipeline
│   └── All_Areas_Statistical_Analysis_fMRI.ipynb  # ROI-level correlation analysis
│
├── 📁 TVSD-dataset-analysis/
│   ├── 📁 SVC and MLP TVSD/
│   │   └── SVC_and_MLP_TVSD.ipynb             # ML models for monkey data
│   │
│   └── 📁 TVSD preprocess and data exploration/
│       ├── Extract_data_TVSD.ipynb            # Raw data extraction
│       └── Preprocess_TVSD.ipynb              # Preprocessing pipeline
│
├── README.md                                   # Project documentation
└── workspace.code-workspace                    # VSCode workspace file

```

---

## **Dependencies**

To run this project, install required libraries:

```bash
pip install numpy pandas torch scikit-learn imbalanced-learn matplotlib seaborn h5py plotly
pip install ipykernel  # For running Jupyter notebooks
```

---

## **Usage**

**1. Preprocess Human fMRI Data:**

```bash
THINGS-fMRI-dataset-analysis/THINGS fMRI preprocess and data.ipynb
```

**2. Train & evaluate Models:**

```bash
THINGS-fMRI-dataset-analysis/SVC_&_MLP_fMRI_THINGS.ipynb
```

**3. Preprocess Monkey TVSD Data:**

```bash
TVSD-dataset-analysis/TVSD preprocess and data exploration/Preprocess_TVSD.ipynb
```

**4️. Train & Evaluate Models (Monkey TVSD):**

```bash
TVSD-dataset-analysis/SVC and MLP TVSD/SVC_and_MLP_TVSD.ipynb
```

---

## **Contact**

## For any questions or contributions, feel free to reach out!

Email - amityelin@gmail.com
