# Quick Start Guide 🚀

Get up and running in under 10 minutes!

## Prerequisites Check ✓

Before starting, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] pip package manager
- [ ] 2GB free disk space
- [ ] Internet connection
- [ ] IBM Cloud account (sign up is free)

## Step-by-Step Setup

### 1. Clone & Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/autoai-image-classification.git
cd autoai-image-classification

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup.py
```

**Expected output:** All checks should pass ✓

---

### 2. Extract Features (3 minutes)

```bash
# Run feature extraction
python src/1_feature_extraction.py
```

**What happens:**
- Creates sample dataset (or uses your images)
- Loads ResNet50 model
- Extracts 2048-dimensional features
- Creates CSV files

**Output files:**
- `autoai_image_features_dataset.csv` ← Upload this to Watson!
- `image_features_dataset.csv`
- `dataset_summary.txt`
- `sample_images.png`
- `feature_extraction_documentation.txt`

**Verify:** Check that CSV file exists and has 2049 columns (2048 features + 1 label)

---

### 3. Watson Studio (5 minutes)

#### 3.1 Create Account
1. Go to [IBM Cloud](https://cloud.ibm.com)
2. Sign up (free, no credit card)
3. Verify email

#### 3.2 Create Services
1. Search "Watson Studio" → Create
2. Search "Cloud Object Storage" → Create (Lite plan)
3. Search "Watson Machine Learning" → Create (Lite plan)

#### 3.3 Create Project
1. Go to [Watson Studio](https://dataplatform.cloud.ibm.com)
2. Click "Create a project"
3. Choose "Create an empty project"
4. Name: "AutoAI Image Classification"
5. Select your Cloud Object Storage
6. Click "Create"

#### 3.4 Upload Dataset
1. In your project, click "Assets" tab
2. Drag and drop `autoai_image_features_dataset.csv`
3. Wait for upload (green checkmark)

**Checkpoint:** Dataset should appear in Assets list

---

### 4. Run AutoAI (30 minutes automated)

#### 4.1 Create Experiment
1. Click "Add to project" → "AutoAI experiment"
2. Name: "Image Classification AutoAI"
3. Select your Watson Machine Learning service
4. Click "Create"

#### 4.2 Configure & Run
1. Click "Select from project"
2. Choose `autoai_image_features_dataset.csv`
3. Select "label" as target column
4. Verify: Shows "Multiclass Classification"
5. Click "Run experiment"

#### 4.3 Wait & Capture Screenshots
While experiment runs (~30 min):
- ✅ Take screenshot of progress screen
- ✅ When complete, screenshot leaderboard
- ✅ Screenshot top 3 pipelines
- ✅ Screenshot confusion matrix
- ✅ Screenshot feature importance

**Important:** Can't go back for screenshots later!

---

### 5. Analyze Results (10 minutes)

#### 5.1 View Leaderboard
- Note top 3 pipeline algorithms
- Record accuracy, precision, recall, F1
- Compare metrics

#### 5.2 Examine Top Pipeline
Click on Pipeline 1 (top ranked):
- View relationship map
- Check feature importance
- Review confusion matrix
- Compare holdout vs CV metrics

#### 5.3 Diagnose Fit
Calculate gap:
```
Gap = |CV Accuracy - Holdout Accuracy|

If gap < 5%: ✓ Well-fitted
If gap 5-10%: ⚠️ Mild overfitting
If gap > 10%: ❌ Overfitting
```

---

### 6. Write Report (20 minutes)

Use `docs/REPORT_TEMPLATE.md`:

1. Fill in your information
2. Insert all 19+ screenshots
3. Add your specific results
4. Write analysis sections
5. Check references
6. Export as PDF or DOCX

**Sections to complete:**
- [x] Introduction
- [x] Dataset preparation
- [x] Watson Studio setup
- [x] AutoAI execution
- [x] Results interpretation
- [x] Feature importance
- [x] Overfitting diagnosis
- [x] Conclusions
- [x] References

---

## ✅ Completion Checklist

Before submitting:

### Code & Data
- [ ] Feature extraction runs successfully
- [ ] CSV has 2048 features + label column
- [ ] No missing values in dataset
- [ ] All output files generated

### Watson Studio
- [ ] Project created
- [ ] Dataset uploaded
- [ ] AutoAI experiment completed
- [ ] 19+ screenshots captured

### Analysis
- [ ] Top 3 pipelines compared
- [ ] Feature importance analyzed
- [ ] Overfitting/underfitting diagnosed
- [ ] Metrics interpreted

### Report
- [ ] 2-4 pages (PDF or DOCX)
- [ ] All sections included
- [ ] Screenshots inserted with captions
- [ ] 3+ references cited
- [ ] Proofread

---

## 🎯 Expected Results

### Feature Extraction
```
✓ 60-80 images processed
✓ 2048 features per image
✓ 3 classes balanced
✓ CSV size: ~1-2 MB
✓ Processing time: ~2 minutes
```

### AutoAI
```
✓ 8 pipelines generated
✓ Top accuracy: 90-95%
✓ Best algorithm: XGBoost or Random Forest
✓ Runtime: 20-30 minutes
✓ Status: Complete
```

### Model Performance
```
✓ Accuracy: ~94%
✓ Precision: ~93%
✓ Recall: ~94%
✓ F1-Score: ~94%
✓ Gap: <2% (well-fitted)
```

---

## 🆘 Quick Troubleshooting

### "TensorFlow won't install"
```bash
pip install tensorflow-cpu
```

### "Out of memory"
Edit `src/1_feature_extraction.py` line 298:
```python
batch_size=8  # Reduce from 16
```

### "Watson Studio upload fails"
- Check file <500MB
- Use Chrome or Firefox
- Verify CSV format

### "AutoAI won't start"
- Verify target column selected
- Check ≥3 classes
- Ensure ≥30 samples

---

## 🎉 Success Indicators

You're on track if:
- ✅ Script runs without errors
- ✅ CSV opens in Excel/Numbers
- ✅ Watson Studio shows dataset
- ✅ AutoAI completes experiment
- ✅ Leaderboard shows results
- ✅ Accuracy >85%

Expected grade: **95-100/100** 🎓

---

## 📞 Need Help?

1. Check [FAQ](docs/FAQ.md)
2. Review [Watson Studio Guide](docs/WATSON_STUDIO_GUIDE.md)
3. See [example output](examples/sample_output.txt)
4. Open GitHub issue

---

## ⏱️ Time Breakdown

| Task | Time |
|------|------|
| Setup & installation | 2 min |
| Feature extraction | 3 min |
| Watson Studio setup | 5 min |
| AutoAI (automated) | 30 min |
| Screenshot capture | 5 min |
| Results analysis | 10 min |
| Report writing | 20 min |
| **Total** | **~75 min** |

---

**You're ready! Start with Step 1 above. Good luck! 🚀**
