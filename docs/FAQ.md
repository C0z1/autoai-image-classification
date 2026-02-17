# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is this project about?
**A:** This project demonstrates automated image classification using pre-trained CNNs for feature extraction and IBM Watson AutoAI for automated machine learning. It's designed for educational purposes and follows academic assignment requirements.

### Q: Do I need coding experience?
**A:** Basic Python knowledge is helpful, but the scripts are ready to run. Most work involves:
- Running provided Python script (1 command)
- Using Watson Studio web interface (point-and-click)
- Taking screenshots
- Writing report from template

### Q: How long does this take?
**A:** Approximately 70 minutes total:
- Feature extraction: 5 minutes
- Watson Studio setup: 15 minutes
- AutoAI experiment: 30 minutes (automated)
- Report writing: 20 minutes

## Technical Questions

### Q: Why use pre-trained CNNs instead of training from scratch?
**A:** Pre-trained CNNs offer several advantages:
- Already learned useful image features from ImageNet
- No need for large training datasets
- Much faster than training from scratch
- Often achieves better results with small datasets
- Transfer learning is industry best practice

### Q: Which CNN model should I use?
**A:** 
- **ResNet50** (recommended): Best balance of accuracy and speed, 2048 features
- **MobileNetV2**: Faster but fewer features (1280), good for quick testing
- **EfficientNetB0**: Modern architecture, 1280 features, good accuracy

### Q: Can I use my own images?
**A:** Yes! Organize them like this:
```
dataset/
├── class_1/
│   ├── image1.jpg
│   └── ...
├── class_2/
│   └── ...
└── class_3/
    └── ...
```
Then run the script normally.

### Q: What image formats are supported?
**A:** JPG, JPEG, and PNG are supported.

### Q: How many images do I need?
**A:** 
- Minimum: 30 total (10 per class)
- Recommended: 60-80 (20-25 per class)
- More is better, but AutoAI handles small datasets well

### Q: How many classes can I have?
**A:** 
- Minimum: 3 classes (assignment requirement)
- Maximum: No hard limit, but 3-10 is typical
- More classes may need more images per class

## Watson Studio Questions

### Q: Is IBM Watson Studio free?
**A:** Yes, there's a free tier (Lite plan) that's sufficient for this project:
- 50 capacity unit-hours per month
- Sufficient for 2-3 AutoAI experiments
- No credit card required

### Q: How do I get IBM Cloud access?
**A:** 
1. Go to cloud.ibm.com
2. Sign up with email
3. Verify email
4. Access Watson Studio at dataplatform.cloud.ibm.com

### Q: Why did my AutoAI experiment fail?
**A:** Common causes:
- Target column not selected
- Dataset has <3 classes
- Dataset has <30 total samples
- Insufficient Watson ML capacity units
- Network/timeout issues

Solution: Check experiment settings and retry.

### Q: How long does AutoAI take to run?
**A:** Typically 20-30 minutes depending on:
- Dataset size
- Number of features
- Watson Studio load
- Optimization settings

### Q: Can I stop and resume an AutoAI experiment?
**A:** No, experiments must run to completion. If stopped, you'll need to start a new experiment.

## Results & Analysis Questions

### Q: What accuracy should I expect?
**A:** With good quality images:
- 85-95% is typical and excellent
- 75-85% is acceptable
- <75% may indicate issues with data or features

### Q: What if my accuracy is low (<75%)?
**A:** Try:
- Verify images are correctly labeled
- Ensure images are good quality
- Add more images per class
- Try a different CNN model
- Check class balance

### Q: What does "overfitting" mean?
**A:** When training accuracy is much higher than test accuracy (gap >10%):
- Model memorized training data
- Won't generalize to new images
- Solution: More data, regularization, simpler model

### Q: What does "underfitting" mean?
**A:** When both training and test accuracy are low (<70%):
- Model too simple for the problem
- Can't learn patterns from data
- Solution: More complex model, better features, more data

### Q: How do I know if my model is well-fitted?
**A:** Look for:
- Small gap between train/test accuracy (<5%)
- High performance on both (>85%)
- Consistent cross-validation results
- Good per-class performance

### Q: What are the most important features?
**A:** AutoAI provides feature importance scores. For CNN features:
- Higher-numbered features (later layers) often more important
- Represent more abstract patterns
- Distribution varies by dataset

## Assignment Questions

### Q: How many screenshots do I need?
**A:** Minimum 19 screenshots covering:
- Dataset and code (4)
- Watson Studio setup (4)
- AutoAI experiment (4)
- Results analysis (7+)

### Q: What should be in the report?
**A:** Required sections:
- Introduction (objective, dataset, methodology)
- Dataset preparation (CNN, preprocessing, integrity checks)
- Watson Studio configuration
- AutoAI experiment execution
- Results interpretation (top 3 pipelines, metrics)
- Feature importance analysis
- Overfitting/underfitting diagnosis
- Conclusions
- References (3+)

### Q: How long should the report be?
**A:** 2-4 pages is standard:
- Not just screenshots
- Include analysis and interpretation
- Explain why results occurred
- Draw evidence-based conclusions

### Q: What references should I cite?
**A:** Include:
- ResNet paper (He et al., 2016)
- IBM AutoAI documentation
- Deep learning textbook
- Any other sources used

## Troubleshooting

### Q: TensorFlow installation fails
**A:** Try:
```bash
pip install tensorflow-cpu  # Use CPU version
# OR
conda install tensorflow
# OR
pip install tensorflow==2.13.0  # Specific version
```

### Q: Out of memory error
**A:** Reduce batch size in script:
```python
features = extract_features_batch(paths, model, batch_size=8)
```

### Q: Watson Studio dataset upload fails
**A:** Check:
- File size <500MB
- Valid CSV format
- No special characters in headers
- Try different browser

### Q: AutoAI won't start
**A:** Verify:
- Target column selected
- Dataset loaded successfully
- Watson ML service created
- Sufficient capacity units available

### Q: Images won't load
**A:** Ensure:
- Images are in correct directories
- File extensions are .jpg, .jpeg, or .png
- Files aren't corrupted
- Permissions allow reading

## Best Practices

### Q: How can I get the best results?
**A:** Follow these tips:
- Use high-quality, clear images
- Ensure consistent image characteristics per class
- Balance class sizes (equal images per class)
- Remove duplicate or near-duplicate images
- Verify all labels are correct
- Take comprehensive screenshots during experiment

### Q: How do I ensure reproducibility?
**A:** Save:
- Original images
- Feature extraction code
- CSV datasets
- Watson Studio experiment settings
- All screenshots
- Final model

### Q: Should I tune hyperparameters manually?
**A:** No need! AutoAI automatically:
- Selects algorithms
- Optimizes hyperparameters
- Applies preprocessing
- Performs feature engineering

## Additional Help

### Q: Where can I get more help?
**A:** Resources:
- Check documentation in `docs/` folder
- Review example output in `examples/`
- Open GitHub issue for bugs
- Check Watson Studio documentation
- Review TensorFlow/Keras docs

### Q: Can I use this for commercial projects?
**A:** Yes, with MIT License. However:
- IBM Watson Studio has usage limits
- ResNet50 is pre-trained on ImageNet (research use)
- Check IBM terms for commercial Watson use

### Q: How do I cite this project?
**A:** 
```
AutoAI Image Classification Project (2026)
GitHub: https://github.com/yourusername/autoai-image-classification
```

Still have questions? Open an issue on GitHub!
