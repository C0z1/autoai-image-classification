# Advanced Usage Guide

This guide covers advanced topics and customization options for the AutoAI Image Classification project.

## Table of Contents
1. [Custom Dataset Integration](#custom-dataset-integration)
2. [Model Customization](#model-customization)
3. [Performance Optimization](#performance-optimization)
4. [Batch Processing](#batch-processing)
5. [Integration with Other Tools](#integration-with-other-tools)
6. [Production Deployment](#production-deployment)

---

## Custom Dataset Integration

### Using Your Own Images

#### Method 1: Replace Sample Dataset

1. **Organize your images:**
```
dataset/
├── cats/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   └── ... (20+ images)
├── dogs/
│   ├── dog_001.jpg
│   └── ...
└── birds/
    ├── bird_001.jpg
    └── ...
```

2. **Edit configuration in `src/1_feature_extraction.py`:**
```python
class Config:
    DATASET_DIR = 'dataset'  # Your images directory
    CLASSES = ['cats', 'dogs', 'birds']  # Auto-detected
    MIN_IMAGES_PER_CLASS = 10
    MAX_IMAGES_PER_CLASS = 50  # Increase if you have more
```

3. **Run extraction:**
```bash
python src/1_feature_extraction.py
```

#### Method 2: Programmatic Integration

```python
from pathlib import Path
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Process single image
def extract_single_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features[0]

# Process directory
def extract_directory(directory):
    results = []
    for img_file in Path(directory).glob('*.jpg'):
        features = extract_single_image(img_file)
        results.append({
            'filename': img_file.name,
            'features': features
        })
    return results
```

---

## Model Customization

### Using Different CNN Architectures

#### EfficientNetB3 (More Accurate)

```python
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# In 1_feature_extraction.py, modify load_feature_extractor:
def load_feature_extractor(model_name='EfficientNetB3'):
    if model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        feature_dim = 1536  # EfficientNetB3 output
        return base_model, feature_dim
```

#### ResNet152 (Deeper Network)

```python
from tensorflow.keras.applications import ResNet152

def load_feature_extractor(model_name='ResNet152'):
    if model_name == 'ResNet152':
        base_model = ResNet152(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        feature_dim = 2048
        return base_model, feature_dim
```

### Multi-Layer Feature Extraction

Extract features from multiple layers for richer representation:

```python
from tensorflow.keras.models import Model

def create_multi_layer_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Extract from multiple layers
    layer_names = ['conv4_block6_out', 'conv5_block3_out']
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    # Add global average pooling to each
    from tensorflow.keras.layers import GlobalAveragePooling2D
    pooled = [GlobalAveragePooling2D()(layer) for layer in layers]
    
    # Concatenate features
    from tensorflow.keras.layers import Concatenate
    combined = Concatenate()(pooled)
    
    model = Model(inputs=base_model.input, outputs=combined)
    return model
```

---

## Performance Optimization

### Memory Optimization

For large datasets or limited RAM:

```python
# Process in smaller batches
def extract_features_streaming(image_paths, model, batch_size=4):
    """Memory-efficient feature extraction"""
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = load_batch(batch_paths)
        batch_features = model.predict(batch_images, verbose=0)
        
        # Save immediately to disk
        np.save(f'features_batch_{i}.npy', batch_features)
        
        # Clear memory
        del batch_images, batch_features
        
    # Combine later
    all_features = []
    for i in range(0, len(image_paths), batch_size):
        features = np.load(f'features_batch_{i}.npy')
        all_features.append(features)
    
    return np.vstack(all_features)
```

### GPU Acceleration

Enable GPU for faster processing:

```python
import tensorflow as tf

# Check GPU availability
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Use mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Parallel Processing

Process multiple images simultaneously:

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def extract_parallel(image_paths, model, n_workers=4):
    """Parallel feature extraction"""
    
    def process_chunk(chunk):
        return model.predict(np.array([
            preprocess_image(p) for p in chunk
        ]), verbose=0)
    
    # Split into chunks
    chunk_size = len(image_paths) // n_workers
    chunks = [
        image_paths[i:i+chunk_size]
        for i in range(0, len(image_paths), chunk_size)
    ]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    return np.vstack(results)
```

---

## Batch Processing

### Processing Large Image Collections

```python
import pandas as pd
from pathlib import Path

def process_large_dataset(root_dir, output_csv, batch_size=100):
    """Process dataset larger than memory"""
    
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Get all images
    image_paths = list(Path(root_dir).rglob('*.jpg'))
    print(f"Found {len(image_paths)} images")
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(image_paths), batch_size)):
        print(f"Processing batch {batch_num+1}...")
        
        batch_paths = image_paths[i:i+batch_size]
        features = extract_features_batch(batch_paths, model)
        labels = [p.parent.name for p in batch_paths]
        
        # Create DataFrame
        feature_cols = [f'feature_{j:04d}' for j in range(features.shape[1])]
        df = pd.DataFrame(features, columns=feature_cols)
        df['label'] = labels
        df['image_file'] = [p.name for p in batch_paths]
        
        # Append to CSV
        mode = 'w' if batch_num == 0 else 'a'
        header = batch_num == 0
        df.to_csv(output_csv, mode=mode, header=header, index=False)
    
    print(f"Complete! Saved to {output_csv}")
```

---

## Integration with Other Tools

### Export to NumPy

```python
# Save features as NumPy array
import numpy as np

features = np.array(extracted_features)
labels = np.array(label_list)

np.save('features.npy', features)
np.save('labels.npy', labels)

# Load later
features = np.load('features.npy')
labels = np.load('labels.npy')
```

### Export to HDF5

```python
import h5py

def save_to_hdf5(features, labels, image_files, output_path):
    """Save to HDF5 format for efficient storage"""
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=features, compression='gzip')
        f.create_dataset('labels', data=labels.astype('S'))
        f.create_dataset('image_files', data=image_files.astype('S'))
        
        # Add metadata
        f.attrs['feature_dim'] = features.shape[1]
        f.attrs['num_samples'] = features.shape[0]
        f.attrs['model'] = 'ResNet50'

# Load from HDF5
def load_from_hdf5(input_path):
    with h5py.File(input_path, 'r') as f:
        features = f['features'][:]
        labels = f['labels'][:].astype(str)
        return features, labels
```

### Integration with scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load features
df = pd.read_csv('autoai_image_features_dataset.csv')
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Production Deployment

### Create Inference Pipeline

```python
class ImageClassifier:
    """Production-ready classifier"""
    
    def __init__(self, model_path, feature_extractor='ResNet50'):
        self.feature_extractor = self._load_feature_extractor(feature_extractor)
        self.classifier = self._load_classifier(model_path)
    
    def _load_feature_extractor(self, model_name):
        if model_name == 'ResNet50':
            return ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    def _load_classifier(self, path):
        # Load Watson AutoAI model or scikit-learn model
        import joblib
        return joblib.load(path)
    
    def predict(self, image_path):
        """Predict single image"""
        # Extract features
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.feature_extractor.predict(x, verbose=0)
        
        # Predict
        prediction = self.classifier.predict(features)
        probability = self.classifier.predict_proba(features)
        
        return {
            'class': prediction[0],
            'confidence': float(np.max(probability)),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.classifier.classes_, probability[0])
            }
        }
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        return [self.predict(path) for path in image_paths]

# Usage
classifier = ImageClassifier('model.pkl')
result = classifier.predict('test_image.jpg')
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

### REST API with Flask

```python
from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)
classifier = ImageClassifier('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for image classification"""
    
    # Get image from request
    if 'image' in request.files:
        image_file = request.files['image']
        image_path = save_upload(image_file)
    elif 'base64' in request.json:
        image_data = base64.b64decode(request.json['base64'])
        image_path = save_base64(image_data)
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    # Predict
    try:
        result = classifier.predict(image_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY model.pkl .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t image-classifier .
docker run -p 5000:5000 image-classifier
```

---

## Best Practices

### 1. Data Validation

```python
def validate_dataset(df):
    """Validate dataset before AutoAI"""
    
    checks = {
        'Missing values': df.isnull().sum().sum() == 0,
        'Uniform dimensions': df.shape[0] > 0,
        'Label column exists': 'label' in df.columns,
        'Min samples per class': df['label'].value_counts().min() >= 10,
        'Feature dtype': df.drop('label', axis=1).dtypes.unique()[0] == 'float64'
    }
    
    print("Dataset Validation:")
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check}")
    
    return all(checks.values())
```

### 2. Experiment Tracking

```python
import mlflow

def track_experiment(model_name, features, labels):
    """Track experiments with MLflow"""
    
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        mlflow.log_param("features_dim", features.shape[1])
        mlflow.log_param("num_samples", features.shape[0])
        mlflow.log_param("num_classes", len(np.unique(labels)))
        
        # Train and log metrics
        # ... (your training code)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact("model.pkl")
```

### 3. Monitoring

```python
def monitor_predictions(predictions, true_labels):
    """Monitor model performance in production"""
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'timestamp': datetime.now().isoformat(),
        'num_predictions': len(predictions)
    }
    
    # Log to monitoring system
    with open('monitoring_log.json', 'a') as f:
        json.dump(metrics, f)
        f.write('\n')
    
    # Alert if performance drops
    if metrics['accuracy'] < 0.80:
        send_alert(f"Model accuracy dropped to {metrics['accuracy']:.2%}")
```

---

## Troubleshooting

### Large Dataset Issues

**Problem:** Out of memory with 1000+ images

**Solution:**
```python
# Use generator
def image_generator(paths, batch_size=32):
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        yield load_images(batch)

# Process with generator
for batch in image_generator(all_paths):
    features = model.predict(batch)
    save_batch(features)
```

### Slow Processing

**Problem:** Feature extraction takes too long

**Solutions:**
1. Use GPU: `pip install tensorflow-gpu`
2. Reduce batch size for better pipelining
3. Use faster model: MobileNetV2
4. Use mixed precision: `tf.keras.mixed_precision`

---

For more help, see [FAQ.md](FAQ.md) or open an issue on GitHub.
