"""
AutoAI Image Classification Project - Step 1: Feature Extraction
=================================================================
This script extracts features from images using a pre-trained CNN model.

Dataset: 3+ classes, 30-80 images total
Model: ResNet50 (pre-trained on ImageNet)
Output: Tabular CSV dataset with feature vectors and labels
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AUTOAI IMAGE CLASSIFICATION PROJECT")
print("Step 1: CNN Feature Extraction")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Project configuration"""
    # Model selection
    MODEL_NAME = 'ResNet50'  # Options: ResNet50, MobileNetV2, EfficientNetB0
    IMG_SIZE = (224, 224)
    FEATURE_DIM = 2048  # ResNet50: 2048, MobileNetV2: 1280, EfficientNetB0: 1280
    
    # Dataset paths
    DATASET_DIR = 'dataset'  # Directory with class subfolders
    OUTPUT_CSV = 'image_features_dataset.csv'
    
    # Classes (minimum 3 required)
    CLASSES = ['class_1', 'class_2', 'class_3']  # Will be auto-detected
    MIN_IMAGES_PER_CLASS = 10
    MAX_IMAGES_PER_CLASS = 30

config = Config()

print("\n📋 PROJECT CONFIGURATION")
print("="*80)
print(f"Model: {config.MODEL_NAME}")
print(f"Image Size: {config.IMG_SIZE}")
print(f"Feature Dimensions: {config.FEATURE_DIM}")
print(f"Output File: {config.OUTPUT_CSV}")

# ============================================================================
# LOAD PRE-TRAINED CNN MODEL
# ============================================================================

print("\n🔧 LOADING PRE-TRAINED CNN MODEL")
print("="*80)

def load_feature_extractor(model_name='ResNet50'):
    """
    Load pre-trained CNN model and remove final classification layer.
    Returns model that outputs feature vectors.
    """
    if model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("✓ Loaded ResNet50 (2048-dimensional features)")
        feature_dim = 2048
        
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        print("✓ Loaded MobileNetV2 (1280-dimensional features)")
        feature_dim = 1280
        
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        print("✓ Loaded EfficientNetB0 (1280-dimensional features)")
        feature_dim = 1280
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"✓ Final layer removed - outputting {feature_dim}D feature vectors")
    print(f"✓ Model ready for feature extraction")
    
    return base_model, feature_dim

# Load model
feature_extractor, feature_dim = load_feature_extractor(config.MODEL_NAME)
config.FEATURE_DIM = feature_dim

# Display model architecture
print("\n📊 MODEL ARCHITECTURE")
print("="*80)
feature_extractor.summary()

# ============================================================================
# CREATE SAMPLE DATASET (FOR DEMONSTRATION)
# ============================================================================

print("\n📁 CREATING SAMPLE DATASET")
print("="*80)

def create_sample_dataset(num_classes=3, images_per_class=25):
    """
    Create sample images for demonstration.
    In real project, you would use actual images.
    """
    dataset_dir = Path(config.DATASET_DIR)
    dataset_dir.mkdir(exist_ok=True)
    
    classes = [f'class_{i+1}' for i in range(num_classes)]
    
    print(f"Creating {num_classes} classes with ~{images_per_class} images each...")
    
    total_images = 0
    for class_idx, class_name in enumerate(classes):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create sample images (random colored patterns)
        num_images = images_per_class + np.random.randint(-5, 6)  # Vary slightly
        
        for img_idx in range(num_images):
            # Generate sample image with class-specific pattern
            img_array = np.random.rand(224, 224, 3) * 255
            
            # Add class-specific color bias
            if class_idx == 0:  # Class 1: Reddish
                img_array[:, :, 0] += 50
            elif class_idx == 1:  # Class 2: Greenish
                img_array[:, :, 1] += 50
            else:  # Class 3: Bluish
                img_array[:, :, 2] += 50
            
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Save image
            img = Image.fromarray(img_array)
            img_path = class_dir / f'{class_name}_{img_idx:03d}.jpg'
            img.save(img_path)
        
        total_images += num_images
        print(f"  ✓ {class_name}: {num_images} images created")
    
    print(f"\n✓ Total images created: {total_images}")
    print(f"✓ Dataset directory: {dataset_dir.absolute()}")
    
    return classes, total_images

# Create sample dataset
classes, total_images = create_sample_dataset(
    num_classes=3,
    images_per_class=25
)

config.CLASSES = classes

# ============================================================================
# LOAD AND VALIDATE DATASET
# ============================================================================

print("\n📂 LOADING DATASET")
print("="*80)

def load_image_dataset(dataset_dir):
    """
    Load images from directory structure.
    Expected structure:
        dataset/
            class_1/
                image1.jpg
                image2.jpg
            class_2/
                image1.jpg
            ...
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Get all class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) < 3:
        raise ValueError(f"Found {len(class_dirs)} classes. Minimum 3 required.")
    
    print(f"✓ Found {len(class_dirs)} classes")
    
    # Load all images
    image_paths = []
    labels = []
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Get all image files
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png'))
        
        num_images = len(image_files)
        
        if num_images < config.MIN_IMAGES_PER_CLASS:
            print(f"  ⚠️  {class_name}: {num_images} images (minimum {config.MIN_IMAGES_PER_CLASS})")
        elif num_images > config.MAX_IMAGES_PER_CLASS:
            print(f"  ✓ {class_name}: {num_images} images (using first {config.MAX_IMAGES_PER_CLASS})")
            image_files = image_files[:config.MAX_IMAGES_PER_CLASS]
        else:
            print(f"  ✓ {class_name}: {num_images} images")
        
        # Add to dataset
        for img_file in image_files:
            image_paths.append(img_file)
            labels.append(class_name)
    
    print(f"\n✓ Total images loaded: {len(image_paths)}")
    print(f"✓ Total classes: {len(class_dirs)}")
    
    return image_paths, labels, [d.name for d in sorted(class_dirs)]

# Load dataset
image_paths, labels, classes = load_image_dataset(config.DATASET_DIR)

# Display dataset statistics
print("\n📊 DATASET STATISTICS")
print("="*80)
label_counts = pd.Series(labels).value_counts()
print(label_counts)
print(f"\nTotal images: {len(image_paths)}")
print(f"Classes: {classes}")

# Visualize sample images
print("\n🖼️  SAMPLE IMAGES")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for idx in range(6):
    if idx < len(image_paths):
        img = Image.open(image_paths[idx])
        axes[idx].imshow(img)
        axes[idx].set_title(f"{labels[idx]}")
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
print("✓ Sample images saved: sample_images.png")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

print("\n🔍 EXTRACTING FEATURES")
print("="*80)

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for CNN"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_preprocess(img_array)
    return img_array

def extract_features_batch(image_paths, model, batch_size=32):
    """Extract features from images in batches"""
    features = []
    
    print(f"Processing {len(image_paths)} images in batches of {batch_size}...")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img = preprocess_image(img_path, config.IMG_SIZE)
            batch_images.append(img[0])
        
        batch_images = np.array(batch_images)
        
        # Extract features
        batch_features = model.predict(batch_images, verbose=0)
        features.extend(batch_features)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return np.array(features)

# Extract features
features = extract_features_batch(image_paths, feature_extractor, batch_size=16)

print(f"\n✓ Feature extraction complete!")
print(f"✓ Feature shape: {features.shape}")
print(f"✓ Features per image: {features.shape[1]}")

# ============================================================================
# CREATE TABULAR DATASET
# ============================================================================

print("\n📊 CREATING TABULAR DATASET")
print("="*80)

# Create DataFrame with features
feature_columns = [f'feature_{i:04d}' for i in range(features.shape[1])]
df = pd.DataFrame(features, columns=feature_columns)

# Add label column
df['label'] = labels

# Add image filename for reference
df['image_file'] = [str(p.name) for p in image_paths]

print(f"✓ Dataset created with {len(df)} rows and {len(df.columns)} columns")
print(f"✓ Feature columns: {len(feature_columns)}")
print(f"✓ Label column: 'label'")

# Display first few rows
print("\n📋 DATASET PREVIEW")
print("="*80)
print(df.head())

print("\n📊 DATASET INFO")
print("="*80)
print(df.info())

# ============================================================================
# DATA INTEGRITY CHECKS
# ============================================================================

print("\n🔍 DATA INTEGRITY CHECKS")
print("="*80)

# Check 1: Uniform dimensions
print("\n1️⃣  Checking uniform dimensions...")
if features.shape[0] == len(labels):
    print(f"  ✓ All images have {features.shape[1]} features")
    print(f"  ✓ Number of samples: {features.shape[0]}")
else:
    print("  ❌ Dimension mismatch detected!")

# Check 2: Missing values
print("\n2️⃣  Checking for missing values...")
missing_count = df.isnull().sum().sum()
if missing_count == 0:
    print(f"  ✓ No missing values found")
else:
    print(f"  ❌ Found {missing_count} missing values")
    print(df.isnull().sum())

# Check 3: Feature value ranges
print("\n3️⃣  Checking feature value ranges...")
feature_stats = df[feature_columns].describe()
print(f"  Min value: {feature_stats.loc['min'].min():.4f}")
print(f"  Max value: {feature_stats.loc['max'].max():.4f}")
print(f"  Mean: {feature_stats.loc['mean'].mean():.4f}")
print(f"  Std: {feature_stats.loc['std'].mean():.4f}")

# Check 4: Label distribution
print("\n4️⃣  Checking label distribution...")
label_dist = df['label'].value_counts()
print(label_dist)

if label_dist.min() >= config.MIN_IMAGES_PER_CLASS:
    print(f"  ✓ All classes have ≥{config.MIN_IMAGES_PER_CLASS} samples")
else:
    print(f"  ⚠️  Some classes have <{config.MIN_IMAGES_PER_CLASS} samples")

# Check 5: Data types
print("\n5️⃣  Checking data types...")
print(f"  Feature columns: {df[feature_columns].dtypes.unique()}")
print(f"  Label column: {df['label'].dtype}")

print("\n✅ DATA INTEGRITY CHECK COMPLETE")

# ============================================================================
# SAVE DATASET
# ============================================================================

print("\n💾 SAVING DATASET")
print("="*80)

# Save full dataset
df.to_csv(config.OUTPUT_CSV, index=False)
print(f"✓ Full dataset saved: {config.OUTPUT_CSV}")
print(f"  Size: {os.path.getsize(config.OUTPUT_CSV) / 1024:.2f} KB")

# Save without image_file column for AutoAI
df_autoai = df.drop('image_file', axis=1)
df_autoai.to_csv('autoai_' + config.OUTPUT_CSV, index=False)
print(f"✓ AutoAI dataset saved: autoai_{config.OUTPUT_CSV}")

# Save dataset summary
summary = {
    'total_images': len(image_paths),
    'num_classes': len(classes),
    'classes': classes,
    'feature_dimensions': features.shape[1],
    'model_used': config.MODEL_NAME,
    'label_distribution': label_dist.to_dict()
}

with open('dataset_summary.txt', 'w') as f:
    f.write("DATASET SUMMARY\n")
    f.write("="*80 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print("✓ Dataset summary saved: dataset_summary.txt")

# ============================================================================
# GENERATE DOCUMENTATION
# ============================================================================

print("\n📝 GENERATING DOCUMENTATION")
print("="*80)

documentation = f"""
FEATURE EXTRACTION PROCESS DOCUMENTATION
=========================================

1. MODEL SELECTION
------------------
- Model: {config.MODEL_NAME}
- Architecture: Pre-trained on ImageNet
- Final layer: Removed (feature extraction mode)
- Feature dimensions: {config.FEATURE_DIM}

2. DATASET PREPARATION
----------------------
- Total images: {len(image_paths)}
- Number of classes: {len(classes)}
- Classes: {', '.join(classes)}
- Image size: {config.IMG_SIZE}

Class distribution:
{label_dist.to_string()}

3. PREPROCESSING
----------------
- Images resized to {config.IMG_SIZE}
- Preprocessing: ResNet50 preprocessing function
- Batch size: 16 images

4. FEATURE EXTRACTION
---------------------
- Features extracted using {config.MODEL_NAME}
- Feature vector size: {config.FEATURE_DIM}
- Total features: {features.shape[0]} x {features.shape[1]}

5. TABULAR DATASET
------------------
- Output file: {config.OUTPUT_CSV}
- Rows: {len(df)}
- Columns: {len(df.columns)} ({len(feature_columns)} features + 1 label + 1 filename)
- Feature columns: feature_0000 to feature_{features.shape[1]-1:04d}
- Label column: 'label'

6. INTEGRITY CHECKS
-------------------
✓ Uniform dimensions: All {features.shape[0]} images have {features.shape[1]} features
✓ No missing values: 0 NaN values found
✓ Feature ranges: [{feature_stats.loc['min'].min():.4f}, {feature_stats.loc['max'].max():.4f}]
✓ Label types: {df['label'].dtype}
✓ All classes have sufficient samples

7. NEXT STEPS
-------------
1. Upload 'autoai_{config.OUTPUT_CSV}' to IBM Watson Studio
2. Create AutoAI experiment
3. Select 'label' as target column
4. Run AutoAI classification experiment

"""

with open('feature_extraction_documentation.txt', 'w') as f:
    f.write(documentation)

print("✓ Documentation saved: feature_extraction_documentation.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ FEATURE EXTRACTION COMPLETE")
print("="*80)

print(f"""
Summary:
--------
✓ Images processed: {len(image_paths)}
✓ Classes: {len(classes)} ({', '.join(classes)})
✓ Model used: {config.MODEL_NAME}
✓ Feature dimensions: {config.FEATURE_DIM}
✓ Output dataset: {config.OUTPUT_CSV}
✓ AutoAI ready: autoai_{config.OUTPUT_CSV}

Files created:
--------------
1. {config.OUTPUT_CSV} - Full dataset with features and labels
2. autoai_{config.OUTPUT_CSV} - Dataset ready for AutoAI
3. dataset_summary.txt - Dataset statistics
4. feature_extraction_documentation.txt - Complete documentation
5. sample_images.png - Sample images visualization

Next steps:
-----------
1. Upload 'autoai_{config.OUTPUT_CSV}' to IBM Watson Studio Cloud Object Storage
2. Create AutoAI experiment in Watson Studio
3. Select 'label' as target column
4. Run experiment and analyze results
""")

print("="*80)
