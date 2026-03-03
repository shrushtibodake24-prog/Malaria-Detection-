import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Check and install required packages
try:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    import seaborn as sns
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Installing missing packages...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "scikit-learn"])
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
        import seaborn as sns
        print("Packages installed successfully!")
    except Exception as install_error:
        print(f"Error installing packages: {install_error}")
        print("Please activate the virtual environment first:")
        print("  .\\tf_env\\Scripts\\Activate.ps1")
        print("Then install dependencies:")
        print("  python -m pip install tensorflow scikit-learn matplotlib seaborn numpy")
        sys.exit(1)

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

# Load test data
print("="*70)
print("Loading Test Data")
print("="*70)
processed_data_dir = "processed_data"

try:
    # Try to load X_test.npy and y_test.npy, if not found use X_val and y_val
    if os.path.exists(os.path.join(processed_data_dir, "X_test.npy")):
        X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))
        print("Loaded X_test.npy and y_test.npy")
    else:
        # Use validation set as test set
        X_test = np.load(os.path.join(processed_data_dir, "X_val.npy"))
        y_test = np.load(os.path.join(processed_data_dir, "y_val.npy"))
        print("Using X_val.npy and y_val.npy as test set")
    
    print(f"\nTest data loaded successfully!")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Data type: {X_test.dtype}")
    print(f"  Value range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"  Number of samples: {len(X_test)}")
    print(f"  Class distribution: Parasitized={np.sum(y_test)}, Uninfected={len(y_test)-np.sum(y_test)}")
    
except FileNotFoundError as e:
    print(f"Error: Test data files not found. {e}")
    print("Please run preprocess_data.py first to generate the data files.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading test data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Function to load model from multiple possible locations
def load_model(model_name):
    """Load model from ../models/ or local models/ directory"""
    # Try ../models/ first
    parent_dir = os.path.dirname(os.path.abspath('.'))
    parent_models_path = os.path.join(parent_dir, "models", model_name)
    
    # Try local models/ directory
    local_models_path = os.path.join("models", model_name)
    
    if os.path.exists(parent_models_path):
        print(f"  Loading from: {parent_models_path}")
        return tf.keras.models.load_model(parent_models_path)
    elif os.path.exists(local_models_path):
        print(f"  Loading from: {local_models_path}")
        return tf.keras.models.load_model(local_models_path)
    else:
        raise FileNotFoundError(f"Model {model_name} not found in ../models/ or models/ directory")

# Load all three models
print("\n" + "="*70)
print("Loading Models")
print("="*70)

models = {}
model_names = {
    'custom_cnn': 'custom_cnn_model.h5',
    'mobilenetv2': 'mobilenetv2_model.h5',
    'efficientnetb0': 'efficientnetb0_model.h5'
}

for model_key, model_file in model_names.items():
    print(f"\nLoading {model_key.upper()} model...")
    try:
        models[model_key] = load_model(model_file)
        print(f"  ✓ {model_key.upper()} model loaded successfully!")
        print(f"  Input shape: {models[model_key].input_shape}")
        print(f"  Output shape: {models[model_key].output_shape}")
    except Exception as e:
        print(f"  ✗ Error loading {model_key.upper()} model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Function to evaluate a model
def evaluate_model(model, model_name, X_test, y_test):
    """Evaluate model and return metrics and predictions"""
    print(f"\nEvaluating {model_name.upper()}...")
    
    # Make predictions
    print("  Making predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Evaluate all models
print("\n" + "="*70)
print("Evaluating All Models")
print("="*70)

results = {}
for model_key, model in models.items():
    results[model_key] = evaluate_model(model, model_key, X_test, y_test)

# Print metrics for all models
print("\n" + "="*70)
print("Evaluation Results Summary")
print("="*70)

metrics_table = []
for model_key in model_names.keys():
    r = results[model_key]
    metrics_table.append({
        'Model': model_key.upper(),
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1_score'],
        'ROC-AUC': r['roc_auc']
    })
    print(f"\n{model_key.upper()} Model:")
    print(f"  Accuracy:  {r['accuracy']:.4f}")
    print(f"  Precision: {r['precision']:.4f}")
    print(f"  Recall:    {r['recall']:.4f}")
    print(f"  F1-Score:  {r['f1_score']:.4f}")
    print(f"  ROC-AUC:   {r['roc_auc']:.4f}")

# Generate confusion matrices for all models
print("\n" + "="*70)
print("Generating Confusion Matrices")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)

for idx, (model_key, ax) in enumerate(zip(model_names.keys(), axes)):
    cm = results[model_key]['confusion_matrix']
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Uninfected', 'Parasitized'],
                yticklabels=['Uninfected', 'Parasitized'],
                cbar_kws={'label': 'Count'})
    
    ax.set_title(f'{model_key.upper()}\nAccuracy: {results[model_key]["accuracy"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    
    # Add text annotations for metrics
    tn, fp, fn, tp = cm.ravel()
    textstr = f'TN: {tn}\nFP: {fp}\nFN: {fn}\nTP: {tp}'
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('confusion_matrices_all_models.png', dpi=150, bbox_inches='tight')
print("Confusion matrices saved to: confusion_matrices_all_models.png")
plt.show()

# Generate ROC curves for all models
print("\n" + "="*70)
print("Generating ROC Curves")
print("="*70)

plt.figure(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, (model_key, color) in enumerate(zip(model_names.keys(), colors)):
    r = results[model_key]
    plt.plot(r['fpr'], r['tpr'], 
             color=color, lw=2, 
             label=f"{model_key.upper()} (AUC = {r['roc_auc']:.4f})")

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier (AUC = 0.5000)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_all_models.png', dpi=150, bbox_inches='tight')
print("ROC curves saved to: roc_curves_all_models.png")
plt.show()

# Generate detailed classification reports
print("\n" + "="*70)
print("Detailed Classification Reports")
print("="*70)

for model_key in model_names.keys():
    r = results[model_key]
    print(f"\n{model_key.upper()} Model Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, r['y_pred'], 
                                target_names=['Uninfected', 'Parasitized'],
                                digits=4))

# Create comparison table visualization
print("\n" + "="*70)
print("Creating Comparison Table")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for model_key in model_names.keys():
    r = results[model_key]
    table_data.append([
        model_key.upper(),
        f"{r['accuracy']:.4f}",
        f"{r['precision']:.4f}",
        f"{r['recall']:.4f}",
        f"{r['f1_score']:.4f}",
        f"{r['roc_auc']:.4f}"
    ])

table = ax.table(cellText=table_data, colLabels=headers,
                 cellLoc='center', loc='center',
                 colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style the header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style the cells
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        if j > 0:  # Color code the metric columns
            value = float(table_data[i-1][j])
            if value >= 0.9:
                table[(i, j)].set_facecolor('#c8e6c9')
            elif value >= 0.8:
                table[(i, j)].set_facecolor('#fff9c4')
            else:
                table[(i, j)].set_facecolor('#ffcdd2')

plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
plt.savefig('model_comparison_table.png', dpi=150, bbox_inches='tight')
print("Comparison table saved to: model_comparison_table.png")
plt.show()

# Print final summary
print("\n" + "="*70)
print("Evaluation Complete!")
print("="*70)
print("\nGenerated Files:")
print("  1. confusion_matrices_all_models.png")
print("  2. roc_curves_all_models.png")
print("  3. model_comparison_table.png")
print("\nBest Performing Model:")
best_model = max(model_names.keys(), key=lambda k: results[k]['f1_score'])
best_result = results[best_model]
print(f"  Model: {best_model.upper()}")
print(f"  F1-Score: {best_result['f1_score']:.4f}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print("="*70)
