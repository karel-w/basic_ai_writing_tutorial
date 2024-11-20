from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score

## load embeddings

# Load embeddings (remains the same)
def load_embeddings(embedding_file, label_file):
    """Load embeddings and labels from disk"""
    embeddings = np.load(embedding_file)
    labels = np.load(label_file)
    return embeddings, labels

train_embeddings, train_labels = load_embeddings('embeddings/train_embeddings_20241119_2227.npy', 'embeddings/train_labels_20241119_2227.npy')
val_embeddings, val_labels = load_embeddings('embeddings/valid_embeddings_20241119_2235.npy', 'embeddings/valid_labels_20241119_2235.npy')

#RF model
def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=20, min_samples_split=5):
    # Initialize and train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=-1  # Use all available cores
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Print results
    print("Random Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_model

def train_svm(X_train, X_test, y_train, y_test, kernel='rbf', c='1', gamma='scale'):
    # Initialize and train SVM
    svm_model = SVC(
        kernel='rbf',  # You can try 'linear', 'poly', or 'sigmoid'
        C = c,
        gamma = gamma,
        random_state=42,
        probability=True,
    )
    
    svm_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm_model.predict(X_test)
    
    # Print results
    print("SVM Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return svm_model

#perform grid search
#For Random Forest
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(n_jobs=-1), rf_params, cv=5)
rf_grid.fit(train_embeddings, train_labels)
print("Best RF parameters:", rf_grid.best_params_)

rf_model = train_random_forest(train_embeddings, val_embeddings, train_labels, val_labels, 
                               n_estimators=rf_grid.best_params_['n_estimators'],
                               max_depth=rf_grid.best_params_['max_depth'],
                               min_samples_split=rf_grid.best_params_['min_samples_split'])
y_pred = rf_model.predict(val_embeddings)

# For SVM
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

svm_grid = GridSearchCV(SVC(), svm_params, cv=5)
svm_grid.fit(train_embeddings, train_labels)
print("Best SVM parameters:", svm_grid.best_params_)

svm_model = train_svm(train_embeddings, val_embeddings, train_labels, val_labels,
                      svm_grid.best_params_['kernel'], 
                      svm_grid.best_params_['C'],
                      svm_grid.best_params_['gamma'])

def plot_model_comparison(models_dict, X_test, y_test, figsize=(15, 10)):
    """
    Compare multiple models using ROC curves and Precision-Recall curves
    
    Parameters:
    models_dict: dictionary of format {'model_name': model_object}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ROC Curve
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    
    # Precision-Recall Curve
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('ROC_AUC.png')

def plot_confusion_matrices(models_dict, X_test, y_test, figsize=(12, 5)):
    """
    Plot confusion matrices for multiple models
    """
    fig, axes = plt.subplots(1, len(models_dict), figsize=figsize)
    
    for i, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('ConfusionMatrix.png')

def plot_prediction_distribution(models_dict, X_test, y_test, figsize=(12, 5)):
    """
    Plot distribution of prediction probabilities for each class
    """
    fig, axes = plt.subplots(1, len(models_dict), figsize=figsize)
    
    for i, (name, model) in enumerate(models_dict.items()):
        y_pred_proba = model.predict_proba(X_test)
        
        for j in range(y_pred_proba.shape[1]):
            sns.kdeplot(data=y_pred_proba[:, j][y_test == j], 
                       ax=axes[i], 
                       label=f'Class {j}')
        
        axes[i].set_title(f'{name} Prediction Distribution')
        axes[i].set_xlabel('Prediction Probability')
        axes[i].set_ylabel('Density')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('Class_distribution.png')

def plot_cross_validation_comparison(models_dict, X, y, cv=5, figsize=(10, 6)):
    """
    Compare cross-validation scores across models
    """
    cv_scores = {}
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=cv)
        cv_scores[name] = scores
    
    plt.figure(figsize=figsize)
    plt.boxplot([scores for scores in cv_scores.values()], labels=cv_scores.keys())
    plt.title('Cross-validation Score Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('CrossValidation.png')

# After training your models:
models = {
    'Random Forest': rf_model,
    'SVM': svm_model
}

# # Plot comparisons
plot_model_comparison(models, val_embeddings, val_labels)
plot_confusion_matrices(models, val_embeddings, val_labels)
plot_prediction_distribution(models, val_embeddings, val_labels)
plot_cross_validation_comparison(models, val_embeddings, val_labels)
