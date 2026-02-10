import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def run_classification_task(pickle_file_path):

    # Load the data from the pickle file
    with open(pickle_file_path, 'rb') as handle:
        data = pickle.load(handle)

    # If the data is not already a numpy array, convert it
    if not isinstance(data, np.ndarray):
        embeddings = np.array(data['embeddings'])
        labels = np.array(data['labels'])

    # Filter the embeddings and labels to keep only "tumorali" or "non_tumorali"
    # ignore no_annotations
    mask = (labels == "tumorali") | (labels == "non_tumorali")
    embeddings = embeddings[mask]
    labels = labels[mask]

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Separate the two classes
    embeddings_tumorali = embeddings[labels == "tumorali"]
    labels_tumorali = labels[labels == "tumorali"]
    embeddings_non_tumorali = embeddings[labels == "non_tumorali"]
    labels_non_tumorali = labels[labels == "non_tumorali"]

    # Print the shape information for the separated arrays
    print(f"Shape of embeddings_tumorali: {embeddings_tumorali.shape}")
    print(f"Shape of labels_tumorali: {labels_tumorali.shape}")
    print(f"Shape of embeddings_non_tumorali: {embeddings_non_tumorali.shape}")
    print(f"Shape of labels_non_tumorali: {labels_non_tumorali.shape}")

    # 46000 is about the size of the non_tumorali patches. Make the dataset balanced
    embeddings_tumorali_resampled, labels_tumorali_resampled = resample(
        embeddings_tumorali, labels_tumorali, n_samples=46000, random_state=42)
    embeddings_non_tumorali_resampled, labels_non_tumorali_resampled = resample(
        embeddings_non_tumorali, labels_non_tumorali, n_samples=46000, random_state=42)

    # Combine the resampled data
    resampled_embeddings = np.vstack((embeddings_tumorali_resampled, embeddings_non_tumorali_resampled))
    resampled_labels = np.hstack((labels_tumorali_resampled, labels_non_tumorali_resampled))

    # Encode the resampled labels
    resampled_labels_encoded = label_encoder.transform(resampled_labels)

    # Print the shape and type of the resampled numpy arrays
    print(f"Shape of the resampled embeddings array: {resampled_embeddings.shape}")
    print(f"Shape of the resampled labels array: {resampled_labels.shape}")

    """

    resampled_embeddings = embeddings
    resampled_labels = labels
    resampled_labels_encoded = label_encoder.transform(resampled_labels)


    # Print the shape and type of the resampled numpy arrays
    print(f"Shape of the resampled embeddings array: {resampled_embeddings.shape}")
    print(f"Shape of the resampled labels array: {resampled_labels.shape}")

    # ----- ATTENTION ----- run this cell to see the results with the imbalanced dataset

    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(resampled_embeddings, resampled_labels_encoded, test_size=0.2, random_state=42)

    # Apply PCA to keep 99% of the variance
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Print the number of features kept after PCA
    n_features_kept = X_train_pca.shape[1]
    print(f"Number of features kept after PCA: {n_features_kept}")

    # Define the parameter grid for the RandomizedSearchCV
    random_grid_knn = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

    # Set up the RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=KNeighborsClassifier(),
        param_distributions=random_grid_knn,
        scoring="balanced_accuracy",
        n_iter=50,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model
    search.fit(X_train_pca, y_train)

    # Get the best estimator
    clf = search.best_estimator_

    # Predict on the test set with a progress bar
    y_pred = []
    for i in tqdm(range(len(X_test_pca)), desc="Predicting"):
        y_pred.append(clf.predict(X_test_pca[i].reshape(1, -1)))

    y_pred = np.array(y_pred).flatten()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()