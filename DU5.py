import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

expression_df = pd.read_csv("gene_expression.csv", sep=';', decimal=',', header=None)
labels_df = pd.read_csv("label.csv", header=None)

n_samples = min(len(expression_df), len(labels_df))
expression_df = expression_df.iloc[:n_samples]
labels_df = labels_df.iloc[:n_samples]
class_labels = labels_df.iloc[:, 0].map({1: "ALL", 2: "AML"})

scaler = StandardScaler()
scaled_X = scaler.fit_transform(expression_df)

train_X, test_X, train_y, test_y = train_test_split(
    scaled_X, class_labels, test_size=0.3, random_state=42, stratify=class_labels
)

classifier = DecisionTreeClassifier(max_features='sqrt', min_samples_split=5, min_samples_leaf=3, random_state=42)
classifier.fit(train_X, train_y)

cv_results_main = cross_val_score(classifier, scaled_X, class_labels, cv=5) 
print(f"\nEstimate Accuracy: {cv_results_main.mean():.2f} ± {cv_results_main.std():.2f}")

training_accuracy = classifier.score(train_X, train_y)
print(f"Training Accuracy: {training_accuracy:.2f}")

test_predictions = classifier.predict(test_X)
print("\nTest Set Performance:")
print(classification_report(test_y, test_predictions))

plt.figure(figsize=(20, 10))
plot_tree(classifier, filled=True, max_depth=2, class_names=classifier.classes_)
plt.title("Decision Tree")
plt.show()

feature_importances = classifier.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

print("Top 5 gene indexes by importance:")
for i in range(5):
    print(f"x[{sorted_indices[i]}] (importance = {feature_importances[sorted_indices[i]]:.4f})")

gene_names_series = pd.read_csv("geneNames.txt", header=None)

if len(gene_names_series) > sorted_indices[0]:
    most_imp_gene_name = gene_names_series.iloc[sorted_indices[0], 0]

print(f"\nMost important gene index: {sorted_indices[0]}")
print(f"Most important gene name: {most_imp_gene_name}")

pca_model = PCA()
pca_model.fit(scaled_X)

pca_components = pca_model.components_

print(f"\nPCA matrix V shape: {pca_components.shape}")
print("First 3 PCA components:")
print(pca_components[:3])  # first 3 vectors

n_components_list = [2, 5, 10, 20]

for n_components in n_components_list:
    print(f"\n--- K = {n_components} PCA Components ---")
    X_reduced = pca_model.transform(scaled_X)[:, :n_components]
    classifier_k = DecisionTreeClassifier(random_state=42)
    cv_results_k = cross_val_score(classifier_k, X_reduced, class_labels, cv=5)
    print(f"K = {n_components} → CV Accuracy: {cv_results_k.mean():.2f} ± {cv_results_k.std():.2f}")

# best K = 2
X_reduced_2 = pca_model.transform(scaled_X)[:, :2]

# Final tree
final_classifier = DecisionTreeClassifier(random_state=42)
final_classifier.fit(X_reduced_2, class_labels)

# Cross-validate
cv_results_final = cross_val_score(final_classifier, X_reduced_2, class_labels, cv=5)

print(f"\nFinal Model K=2 Estimated Accuracy: {cv_results_final.mean():.2f} ± {cv_results_final.std():.2f}")

# Get components PC1 and PC2
pc1_vector = pca_components[0]  
pc2_vector = pca_components[1]

gene_coords_2d = np.vstack([pc1_vector, pc2_vector]).T

# KMeans clustering
kmeans_model = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans_model.fit_predict(gene_coords_2d)

# Find the "active" cluster
cluster0_distances = np.linalg.norm(gene_coords_2d[cluster_labels == 0], axis=1)
cluster0_mean_distance = cluster0_distances.mean()
cluster1_distances = np.linalg.norm(gene_coords_2d[cluster_labels == 1], axis=1)
cluster1_mean_distance = cluster1_distances.mean()
active_cluster_label = 0 if cluster0_mean_distance > cluster1_mean_distance else 1

selected_gene_indices = np.where(cluster_labels == active_cluster_label)[0]
selected_gene_names = gene_names_series.iloc[selected_gene_indices, 0].tolist()

print("\nTop 10 genes in active PCA cluster:")
for name in selected_gene_names[:10]:
    print(name)