# -*- coding: utf-8 -*-

!pip install scikit-surprise xgboost

# Install necessary libraries in Google Colab
# Uncomment the next line if libraries are not already installed
# !pip install scikit-surprise xgboost

# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split as surprise_train_test_split
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
u_data_path = "/content/u.data"
u_item_path = "/content/u.item"

u_data_columns = ["user_id", "item_id", "rating", "timestamp"]
u_data = pd.read_csv(u_data_path, sep="\t", names=u_data_columns, header=None)

u_item_columns = [
    "item_id", "title", "release_date", "video_release_date",
    "IMDb_url", "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]
u_item = pd.read_csv(u_item_path, sep="|", names=u_item_columns, header=None, encoding='latin-1')

# Step 1: Memory-Based Collaborative Filtering (User-Based CF)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(u_data[['user_id', 'item_id', 'rating']], reader)
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

sim_options = {'name': 'cosine', 'user_based': True}
user_based_model = KNNBasic(sim_options=sim_options)
user_based_model.fit(trainset)
predictions_user_cf = user_based_model.test(testset)

# Evaluation for User-Based CF
rmse_user_cf = np.sqrt(mean_squared_error([pred.r_ui for pred in predictions_user_cf],
                                          [pred.est for pred in predictions_user_cf]))
mae_user_cf = mean_absolute_error([pred.r_ui for pred in predictions_user_cf],
                                   [pred.est for pred in predictions_user_cf])
print(f"User-Based CF - RMSE: {rmse_user_cf}, MAE: {mae_user_cf}")

# Step 2: Model-Based Collaborative Filtering with SVD
svd_model = SVD()
svd_model.fit(trainset)
predictions_svd = svd_model.test(testset)

# Evaluation for SVD
rmse_svd = np.sqrt(mean_squared_error([pred.r_ui for pred in predictions_svd],
                                      [pred.est for pred in predictions_svd]))
mae_svd = mean_absolute_error([pred.r_ui for pred in predictions_svd],
                               [pred.est for pred in predictions_svd])
print(f"SVD - RMSE: {rmse_svd}, MAE: {mae_svd}")

# Step 3: Model-Based Collaborative Filtering with K-Means
user_item_matrix = u_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
kmeans = KMeans(n_clusters=10, random_state=42)
user_clusters = kmeans.fit_predict(user_item_matrix)

# Create a mapping from user_id to cluster
user_id_to_cluster = {user_id: cluster for user_id, cluster in zip(user_item_matrix.index, user_clusters)}

def predict_kmeans(row):
    cluster = user_id_to_cluster.get(row['user_id'], None)
    if cluster is None:
        return 0  # Default prediction for missing cluster
    cluster_data = u_data[u_data['cluster'] == cluster]
    cluster_avg = cluster_data.groupby('item_id')['rating'].mean()
    return cluster_avg.get(row['item_id'], 0)  # Default to 0 if item is not in cluster

# Predict ratings using K-Means
u_data['cluster'] = [user_clusters[u - 1] for u in u_data['user_id']]
testset_df = pd.DataFrame(testset, columns=["user_id", "item_id", "rating"])
testset_df['predicted'] = testset_df.apply(predict_kmeans, axis=1)

rmse_kmeans = np.sqrt(mean_squared_error(testset_df['rating'], testset_df['predicted']))
mae_kmeans = mean_absolute_error(testset_df['rating'], testset_df['predicted'])
print(f"K-Means - RMSE: {rmse_kmeans}, MAE: {mae_kmeans}")

# Step 4: Model-Based Collaborative Filtering with XGBoost
u_data_encoded = u_data.copy()
label_encoders = {}
for col in ['user_id', 'item_id']:
    le = LabelEncoder()
    u_data_encoded[col] = le.fit_transform(u_data_encoded[col])
    label_encoders[col] = le

X = u_data_encoded[['user_id', 'item_id']]
y = u_data_encoded['rating']
X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
mae_xgb = mean_absolute_error(y_test, y_pred)
print(f"XGBoost - RMSE: {rmse_xgb}, MAE: {mae_xgb}")

# Visualization for RMSE
models = ['User-Based CF', 'SVD', 'K-Means', 'XGBoost']
rmse_values = [rmse_user_cf, rmse_svd, rmse_kmeans, rmse_xgb]

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values)
plt.title('RMSE Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.ylim(0, max(rmse_values) + 0.5)
plt.show()

# Define Precision@k and Recall@k Calculation Function
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_predictions = {}
    for pred in predictions:
        if pred.uid not in user_predictions:
            user_predictions[pred.uid] = []
        user_predictions[pred.uid].append((pred.iid, pred.est, pred.r_ui >= threshold))

    precisions, recalls = [], []
    for uid, user_preds in user_predictions.items():
        user_preds.sort(key=lambda x: x[1], reverse=True)
        top_k = user_preds[:k]
        relevant_top_k = sum((1 for _, _, is_relevant in top_k if is_relevant))
        relevant_total = sum((1 for _, _, is_relevant in user_preds if is_relevant))

        precision = relevant_top_k / k
        recall = relevant_top_k / (relevant_total + 1e-8)  # Avoid division by zero
        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

# Compute Precision@10 and Recall@10 for User-Based CF
precision_user_cf, recall_user_cf = precision_recall_at_k(predictions_user_cf)

# Compute Precision@10 and Recall@10 for SVD
precision_svd, recall_svd = precision_recall_at_k(predictions_svd)

# Compute Precision@10 and Recall@10 for K-Means
def precision_recall_kmeans(testset_df, k=10, threshold=4.0):
    # Group by user and sort by predicted rating
    user_predictions = testset_df.groupby('user_id').apply(
        lambda group: group.sort_values('predicted', ascending=False).head(k)
    )

    precisions, recalls = [], []
    for user_id, group in user_predictions.groupby(level=0):
        top_k = group['predicted'] >= threshold
        relevant = group['rating'] >= threshold
        precision = top_k.sum() / k
        recall = top_k.sum() / relevant.sum()
        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

precision_kmeans, recall_kmeans = precision_recall_kmeans(testset_df)

# Compute Precision@10 and Recall@10 for XGBoost
def precision_recall_xgboost(X_test, y_test, y_pred, k=10, threshold=4.0):
    test_df = pd.DataFrame({'user_id': X_test['user_id'], 'item_id': X_test['item_id'],
                            'true_rating': y_test, 'predicted': y_pred})
    user_predictions = test_df.groupby('user_id').apply(
        lambda group: group.sort_values('predicted', ascending=False).head(k)
    )

    precisions, recalls = [], []
    for user_id, group in user_predictions.groupby(level=0):
        top_k = group['predicted'] >= threshold
        relevant = group['true_rating'] >= threshold
        precision = top_k.sum() / k
        recall = top_k.sum() / relevant.sum()
        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

precision_xgb, recall_xgb = precision_recall_xgboost(X_test, y_test, y_pred)

# Visualization for All Metrics
precision_values = [precision_user_cf, precision_svd, precision_kmeans, precision_xgb]
recall_values = [recall_user_cf, recall_svd, recall_kmeans, recall_xgb]

# Bar Chart for Precision@10
plt.figure(figsize=(10, 6))
plt.bar(models, precision_values, color='green')
plt.title('Precision@10 Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('Precision@10')
plt.ylim(0, max(precision_values) + 0.1)
plt.show()

# Bar Chart for Recall@10
plt.figure(figsize=(10, 6))
plt.bar(models, recall_values, color='purple')
plt.title('Recall@10 Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('Recall@10')
plt.ylim(0, max(recall_values) + 0.1)
plt.show()

# Precision vs Recall Plot
plt.figure(figsize=(10, 6))
for model, precision, recall in zip(models, precision_values, recall_values):
    plt.scatter(recall, precision, label=model, s=100)  # Scatter plot with size adjustment
plt.plot(recall_values, precision_values, linestyle='--', color='gray', alpha=0.5)  # Line connecting points
plt.title('Precision vs Recall Comparison')
plt.xlabel('Recall@10')
plt.ylabel('Precision@10')
plt.legend()
plt.grid()
plt.show()

# MAE Comparison Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(models, mae_values, color='orange')
plt.title('MAE Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('MAE')
plt.ylim(0, max(mae_values) + 0.1)
plt.show()

# Print Precision and Recall for each model
print(f"User-Based CF - Precision@10: {precision_user_cf:.4f}, Recall@10: {recall_user_cf:.4f}")
print(f"SVD - Precision@10: {precision_svd:.4f}, Recall@10: {recall_svd:.4f}")
print(f"K-Means - Precision@10: {precision_kmeans:.4f}, Recall@10: {recall_kmeans:.4f}")
print(f"XGBoost - Precision@10: {precision_xgb:.4f}, Recall@10: {recall_xgb:.4f}")

# Visualization for Precision@10
plt.figure(figsize=(10, 6))
plt.bar(models, precision_values, color='green')
plt.title('Precision@10 Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('Precision@10')
plt.ylim(0, max(precision_values) + 0.1)
plt.show()

# Visualization for Recall@10
plt.figure(figsize=(10, 6))
plt.bar(models, recall_values, color='purple')
plt.title('Recall@10 Comparison Across Models')
plt.xlabel('Model')
plt.ylabel('Recall@10')
plt.ylim(0, max(recall_values) + 0.1)
plt.show()

from sklearn.metrics.pairwise import cosine_similarity

# Compute the similarity matrix for user-item matrix
similarity_matrix = cosine_similarity(user_item_matrix)

# Convert to DataFrame for better readability (optional)
similarity_df = pd.DataFrame(similarity_matrix,
                             index=user_item_matrix.index,
                             columns=user_item_matrix.index)

# Display the similarity matrix
print("User-User Similarity Matrix:")
print(similarity_df)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot heatmap of user-user similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df.iloc[:30, :30], cmap='coolwarm', annot=False)
plt.title('User-User Similarity Heatmap (Sample)')
plt.xlabel('User ID')
plt.ylabel('User ID')
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use KMeans to cluster users based on the similarity matrix
n_clusters = 5  # Set initial number of clusters (can be optimized)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
user_clusters = kmeans.fit_predict(similarity_matrix)

# Add cluster labels to a DataFrame for analysis
user_cluster_df = pd.DataFrame({'user_id': user_item_matrix.index, 'cluster': user_clusters})

# Print cluster assignments
print("User Clusters:")
print(user_cluster_df.head())

# Evaluate clustering with silhouette score
silhouette_avg = silhouette_score(similarity_matrix, user_clusters)
print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

import seaborn as sns
import matplotlib.pyplot as plt

# Reduce dimensionality using PCA for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_similarity = pca.fit_transform(similarity_matrix)

# Plot clusters
plt.figure(figsize=(10, 8))
for cluster in range(n_clusters):
    cluster_points = reduced_similarity[user_clusters == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
plt.title('User Clusters Based on Similarity')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Compute average similarity for each user
average_similarity = similarity_matrix.mean(axis=1)

# Threshold for outlier detection (e.g., bottom 5%)
threshold = np.percentile(average_similarity, 5)
outliers = user_item_matrix.index[average_similarity < threshold]

print("Outliers (Users with low similarity):")
print(outliers)

# Analyze top movies in each cluster
for cluster in range(n_clusters):
    cluster_users = user_cluster_df[user_cluster_df['cluster'] == cluster]['user_id']
    cluster_data = u_data[u_data['user_id'].isin(cluster_users)]
    top_movies = cluster_data['item_id'].value_counts().head(5)
    print(f"Cluster {cluster} - Top Movies:")
    print(top_movies)

# Analyze ratings from outliers
outlier_data = u_data[u_data['user_id'].isin(outliers)]
print("Ratings from Outlier Users:")
print(outlier_data.head())

# Sort similarity matrix by cluster
sorted_indices = user_cluster_df.sort_values('cluster')['user_id']
sorted_similarity_matrix = similarity_df.loc[sorted_indices, sorted_indices]

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(sorted_similarity_matrix, cmap='coolwarm', annot=False)
plt.title('User Similarity Heatmap (Sorted by Clusters)')
plt.xlabel('User ID')
plt.ylabel('User ID')
plt.show()

# Map item_id to movie titles
movie_mapping = u_item.set_index('item_id')['title'].to_dict()

for cluster in range(n_clusters):
    cluster_users = user_cluster_df[user_cluster_df['cluster'] == cluster]['user_id']
    cluster_data = u_data[u_data['user_id'].isin(cluster_users)]
    top_movies = cluster_data['item_id'].value_counts().head(5).index.map(movie_mapping)
    print(f"Cluster {cluster} - Top Movies:")
    print(list(top_movies))

# Elbow Method for Optimal Clusters
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

sse = []
for k in range(2, 10):  # Experiment with different cluster numbers
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(similarity_matrix)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 10), sse, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()

# Re-run KMeans with 4 clusters
n_clusters_optimal = 4
kmeans_optimal = KMeans(n_clusters=n_clusters_optimal, random_state=42)
user_clusters_optimal = kmeans_optimal.fit_predict(similarity_matrix)

# Evaluate new clustering
silhouette_optimal = silhouette_score(similarity_matrix, user_clusters_optimal)
print(f"Silhouette Score for {n_clusters_optimal} clusters: {silhouette_optimal}")

# Visualize the new clusters
user_cluster_df_optimal = pd.DataFrame({'user_id': user_item_matrix.index, 'cluster': user_clusters_optimal})

# Recommend top movies to a specific cluster
cluster_to_recommend = 2
cluster_users = user_cluster_df[user_cluster_df['cluster'] == cluster_to_recommend]['user_id']
cluster_top_movies = u_data[u_data['user_id'].isin(cluster_users)]['item_id'].value_counts().head(10)

# Map item_id to titles
cluster_top_movies_titles = cluster_top_movies.index.map(movie_mapping)
print(f"Recommended Movies for Cluster {cluster_to_recommend}:")
print(list(cluster_top_movies_titles))

print("user item matirx: \n",user_item_matrix)

