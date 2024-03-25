import pandas as pd
import numpy as np
import json
import os
import jieba
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt

# Define function to read raw data
def get_rawdata(folder_path="./test"):
    result_list = []
    rawdata_file_names = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    for file_name in rawdata_file_names:
        try:
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                data = json.load(file)
                result_list.extend(data)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return result_list

# Load data
documents = get_rawdata()
df = pd.DataFrame(documents)

# Prepare stopwords
stopword_file_path = './stopwords.txt'
stop_words = [line.strip() for line in open(stopword_file_path, 'r', encoding='utf-8')]

# Process comments and tags
comment_documents, tag_documents = [], []
label_encoder = LabelEncoder()
df['景點名稱_label'] = label_encoder.fit_transform(df['景點名稱'])

for document in documents:
    words = jieba.lcut(document.get("評論", ""))
    filtered_words = [word for word in words if word.strip() not in stop_words]
    comment_documents.append(TaggedDocument(filtered_words, [len(comment_documents)]))
    
    tags = document.get("tag", "").split(",")
    filtered_tags = [tag.strip() for tag in tags if tag.strip() not in stop_words]
    tag_documents.append(filtered_tags)

# Train Doc2Vec (for comments)
comment_model = Doc2Vec(vector_size=100, window=5, min_count=1, epochs=20)
comment_model.build_vocab(comment_documents)
comment_model.train(comment_documents, total_examples=comment_model.corpus_count, epochs=comment_model.epochs)
comment_vectors = [comment_model.dv[i] for i in range(len(comment_documents))]

# Train Word2Vec (for tags)
tag_model = Word2Vec(tag_documents, vector_size=100, window=2, sg=1, min_count=1, seed=42, epochs=20)
tag_vectors_weighted = pd.DataFrame(tag_model.wv.vectors * 0.3)  # Adjust weights as needed

# Merge comment vectors and weighted tag vectors
df_combined = pd.concat([df['景點名稱_label'].astype(str), pd.DataFrame(comment_vectors), tag_vectors_weighted], axis=1)
df_combined.columns = df_combined.columns.astype(str)

# Fill missing values and scale data
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
df_combined_filled = pd.DataFrame(imputer.fit_transform(df_combined), columns=df_combined.columns)
df_combined_scaled = scaler.fit_transform(df_combined_filled)

# Dimensionality reduction with PCA
pca = PCA(n_components=3)
df_reduced_pca = pca.fit_transform(df_combined_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=12, random_state=0)
clusters = kmeans.fit_predict(df_reduced_pca)

# Attach clusters back to the dataframe (optional for further analysis)
df['cluster'] = clusters

# Plotting the PCA-reduced data with cluster assignments
plt.figure(figsize=(10, 8))
plt.scatter(df_reduced_pca[:, 0], df_reduced_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5, edgecolor='k')
plt.title('PCA Reduced Data with KMeans Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster ID')
plt.show()


