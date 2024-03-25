from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import jieba
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 連接MongoDB
client = MongoClient('mongodb+srv://root:Anny12345!@cluster0.sbvqogz.mongodb.net/?retryWrites=true&w=majority')

# 選擇數據庫
db = client.Taiwan_travel
collection = db.data

col = collection.find()
documents = list(col)

# 將列表轉 DataFrame（如果需要的话）
df = pd.DataFrame(documents)

# 指定要找的key
specific_key = "評論"

# 查詢某key
cursor = collection.find({}, {specific_key: 1})

# 儲存評論的列表
documents = []
# label景點
label_encoder = LabelEncoder()
df['景點名稱_label'] = label_encoder.fit_transform(df['景點名稱'])


# 儲存評論的列表
documents = []


for document in cursor:
    if specific_key in document and document[specific_key] != '':
        words = jieba.lcut(document[specific_key])
    else:
        words = []  # 如果評論為空，append[]
    documents.append(words)

# 將文本標記 TaggedDocument 格式
tagged_documents = [TaggedDocument(doc, [index]) for index, doc in enumerate(documents)]

# 停用字文黨的路徑
stopword_file_path = './stopwords.txt'

# 定義一个空列表用於儲存停用词
stopword = []

# 讀取停用字文黨並將其儲存到 stopword 列表中
with open(stopword_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        stopword.append(line.strip())  # 去除每行末尾的换行符并添加到列表中

# 過濾停用词
filtered_documents = []
for doc in documents:
    filtered_doc = [word for word in doc if word not in stopword]
    filtered_documents.append(filtered_doc)

# 建立 Doc2Vec 模型
model = Doc2Vec(
    vector_size=3, 
    window=2, 
    min_count=1,
    epochs=20
)

# 建構詞表
model.build_vocab(tagged_documents)

# 訓練模型
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)


vector = model.dv[0]

# 保存模型
model.save('doc2vec.model')

# 輸出索引0的向量
print("Vector for document at index 0:", vector)

# 數據欲處理和K-means分群
# 將數據轉成適合K-means的格式
df_kmeans = pd.DataFrame([model.dv[i] for i in range(len(tagged_documents))])

# 將數據標準化
scaler = StandardScaler()
df_kmeans_scaled = scaler.fit_transform(df_kmeans)

#--------------------------------k-means分群
ks = range(1, 10)
inertias = [] # innertias = distance

# 找尋最適合的k
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_kmeans_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(10, 6))
plt.style.use("bmh")
plt.plot(ks, inertias, "-o")
plt.xlabel("K value")
plt.ylabel("Inertias")
plt.xticks(ks)
plt.show()

# 定義K-means模型
def k_means(n_clusters, your_data, true_labels):
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=30, random_state=0)
    kmeans_model.fit(your_data)
    c_labels = kmeans_model.labels_

    # 創建DataFrame，將聚類標籤和景点名稱放在一起
    result_df = pd.DataFrame({"景點名稱": true_labels, "cluster_label": c_labels})

    # print出每个群组中的景点名稱
    for cluster_num in range(n_clusters):
        print(f"Cluster {cluster_num}:")
        print(result_df[result_df['cluster_label'] == cluster_num]['景點名稱'])

    # 計算輪廓係數
    sil_avg = silhouette_score(your_data, c_labels)
    print("Average silhouette score:", sil_avg)

    # 計算每個群的輪廓係數
    sample_silhouette_values = silhouette_samples(your_data, c_labels)
    means_list = []
    for label in range(n_clusters):
        means_list.append(sample_silhouette_values[c_labels == label].mean())

    print("Silhouette score for each cluster:", means_list)

# 用我寫的kmeans func
k_means(100, df_kmeans_scaled, df['景點名稱'])
