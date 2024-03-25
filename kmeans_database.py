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

# 连接到MongoDB数据库
client = MongoClient('mongodb+srv://root:Anny12345!@cluster0.sbvqogz.mongodb.net/?retryWrites=true&w=majority')

# 选择数据库和集合
db = client.Taiwan_travel
collection = db.data

col = collection.find()
documents1 = list(col)

# 将列表转换为 DataFrame（如果需要的话）
df = pd.DataFrame(documents1)

# 指定要查询的特定键为“评论”
specific_key = "評論"

# 查询并只返回特定键的值
cursor = collection.find({}, {specific_key: 1})

# 存储评论的列表
documents = []
# 对景點名稱进行标签编码
label_encoder = LabelEncoder()
df['景點名稱_label'] = label_encoder.fit_transform(df['景點名稱'])

# 指定要查询的特定键为“评论”
specific_key = "評論"

# 遍历结果集
for document in cursor:
    if specific_key in document and document[specific_key] != '':
        words = jieba.lcut(document[specific_key])
    else:
        words = []  # 如果评论为空，将其置为空列表
    documents.append(words)

# 将文本标记为 TaggedDocument 格式
tagged_documents = [TaggedDocument(doc, [index]) for index, doc in enumerate(documents)]

# 停用字文档的路径
stopword_file_path = './stopwords.txt'

# 定义一个空列表用于存储停用词
stopword = []

# 读取停用字文档并将其存储到 stopword 列表中
with open(stopword_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        stopword.append(line.strip())  # 去除每行末尾的换行符并添加到列表中

# 过滤停用词
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

# 构建词汇表
model.build_vocab(tagged_documents)

# 训练模型
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

# 查找索引为 0 的文档向量
vector = model.dv[0]

# 保存模型
model.save('doc2vec.model')

# 输出索引为 0 的文档向量
print("Vector for document at index 0:", vector)

# 数据预处理和K-means聚类
# 将数据转换为适合K-means的格式
df_kmeans = pd.DataFrame([model.dv[i] for i in range(len(tagged_documents))])

# 将数据标准化
scaler = StandardScaler()
df_kmeans_scaled = scaler.fit_transform(df_kmeans)


# 定义K-means模型
def k_means(n_clusters, your_data, true_labels):
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=30, random_state=0)
    kmeans_model.fit(your_data)
    c_labels = kmeans_model.labels_  # 获取聚类标签

    # 创建一个DataFrame，将聚类标签和景点名称放在一起
    result_df = pd.DataFrame({"景點名稱": true_labels, "cluster_label": c_labels})

    # 打印每个群组中的景点名称
    # for cluster_num in range(n_clusters):
    #     print(f"Cluster {cluster_num}:")
    #     print(result_df[result_df['cluster_label'] == cluster_num]['景點名稱'])


    # 将文档插入到相应的聚类集合中
    for i in range(n_clusters):
        cluster_collection_name = f"cluster_{i}"  # Dynamic collection name based on cluster number

        # Connect to the new database for storing cluster documents
        db_cluster = client.Taiwan_kmeans
        collection_cluster = db_cluster[cluster_collection_name]

        # Find documents with cluster label i
        for idx, label in enumerate(c_labels):
            if label == i:
                # Insert the document into the collection
                collection_cluster.insert_one(documents1[idx])
                
k_means(12, df_kmeans_scaled, df['景點名稱'])