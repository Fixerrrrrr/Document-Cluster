# coding=utf-8
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import klkmeans as klk
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy
import klagglomerative as kla

f = 'data.csv'


def clustering(inputs, alg, n_clusters, dis, fi=0.5):
    stops = open("stops.txt", "r").read().split("\r\n")
    tfidf = TfidfVectorizer(token_pattern=r"\b[a-z]+[a-z\-][a-z]+\b", stop_words=stops)
    weight = tfidf.fit_transform(inputs).toarray()

    if alg == 'Kmeans' and dis == 'Euclidean':
        kmeans = KMeans(n_clusters)
        kmeans.fit(weight)
        return list(kmeans.labels_), weight

    if alg == 'Kmeans' and dis == 'KL':
        C, assign = klk.klkmeans(weight, n_clusters, fi)
        return list(assign), weight

    if alg == "Agg" and dis == "Euclidean":
        model = AgglomerativeClustering(n_clusters,affinity="", linkage='complete')
        model.fit(weight)
        return list(model.labels_), weight

    if alg == "Agg" and dis == "KL":
        hc = kla.Hierarchical_Clustering(weight, n_clusters, fi)
        hc.initialize()
        current_clusters = hc.hierarchical_clustering()
        return current_clusters, weight


def read_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna(axis=0, how="any")
    title = list(df['title'])
    authors = list(df['authors'])
    groups = list(df['groups'])
    keywords = list(df['keywords'])
    topics = list(df['topics'])
    abstract = list(df['abstract'])
    return df.shape[0], title, authors, groups, keywords, topics, abstract


# 计算准确率
def eva(assign, labels, cls, n):
    lbs = []
    len_lbs = []
    for _n in range(n):
        lbs.append([])
    for i in range(len(assign)):
        lbs[assign[i]].append(labels[i])
    for _n in range(n):
        len_lbs.append(len(lbs[_n]))

    # print len_lbs
    count = np.zeros((n, len(cls)))
    for x in range(n):
        for y in lbs[x]:
            for z in y:
                for i in range(len(cls)):
                    if z == cls[i]:
                        count[x][i] += 1
    acc = 0
    for _n in range(n):
        l = list(count[_n])
        w = float(len_lbs[_n])/sum(len_lbs)
        # if sum(l) != 0:
        acc += w * max(l) / sum(l)

    return acc


# 可视化
def show(outputs, weight):
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(weight)

    x = []
    y = []

    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=outputs, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    # plt.savefig('./sample.png', aspect=1)


if __name__ == '__main__':
    # 预处理
    num, title, authors, groups, keywords, topics, abstract = read_data(f)
    inputs = []  # 输入文档
    cls = []  # 所有分类
    labels = []  # 每个文档的group
    for g in groups:
        labels.append(re.findall(r"[(](.*)[)]", g))

    for i in range(num):
        str = title[i] + ' ' + keywords[i] + ' ' + abstract[i]
        str = str.lower()
        inputs.append(str.decode("utf8", "ignore"))

    for i in range(num):
        for j in range(len(labels[i])):
            c = 0
            while c < len(cls):
                if cls[c] != labels[i][j]:
                    c += 1
                else:
                    break
            if c == len(cls):
                cls.append(labels[i][j])

    # print cls
    n_clusters = 3  # 设置聚类数量

    outputs, tfidf = clustering(inputs, "Agg", n_clusters, "KL")
    print eva(outputs, labels, cls, n_clusters)
    # show(outputs, tfidf)
