import model.TF_IDFAdapter as tfidf
import model.ldaAdapter as ldaa
import numpy as np
import cluster.kmeans as kmn
import data.data_util as du
import cluster.cluster_result as cr
import model.word2vecAdapter as w2v


# 调用使用gibbs的lda模型
def clusterResult_gibbs(k, topic, model, doc, num=5, sim_num=3, iterator=500):
    # 此时doc为原文本(经过预处理分词)
    # 对文本进行处理，获取tf——idf，使用w2v进行扩容
    # 预计取前5个，扩容为3个
    # num 为 keyword数量
    # sim_num 为 扩容数
    # model = w2v.load_model_binary(r"E:\学校\快乐推荐\word2vec\saveVec")
    print("拓展文档语料")
    doc = tfidf.expend_word(model, doc, num, sim_num)

    # 返回对应的聚类结果
    # 获取lda模型和词袋
    print("创建主题模型")
    word_list, r_model = ldaa.lda_model(doc, k, iterator)

    # 获取文档——主题分布
    doc_topic = r_model.doc_topic_

    # 转为普通list进行聚类
    doc_topic_list = np.array(doc_topic).tolist()
    estimator = kmn.kMeansByFeature(topic, doc_topic_list)
    labels = estimator.labels_

    return list(labels)


if __name__ == "__main__":
    r_k = 8
    r_topic = 8

    file_name = "data8.csv"
    doc = du.getDocAsWordArray(file_name, 3)
    # 获取标签信息

    former = du.getFormerCategory(file_name)
    model = w2v.load_model_binary(r"E:\学校\快乐推荐\word2vec\api_saveVec")

    result = clusterResult_gibbs(r_k, r_topic, model, doc)
    result_list = cr.printResult(r_topic, result, former)
    print(result_list)

