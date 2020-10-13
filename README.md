# Mashup-topic-number-test
测试Mashup服务主题个数设置对算法结果的影响。<br>
LDA+SC：通过LDA主题模型对服务建模，采用谱聚类方法聚类。在LDA文件夹LDA_K.py中实现<br>
NMF+SC：通过传统非负矩阵分解对Mashup服务主机建模，采用谱聚类方法聚类。<br>
TWE-NMF+SC：我们所提方法,在非负矩阵分解中综合词嵌入信息和TCSW方法计算的单词权重信息对Mashup服务进行主题建模，采用谱聚类的方式对最后结果聚类。<br>

