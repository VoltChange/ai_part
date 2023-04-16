import jieba
from snownlp import sentiment, SnowNLP


def handle(self, doc):
    words = jieba.lcut(doc)  ##原本使用的是snownlp自带的seg分词功能，words = seg.seg(doc) 替换为jieba.lcut
    return words


def analyse(string):
    sentiment.Sentiment.handle = handle  # 重写handle，用结巴分词
    sent = sentiment.Sentiment()
    words_list = sentiment.Sentiment.handle(sent, string)
    return SnowNLP(string).sentiments
