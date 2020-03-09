import pyltp
from gensim.models import Word2Vec
from pyltp import Segmentor
import jieba
from gensim.models.word2vec import LineSentence
from pyltp import  SentenceSplitter,NamedEntityRecognizer,Postagger,Parser,Segmentor
from gensim import models
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba
import random
import os
from model.Bert import Bert
from config import *

# def cut(string):
#     return ' '.join(token for token in jieba.cut(''.join(re.sub('\\\\n|[\n\u3000\r]', '', string)))
#                     if token not in stopwords)


class SpeakDetect:
    def __init__(self):
        self.bert = Bert() # 用于情感分析

    def get_word_list(self, sentence, model):
        # 得到分词
        segmentor = Segmentor()
        segmentor.load(model)
        word_list = list(segmentor.segment(sentence))
        segmentor.release()
        return word_list

    def get_postag_list(self, word_list, model):
        # 得到词性标注
        postag = Postagger()
        postag.load(model)
        postag_list = list(postag.postag(word_list))
        postag.release()
        return postag_list

    def get_parser_list(self, word_list, postag_list, model):
        # 得到依存关系
        parser = Parser()
        parser.load(model)
        arcs = parser.parse(word_list, postag_list)
        arc_list = [(arc.head, arc.relation) for arc in arcs]
        parser.release()
        return arc_list

    def get_ner(self, word_list,  postag_list, model):
        recognizer = NamedEntityRecognizer()
        recognizer.load(model)
        netags = recognizer.recognize(word_list, postag_list)  # 命名实体识别
        # for word, ntag in zip(word_list, netags):
        #     print(word + '/' + ntag)
        recognizer.release()  # 释放模型
        return list(netags)

    def news_parser(self, news):
        word_list = self.get_word_list(news, LTP_CWS_MODEL)
        postag_list = self.get_postag_list(word_list, LTP_POS_MODEL)
        parser_list = self.get_parser_list(word_list, postag_list, LTP_PAR_MODEL)
        for i in range(len(word_list)):
            print(i + 1, word_list[i], parser_list[i])

    def get_news_parser(self, news):
        word_list =   self.get_word_list(news, LTP_CWS_MODEL)
        postag_list = self.get_postag_list(word_list, LTP_POS_MODEL)
        parser_list = self.get_parser_list(word_list, postag_list, LTP_PAR_MODEL)
        return [(word_list[i], parser_list[i]) for i in range(len(word_list))]

    def get_speak(self, word_parser_list):

        # 将表示说的词加入这个list
        speak_words =  ['表示', '说','回复', '指出', '认为', '坦言', '告诉', '强调', '称', '直言', '普遍认为', '介绍', '透露', '重申', '呼吁', '说道', '感叹', '地说', '写道',
         '中称', '证实', '还称', '猜测', '暗示', '感慨', '热议', '敦促', '指责', '声称', '主张', '反对', '批评', '表态', '中说', '承认', '却说', '感触',
         '提到', '所说', '引述', '质疑', '抨击','回应', '分析']

        speak_indexes = []
        for _, (index, parser) in word_parser_list:
            if parser == 'SBV':
                verb_index = index - 1
                if word_parser_list[verb_index][0] in speak_words:
                    speak_indexes.append(verb_index)
        return sorted(list(set(speak_indexes)))

    def get_speak_sentence(self, word_parser_list, speak_index):
        """
        这里需要返回2个句子：
            1. “说”这个词所在的句子
                1.1 句首为任意标点符号或者是整个字符串开头
                1.2 句尾为任意标点符号或者是整个字符串结尾
            2. “说”的内容
                2.1 该句应该紧贴着“说“这个词所在的句子（前方或者后方）
                2.2 句首为：，或”
                2.3 句尾为：。或“
                2.4 如果“说”的内容在“说”这个词所在句子之前，那么必须以”结尾
        """

        # “说”所在的句子
        speak_start = 0
        for i in range(speak_index, -1, -1):
            if word_parser_list[i][1][1] == 'WP':
                speak_start = i + 1
                break
        speak_end = len(word_parser_list) - 1
        for i in range(speak_index, speak_end + 1):
            if word_parser_list[i][1][1] == 'WP':
                speak_end = i
                break

        speak_words = [word_parser_list[i][0] for i in range(speak_start, speak_end + 1)]
        speak_sentence = ''.join(speak_words)

        # 位于“说”所在句子之前的内容
        content_before = ''
        content_before_end = speak_start - 1
        if content_before_end > 0:
            if word_parser_list[content_before_end][0] == '”':  # 没有详细考虑过所有情况，这里假定位于之前的句子内容必须在引号范围内
                for i in range(content_before_end, -1, -1):
                    if word_parser_list[i][0] == '“':
                        content_before = ''.join(word_parser_list[i][0] for i in range(i, content_before_end + 1))
                        break

        # 位于"说"所在句子之后的内容
        content_after = ''
        content_after_start = speak_end + 1
        if word_parser_list[speak_end][0] != '。' and content_after_start < len(word_parser_list):  # 如果句子以句号结尾，那么说明后面不是说的内容
            if word_parser_list[content_after_start][0] == '“':
                end_wp = '”'
            else:
                end_wp = '。'
            for i in range(content_after_start, len(word_parser_list)):
                if word_parser_list[i][0] == end_wp:
                    content_after = ''.join(word_parser_list[i][0] for i in range(content_after_start, i + 1))
                    break

        return speak_sentence, speak_words, content_before, content_after


    def get_speak_content(self, news):
        news = re.sub('\\\\n|[\n\u3000\r]', '', news)
        word_parser_list = self.get_news_parser(news)
        speak_indexes = self.get_speak(word_parser_list)
        result = []
        for speak_index in speak_indexes:
            result_item = {"speaker":"", "content":"", "sentiment":0}
            speak_sentence, speak_words, content_before, content_after = self.get_speak_sentence(word_parser_list, speak_index)

            # 找包含“说”句子里面的人名
            print(speak_words)
            postag_list = self.get_postag_list(speak_words, LTP_POS_MODEL)
            ner_list = self.get_ner(speak_words, postag_list, LTP_NER_MODEL)
            person_name_id = [idx for idx, ner in enumerate(ner_list) if ner=='S-Nh']

            # 若没有找到人名，认为不是我们要找的句子，跳过
            if len(person_name_id) == 0:
                continue

            # 提取人名
            person_name = speak_words[person_name_id[0]:person_name_id[-1]+1] # 可能出现“x和y说.."，取最大范围
            person_name = "".join(person_name)
            result_item["speaker"] = person_name

            # 组合所说的话（前后的语句合并）
            speak_content = ""
            if content_before:
                # print('content_before:')
                # print(content_before)
                speak_content += content_before
            if content_after:
                # print('content_after:')
                # print(content_after)
                speak_content += " " + content_after
            result_item["content"] = speak_content

            # 情感分析
            sentiment = self.bert.predict(speak_content)
            result_item["sentiment"] = sentiment

            # 添加到结果列表
            result.append(result_item)

        return result


if __name__ == '__main__':
    # path = os.path.join(dirname, '../data/sqlResult_1558435.csv')
    # news = pd.read_csv(path, encoding='gb18030')
    # news = news.fillna('')
    # content = news['content'].tolist()
    # # 读取中文停用词
    # with open(os.path.join(dirname, '../data/stop_word.txt'), 'r', encoding='utf8') as f:
    #     stopwords = [w.strip() for w in f.readlines()]
    #
    # sample_content = random.sample(content, 100)
    # test_news = random.choice(sample_content)

    test_news = '''2月29日，在美国华盛顿，美国副总统彭斯参加白宫记者会。新华社记者 刘杰 摄

2日，彭斯与卫生部门官员共同就新冠病毒肺炎召开记者会，有政府官员称，美国新增4个新冠病毒肺炎死亡病例，死亡总数为6人。

“尽管今天有这个坏消息，但是我们仍要说清楚，据所有与我们政府部门合作的专家，美国人感染新型冠状病毒的几率仍然很低，”彭斯说。

他还提到，国内病例中有29例来自加利福尼亚州或华盛顿特区。

卫生官员称，43个确诊患者中，有17个病例是在旅行中感染新冠病毒，26人是通过人与人的传播感染。

目前，彭斯负责领导特朗普政府对新冠病毒疫情的应对工作，他已经选择德博拉·伯克斯协助自己应对此次疫情危机。

不过美国《纽约时报》援引美国卫生部门官员表态称，截至当地时间2日晚，美国境内新冠病毒感染病例数已达100例，其中包括6例死亡病例（全部发生在美国西北部华盛顿州）。'''
    print(test_news)
    s = SpeakDetect()
    print(s.get_speak_content(test_news))