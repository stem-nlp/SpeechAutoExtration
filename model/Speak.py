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
        print(result)
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

    test_news = '''
    新冠肺炎疫情发生以来，新冠病毒的起源、传播及演变备受关注。多位国内外专家表示，根据目前已有证据还无法确认新冠病毒起源于哪里。
传播“拼图”有缺失
新冠病毒在人类中的传播是如何开始的？从最初报告的病例看，武汉的华南海鲜市场一度被认为是疫情发源地。
然而，在英国《柳叶刀》杂志1月刊登的一篇论文中，武汉金银潭医院副院长黄朝林等人分析了首批确诊的41例新冠肺炎病例，发现其中只有27例去过华南海鲜市场。回溯研究认为首名确诊患者于2019年12月1日发病，并无华南海鲜市场暴露史，也没发现与之后确诊病例间的流行病学联系，而其家人也没出现过发热和呼吸道症状。
美国《科学》杂志网站相关报道中，美国斯克里普斯研究所生物学家克里斯蒂安·安德森推测说，新冠病毒进入华南海鲜市场可能有三种场景：可能由一名感染者、一只动物或一群动物带到该市场。
多位专家及多项研究支持了上述观点。被称为“病毒猎手”的美国哥伦比亚大学梅尔曼公共卫生学院教授维尔特·伊恩·利普金表示，新冠病毒与华南海鲜市场的联系可能不那么直接，也许该市场发生的是“二次传播”，而病毒在早些时候已开始扩散。
中国科学院西双版纳热带植物园等机构研究人员近期以预印本形式发布论文说，他们分析了四大洲12个国家的93个新冠病毒样本的基因组数据，发现其中包含58种单倍型，与华南海鲜市场有关联的患者样本单倍型都是H1或其衍生类型，而H3、H13和H38等更“古老”的单倍型来自华南海鲜市场之外，印证了华南海鲜市场的新冠病毒是从其他地方传入的观点。
要还原新冠病毒传播链，科学家还缺少一些“拼图”，其中最关键一块是常被称为“零号病人”的首个感染者。“零号病人”是众多疑问交汇处，对寻找中间宿主以及解答病毒如何从动物传播给人类等疑问至关重要。
一个著名例子是百年前据估计造成全球数千万人死亡的“西班牙流感”，尽管此次疫情因西班牙最先报道而得名，但后来一些回溯性研究发现，首个感染者可能是来自美国堪萨斯州军营的一名士兵。
美国乔治敦大学传染病专家丹尼尔·卢西表示，考虑到病毒潜伏期等因素，首个新冠病毒感染者可能在2019年11月或更早时候就已经出现了。
从新冠病毒全球传播来看，尽管大多数新冠肺炎病例可以追踪到传染源，但美国等国家已报告了不少无法溯源的病例。在疫情日趋严重的意大利，其国内“零号病人”至今尚未找到。
病毒溯源未完成
新冠病毒源于动物，它进入人体前在自然界是如何生存进化的？中科院武汉病毒研究所等机构研究人员2月在英国《自然》杂志上发表论文说，他们发现新冠病毒与蝙蝠身上的一株冠状病毒（简称TG13）基因序列一致性高达96%。TG13是迄今已知的与新冠病毒基因最相近的毒株，表明蝙蝠很可能是新冠病毒的自然界宿主。
其他一些研究还发现，新冠病毒与穿山甲携带的冠状病毒基因序列有相似性，尤其在允许病毒进入细胞的受体结合域上十分接近。这表明新冠病毒进化过程中，TG13可能和穿山甲携带的冠状病毒之间发生了重组。
虽然相关研究提供了线索，不过多位接受新华社记者采访的专家表示，新冠病毒起源以及中间宿主等还难以定论，对病毒完全溯源可能需要更长时间。
英国诺丁汉大学分子病毒学教授乔纳森·鲍尔说，人类新冠病毒与穿山甲之间的联系仍是一个“小问号”，目前仍然没有得到病毒来源的最终答案。但如果将所有碎片线索放在一起，它们指向一个病毒从动物传播出来的事件。
美国科罗拉多州立大学兽医和生物医药科学学院教授查理·卡利舍表示，他对讨论新冠病毒来源持开放态度，下结论需要科学数据支持，而不仅仅是猜测。
美国艾奥瓦大学微生物学和免疫学教授斯坦利·珀尔曼认为，作为新冠病毒中间宿主的动物有可能来自中国以外，例如走私的穿山甲等动物。
2月底发布的《中国-世界卫生组织新型冠状病毒肺炎（COVID-19）联合考察报告》也指出，“现有知识局限”的问题包括“病毒的动物来源和天然宿主”“初始阶段的动物到人的感染过程”“早期暴露史不详的病例”等。
全球疫情仍在蔓延，诸多疑问还有待各国科研人员携手解答。正如世卫组织总干事谭德塞日前多次强调，在全球共同抗击新冠肺炎疫情时，“需要事实，而非恐惧”“需要科学，而非谣言”“需要团结，而非污名化”。
（原标题为：科普：新冠病毒起源于哪里？专家表示目前尚难下结论）
    '''
    print(test_news)
    s = SpeakDetect()
    print(s.get_speak_content(test_news))