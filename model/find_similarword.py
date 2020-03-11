# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:37:08 2020

@author: zwconnie
"""

from pyltp import Segmentor
from collections import defaultdict
from pyltp import SentenceSplitter, NamedEntityRecognizer, Postagger, Parser, Segmentor
import numpy as np
import pandas as pd
import re
from model.Bert import Bert
from config import *

# 需更改路径
cws_model = LTP_CWS_MODEL
pos_model = LTP_POS_MODEL
par_model = LTP_PAR_MODEL
ner_model = LTP_NER_MODEL
'''
sayverb_file=open('listsayverb.txt‘)
speak_words=[]
for line in sayverb_file.readlines():
    if len(line.split())>0:
        if line.split()[0] not in speak_words:
            speak_words.append(line.split()[0])
sayverb_file.close()
'''

speak_words = ['表示', '说', '回复', '指出', '认为', '坦言', '告诉', '强调', '称', '直言', '普遍认为', '介绍', '透露', '重申', '呼吁', '说道', '感叹',
               '地说', '写道',
               '中称', '证实', '还称', '猜测', '暗示', '感慨', '热议', '敦促', '指责', '声称', '主张', '反对', '批评', '表态', '中说', '承认', '却说',
               '感触',
               '提到', '所说', '引述', '质疑', '抨击', '回应', '分析说', '发现', '表示', '表态', '推测', '推断', '判决', '判定', '要求']


class SpeakDetect:
    def __init__(self):
        self.bert = Bert()  # 用于情感分析

    def get_word_list(self, sentence, model):
        # 得到分词
        segmentor = Segmentor()
        segmentor.load(model)
        sentence = ''.join(re.sub('\\\\n|[\n\r]', '', sentence))
        word_list = list(segmentor.segment(sentence))
        segmentor.release()
        self.word_list = word_list

    def get_postag_list(self, word_list, model):
        # 得到词性标注
        postag = Postagger()
        postag.load(model)
        postag_list = list(postag.postag(word_list))
        postag.release()
        self.postag_list = postag_list
        wordtag_dict = defaultdict(str)
        for i in range(len(word_list)):
            if word_list[i] not in wordtag_dict.keys():
                wordtag_dict[word_list[i]] = postag_list[i]
        self.wordtag_dict = wordtag_dict

    def get_parser_list(self, word_list, postag_list, model):
        # 得到依存关系
        parser = Parser()
        parser.load(model)
        arcs = parser.parse(word_list, postag_list)
        parser_list = [(arc.head, arc.relation) for arc in arcs]
        parser.release()
        self.parser_list = parser_list

    def get_ner(self, word_list, postag_list, model):
        recognizer = NamedEntityRecognizer()
        recognizer.load(model)
        netags = recognizer.recognize(word_list, postag_list)  # 命名实体识别
        recognizer.release()  # 释放模型
        self.ner_list = list(netags)

    def get_news_parser(self, news):
        self.news = news
        self.get_word_list(news, cws_model)
        self.get_postag_list(self.word_list, pos_model)
        self.get_parser_list(self.word_list, self.postag_list, par_model)
        self.get_ner(self.word_list, self.postag_list, ner_model)
        word_rel_list = [[i + 1, self.word_list[i], self.parser_list[i], self.ner_list[i]] for i in
                         range(len(self.word_list))]
        # 若有+名词且依存关系为兼词，删除“有”不影响句意，且能识别正确的动宾关系(例如：有政府官员称，政府官员和称之间的动宾关系未能识别)，删除后重新获取词性标注、依存关系
        for item in word_rel_list:
            if item[1] == '有':
                adj_list = [subitem for subitem in word_rel_list if (
                            (subitem[2][0] == item[0]) & (self.postag_list[subitem[0] - 1][0] == 'n') & (
                                subitem[2][1] == 'DBL'))]
                if len(adj_list) > 0:
                    self.word_list.remove(item[1])
        if len(self.word_list) < len(self.postag_list):
            self.get_postag_list(self.word_list, pos_model)
            self.get_parser_list(self.word_list, self.postag_list, par_model)
            self.get_ner(self.word_list, self.postag_list, ner_model)
            word_rel_list = [[i + 1, self.word_list[i], self.parser_list[i], self.ner_list[i]] for i in
                             range(len(self.word_list))]
        # 合并专有名词B-N,I-N,E-N为同一个词，已有的模型获得列表同时删除对应位置的元素，均用专有名词最后一个词代表整个专有名词
        remove_element = []
        for i in range(len(word_rel_list)):
            if word_rel_list[i][3][:3] == 'B-N':
                complete_word = word_rel_list[i][1]
                remove_element.append(i)
            elif word_rel_list[i][3][:3] == 'I-N':
                complete_word += word_rel_list[i][1]
                remove_element.append(i)
            elif word_rel_list[i][3][:3] == 'E-N':
                word_rel_list[i][1] = complete_word + word_rel_list[i][1]
                complete_word = ''
            elif word_rel_list[i][3] == 'O':
                word_rel_list[i][3] += '-NER'
        self.word_list = [self.word_list[j] for j in range(len(self.word_list)) if j not in remove_element]
        self.postag_list = [self.postag_list[j] for j in range(len(self.postag_list)) if j not in remove_element]
        self.parser_list = [self.parser_list[j] for j in range(len(self.parser_list)) if j not in remove_element]
        self.ner_list = [self.ner_list[j] for j in range(len(self.ner_list)) if j not in remove_element]
        self.word_rel_list = [item[:3] for item in word_rel_list if item[3][:3] not in ['B-N', 'I-N']]
        # word_rel_list的元素为[词的位置,词的字符串，词的依存关系元祖（上级词的位置，依存关系）]

    # 获得依存关系路径
    def get_path(self, word_rel_list, path_list):
        path_end = 0
        for item in word_rel_list:
            if (item[0] == path_list[-1][1]):
                path_list.append((item[0], item[2][0], item[2][1]))
                path_end = 1
                break
        if path_end == 1:
            self.get_path(word_rel_list, path_list)
        return path_list

    # 获得主语的修饰词，规则：与核心词间关系为SBV的词作为主语词基(已经过NER合并专有名词)，寻找所有与该词基有ATT依存关系的词，按词序排列,但最多只取3个修饰词
    def get_ATT(self, word_rel_list, path_list):
        center = path_list[-1][0]
        center_id = 0
        path_end = 0
        for i in range(len(word_rel_list)):
            if word_rel_list[i][0] == center:
                center_id = i
                break
        for i in range(center_id, -1, -1):
            if (word_rel_list[i][2][0] == path_list[-1][0]) & (word_rel_list[i][2][1] in ['ATT']):
                path_list.append((word_rel_list[i][0], word_rel_list[i][1], word_rel_list[i][2][1]))
                path_end = 1
                break
        if path_end == 1:
            self.get_ATT(word_rel_list, path_list)
        sorted_list = sorted(path_list, key=lambda x: x[0])
        for i in range(len(sorted_list)):
            if (sorted_list[i][2] == 'RAD') & (sorted_list[i][1] == '的'):
                sorted_list = sorted_list[i + 1:]
        if len(sorted_list) > 1:
            for i in range(len(word_rel_list)):
                if word_rel_list[i][0] == sorted_list[-2][0]:
                    att_id = i
                elif word_rel_list[i][0] == sorted_list[-1][0]:
                    sbv_id = i
            if att_id != sbv_id - 1:
                sorted_list = sorted_list[-1:]
        ATT_list = [sub[1] for sub in sorted_list]
        num_position = sorted_list[-1][0]
        # dict_wordpos=defaultdict(str)
        # for i in range(len(self.word_list)):
        # dict_wordpos[self.word_list[i]]=self.postag_list[i]
        # ATT_list=[sub[1] for sub in sorted_list if dict_wordpos[sub[1]] !='v']
        ATT_subject = ''.join(ATT_list)
        return [num_position, ATT_subject]

    # 获得内容：所有依存路径中包含该动词的词按顺序排列
    def get_content(self, word_rel_list, content_code, content_center):
        content_list = [content_center]
        for item in word_rel_list:
            item_path = [x[1] for x in item[3]]
            if content_code in item_path:
                content_list.append((item[0], item[1]))
        sorted_list = sorted(content_list, key=lambda x: x[0])
        start_num = sorted_list[0][0]
        end_num = sorted_list[-1][0]
        content_list = [word[1] for word in word_rel_list if ((word[0] >= start_num) & (word[0] <= end_num + 1))]
        for i in range(len(content_list)):
            if content_list[i] in ['。', '！', '？']:
                content_list = content_list[:i]
                break
        # 长度低于5个字的言论作为噪音不提取
        if len(content_list) <= 5:
            content = ''
        else:
            if self.wordtag_dict[content_list[-1]] == 'wp':
                content_list = content_list[:-1]
            elif self.wordtag_dict[content_list[-2]] == 'wp':
                content_list = content_list[:-2]
            content = ''.join(content_list)
        return content, (start_num, end_num)

    def get_speak(self):
        subject_code = []
        speak_code = []
        speak_content = []
        # word_rel_list的元素为[词的位置,词的字符串，词的依存关系元祖（上级词的位置，依存关系），词的依存路径列表[(词自身位置，上级词位置，依存关系)]]
        # 为防止将后一句说话内容作为前一句内容的一部分提取，将所有pos_model定义为动词且在speak_words中的词都作为原始核心词，不与之前的动词产生依存关系，依存于后句核心词的其他词也不依存于前句
        word_rel_list = self.word_rel_list
        word_rel_list = [[word_rel_list[i][0], word_rel_list[i][1], (0, 'HED')] if (
                    (word_rel_list[i][1] in speak_words) & (self.postag_list[i] == 'v')) else word_rel_list[i] for i in
                         range(len(word_rel_list))]
        for item in word_rel_list:
            item.append(self.get_path(word_rel_list, [(item[0], item[2][0], item[2][1])]))
        for item in word_rel_list:
            recorded = []
            if (item[1] in speak_words) & (item[2][0] == 0):
                speak_code.append(item[1])
                speak_num = len(speak_code)
                for subitem in word_rel_list:
                    if (subitem[2][0] == item[0]) & (subitem[2][1] == 'SBV'):
                        if len(subject_code) < speak_num:
                            subject_code.append(
                                [self.get_ATT(word_rel_list, [(subitem[0], subitem[1], subitem[2][1])])])
                        else:
                            subject_code[speak_num - 1].append(
                                self.get_ATT(word_rel_list, [(subitem[0], subitem[1], subitem[2][1])]))
                    if (subitem[2][0] == item[0]) & (subitem[2][1] in ['VOB', 'FOB']):
                        content, position = self.get_content(word_rel_list, subitem[0], (subitem[0], subitem[1]))
                        if len(speak_content) < speak_num:
                            speak_content.append([content])
                        else:
                            speak_content[speak_num - 1].append(content)
                        recorded.append(position)
                # 未能找到核心动词对应的名词和内容时，用空列表填充
                if len(subject_code) < speak_num:
                    subject_code.append([])
                if len(speak_content) < speak_num:
                    speak_content.append([])
        # self.word_rel_list=word_rel_list
        # 多个名词依存于同一个核心动词时，取最近的名词
        subject_code = [x[-1] if len(x) > 0 else ['', ''] for x in subject_code]
        # self.subject_code=subject_code
        # self.speak_code=speak_code
        # self.speak_content=speak_content
        subject_sumlist = []
        for i in range(len(word_rel_list)):
            if (word_rel_list[i][2][1] == 'SBV') & (self.wordtag_dict[word_rel_list[i][1]] != 'r'):
                subject_sumlist.append(
                    self.get_ATT(word_rel_list, [(word_rel_list[i][0], word_rel_list[i][1], word_rel_list[i][2][1])]))
        subject_sumlist = sorted(subject_sumlist, key=lambda x: x[0])
        # 若名词为代词，替换为前面最临近的名词,为空则保留为空
        for i in range(len(subject_code)):
            if self.wordtag_dict[subject_code[i][1]] == 'r':
                for j in range(len(subject_sumlist)):
                    if subject_sumlist[j][0] >= subject_code[i][0]:
                        subject_code[i][1] = subject_sumlist[j - 1][1]
                        break
        subject_code = [item[1] for item in subject_code]
        speak_content = [x[0] if len(x) > 0 else '' for x in speak_content]
        sum_list = list(zip(subject_code, speak_code, speak_content))
        remove_sumlist = []
        for i in range(len(sum_list)):
            if (len(sum_list[i][0]) == 0) | (len(sum_list[i][2]) == 0):
                remove_sumlist.append(i)
        sum_list = [sum_list[i] for i in range(len(sum_list)) if i not in remove_sumlist]
        self.df_result = pd.DataFrame(sum_list, columns=['person', 'verb', 'content'])

    def get_sentiment(self, news):
        self.get_news_parser(news)
        self.get_speak()
        result = []
        self.df_result['sentiment'] = '-'
        # 情感分析
        for i in range(len(self.df_result.person)):
            sentiment = self.bert.predict(self.df_result.loc[i, 'content'])
            self.df_result.loc[i, "sentiment"] = sentiment
            result_item = {"speaker": self.df_result.loc[i, 'person'], "content": self.df_result.loc[i, 'content'],
                           "sentiment": self.df_result.loc[i, 'sentiment']}
            result.append(result_item)
        # 添加到结果列表
        print(result)
        return result

if __name__ == '__main__':
    s = SpeakDetect()
    # test_news='''2月29日，在美国华盛顿，美国副总统彭斯参加白宫记者会。新华社记者 刘杰 摄 2日，彭斯与卫生部门官员共同就新冠病毒肺炎召开记者会，有政府官员称，美国新增4个新冠病毒肺炎死亡病例，死亡总数为6人。“尽管今天有这个坏消息，但是我们仍要说清楚，据所有与我们政府部门合作的专家，美国人感染新型冠状病毒的几率仍然很低“，彭斯说。他还提到，国内病例中有29例来自加利福尼亚州或华盛顿特区。卫生官员称，43个确诊患者中，有17个病例是在旅行中感染新冠病毒，26人是通过人与人的传播感染。目前，彭斯负责领导特朗普政府对新冠病毒疫情的应对工作，他已经选择德博拉·伯克斯协助自己应对此次疫情危机。不过美国《纽约时报》援引美国卫生部门官员表态称，截至当地时间2日晚，美国境内新冠病毒感染病例数已达100例，其中包括6例死亡病例（全部发生在美国西北部华盛顿州）。'''
    test_news  = '''
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
    s.get_sentiment(test_news)
