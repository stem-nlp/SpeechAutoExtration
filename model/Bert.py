import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup
from config import *

# 超参数
weight_decay = 1e-2
hidden_dropout_prob = 0.3
num_labels = 2


class Bert:
    def __init__(self):
        # Bert模型以及相关配置

        # config = BertConfig.from_pretrained('bert-base-chinese',
        #                                     num_labels=num_labels,
        #                                     hidden_dropout_prob=hidden_dropout_prob)
        #
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)

        # self.device = torch.device('cpu')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        self.model.to(self.device)

        # self.model.load_state_dict(torch.load(BERT_MODEL_PATH))
        # self.model.save_pretrained(os.path.dirname(BERT_MODEL_PATH))

    def convert_text_to_ids(self, tokenizer, text, max_len=100):
        if isinstance(text, str):
            tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
            input_ids = [tokenized_text["input_ids"]]
            token_type_ids = [tokenized_text["token_type_ids"]]
        elif isinstance(text, list):
            input_ids = []
            token_type_ids = []
            for t in text:
                tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
                input_ids.append(tokenized_text["input_ids"])
                token_type_ids.append(tokenized_text["token_type_ids"])
        else:
            raise Exception("Input unknown")
        return input_ids, token_type_ids

    def seq_padding(self, tokenizer, X):
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

        if len(X) <= 1:
            return torch.tensor(X)
        max_len = max([len(x) for x in X])
        # padded = [x + [pad_id] * (max_len - len(x)) if len(x) < max_len else x for x in X]
        padded = [x + [pad_id] * (max_len - len(x)) for x in X]
        X = torch.Tensor(padded)
        return X

    def predict(self, text):
        # 测试
        self.model.eval()
        with torch.no_grad():
            input_ids, token_type_ids = self.convert_text_to_ids(self.tokenizer, text)
            input_ids = self.seq_padding(self.tokenizer, input_ids)
            token_type_ids = self.seq_padding(self.tokenizer, token_type_ids)
            input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
            input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
            y_pred_label = output[0].argmax(dim=1)

        return y_pred_label.item()

if __name__ == "__main__":
    b = Bert()
    res = b.predict("这名法国后卫有超强的个人得分能力，堪称刷分机器。")
    print(res)
    res = b.predict("尽管今天有这个坏消息，但是我们仍要说清楚，据所有与我们政府部门合作的专家，美国人感染新型冠状病毒的几率仍然很低，”")
    print(res)