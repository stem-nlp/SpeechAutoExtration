import os
dirname = os.path.dirname(__file__)

BERT_MODEL_PATH = os.path.join(dirname,'save/')
LTP_CWS_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/cws.model")
LTP_POS_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/pos.model")
LTP_PAR_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/parser.model")
LTP_NER_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/ner.model")
SAY_WORDS_FILE = os.path.join(dirname,'data/listsayverb.txt')
# SAY_WORDS_FILE = os.path.join(dirname,'data/listsayverb_large.txt')
