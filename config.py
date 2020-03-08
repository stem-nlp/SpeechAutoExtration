import os
dirname = os.path.dirname(__file__)

BERT_MODEL_PATH = os.path.join(dirname,'save/bert_cla.ckpt')
LTP_CWS_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/cws.model")
LTP_POS_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/pos.model")
LTP_PAR_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/parser.model")
LTP_NER_MODEL = os.path.join(dirname,"data/ltp_data_v3.4.0/ner.model")
