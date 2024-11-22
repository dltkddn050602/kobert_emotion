import torch 
from torch import nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
import gluonnlp as nlp 
import numpy as np 
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from kobert_tokenizer import KoBERTTokenizer

def get_kobert_model (model_path, vocab_file, ctx="cpu") :
    bertmodel = BertModel.from_pretrained(model_path)
    device = torch.device(ctx)
    bertmodel. to(device)
    bertmodel.eval()
    vocab_b_obj = nlp.vocab.BERTVocab. from_sentencepiece(vocab_file,
    padding_token='[PAD]')
    return bertmodel, vocab_b_obj
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_kobert_model('skt/kobert-base-v1',tokenizer.vocab_file)

device = torch.device('cpu')

def get_kobert_model (model_path, vocab_file, ctx="cpu") :
    bertmodel = BertModel.from_pretrained(model_path)
    device = torch.device(ctx)
    bertmodel. to(device)
    bertmodel.eval()
    vocab_b_obj = nlp.vocab.BERTVocab. from_sentencepiece(vocab_file,
    padding_token='[PAD]')
    return bertmodel, vocab_b_obj

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
# Setting parameters
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 100
learning_rate =  5e-5

PATH = '/Users/sangwoolee/Desktop/a/'
model = torch.load(PATH + 'KoBERT_담화.pt') 
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))

tok=tokenizer.tokenize


def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("현실형")
            elif np.argmax(logits) == 1:
                test_eval.append("탐구형")
            elif np.argmax(logits) == 2:
                test_eval.append("예술형")
            elif np.argmax(logits) == 3:
                test_eval.append("사회형")
            elif np.argmax(logits) == 4:
                test_eval.append("기업형")
            elif np.argmax(logits) == 5:
                test_eval.append("관습형")
            

        print(">> 당신은 " + test_eval[0] + " 유형의 학생입니다.")

#질문 무한반복하기! 0 입력시 종료
end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0" :
        break
    predict(sentence)
    print("\n")