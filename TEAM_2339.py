#!/usr/bin/env python
# coding: utf-8

# ---
# # Preparation


# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math
import numpy as np
import time
import torch, pandas as pd
import nltk
import re
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from transformers import set_seed
set_seed(123)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[2]:


df = pd.read_csv('Batch_answers - train_data (no-blank).csv', encoding="ISO-8859-1")
df = df.drop(columns=['Unnamed: 6', 'total no.: 7987'])
# df_test = pd.read_csv('Batch_answers - test_data(no_label).csv', encoding="ISO-8859-1")
df


# ---
# # Data Processing

# In[4]:


df[['q','r',"q'","r'"]] = df[['q','r',"q'","r'"]].apply(lambda x: x.str.strip('\"'))

# # augment 50%
df_aug = df.sample(n=int(0.5*len(df)), random_state=1)
df = pd.concat([df, df_aug], ignore_index=True)

    

df['r'] = df['s'] + ':' + df['r']
df['sub_q_true'] = [1 if x in y else 0 for x,y in zip(df["q'"],df["q"])]
df['sub_r_true'] = [1 if x in y else 0 for x,y in zip(df["r'"],df["r"])]
df['sub_both'] = df['sub_q_true']*df['sub_r_true']
df.head(3)


# In[5]:


data = df.loc[df['sub_both'] == 1]
data['q_start'] = [y.index(x) for x,y in zip(data["q'"],data["q"])]
data['r_start'] = [y.index(x) for x,y in zip(data["r'"],data["r"])]
data['q_end'] = [x+len(y)-1 for x,y in zip(data["q_start"],data["q'"])]
data['r_end'] = [x+len(y)-1 for x,y in zip(data["r_start"],data["r'"])]
data


# In[6]:


from sklearn.model_selection import train_test_split

train, valid = train_test_split(data, test_size=0.1)


# ---
# # Tokenize

# In[7]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


# In[8]:


train_data_q = train['q'].tolist()
valid_data_q = valid['q'].tolist()

train_data_r = train['r'].tolist()
valid_data_r = valid['r'].tolist()


# In[9]:


train_encodings = tokenizer(train_data_q, train_data_r, truncation=True, padding=True)
val_encodings = tokenizer(valid_data_q, valid_data_r, truncation=True, padding=True)


# In[10]:


train_answer = train[['q_start', 'r_start',	'q_end', 'r_end']].to_dict('records')
valid_answer = valid[['q_start', 'r_start',	'q_end', 'r_end']].to_dict('records')


# In[11]:


def add_token_positions(encodings, answers):
    q_start, r_start, q_end, r_end = [],[],[],[]

    for i in range(len(answers)):
        # print(i)
        q_start.append(encodings.char_to_token(i, answers[i]['q_start'], 0))
        r_start.append(encodings.char_to_token(i, answers[i]['r_start'], 1))
        q_end.append(encodings.char_to_token(i, answers[i]['q_end'], 0))
        r_end.append(encodings.char_to_token(i, answers[i]['r_end'], 1))

        if q_start[-1] is None:
            q_start[-1] = 0
            q_end[-1] = 0
            # continue

        if r_start[-1] is None:
            r_start[-1] = 0
            r_end[-1] = 0
            # continue

        shift = 1
        while q_end[-1] is None:
            q_end[-1] = encodings.char_to_token(i, answers[i]['q_end'] - shift)
            shift += 1
        shift = 1
        while r_end[-1] is None:
            r_end[-1] = encodings.char_to_token(i, answers[i]['r_end'] - shift)
            shift += 1
    encodings.update({'q_start':q_start, 'r_start':r_start,	'q_end':q_end, 'r_end':r_end})


# In[12]:


# Convert char_based_id to token_based_id
# Find the corresponding token id after input being tokenized
add_token_positions(train_encodings, train_answer)
add_token_positions(val_encodings, valid_answer)


# In[13]:


print(train_encodings.keys())
print(train_encodings['q_start'][0])
print(train_encodings['r_start'][0])
print(train_encodings['q_end'][0])
print(train_encodings['r_end'][0])


# ---
# # Dataset

# In[14]:


class qrDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


# In[15]:


train_dataset = qrDataset(train_encodings)
val_dataset = qrDataset(val_encodings)


# ---
# # Model

# In[17]:


from transformers import BertModel

class myModel(torch.nn.Module):

    def __init__(self):

        super(myModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(256*2, 4)
        

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        logits = output[0]
        logits, _ = self.lstm(logits)
        # print(logits)
        # print(logits.shape)
        out = self.fc(logits)

        return out


# In[20]:


from transformers import AdamW
from tqdm import tqdm
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)


epochs = 2
loss_fct = CrossEntropyLoss()
total_steps = len(train_loader) * epochs
model = myModel().to(device)
optim = AdamW(model.parameters(), lr=3e-5)


# In[21]:


def evaluate(valid_loader):
    q_acc, r_acc = [], []
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(valid_loader, leave=True)
        for batch_id, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            q_start = batch['q_start'].to(device)
            r_start = batch['r_start'].to(device)
            q_end = batch['q_end'].to(device)
            r_end = batch['r_end'].to(device)

            # model output
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            # q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 1)
            q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)
            
            
            q_start_pred = torch.argmax(q_start_logits, dim=1)
            r_start_pred = torch.argmax(r_start_logits, dim=1)
            q_end_pred = torch.argmax(q_end_logits, dim=1)
            r_end_pred = torch.argmax(r_end_logits, dim=1)

            q_start_logits = q_start_logits.squeeze(-1).contiguous()
            r_start_logits = r_start_logits.squeeze(-1).contiguous()
            q_end_logits = q_end_logits.squeeze(-1).contiguous()
            r_end_logits = r_end_logits.squeeze(-1).contiguous()

            q_start_loss = loss_fct(q_start_logits, q_start)
            r_start_loss = loss_fct(r_start_logits, r_start)
            q_end_loss = loss_fct(q_end_logits, q_end)
            r_end_loss = loss_fct(r_end_logits, r_end)

            loss = q_start_loss + r_start_loss + q_end_loss + r_end_loss
            
            q_start = q_start.reshape([q_start.size(dim=0),1])
            r_start = r_start.reshape([r_start.size(dim=0),1])
            q_end = q_end.reshape([q_end.size(dim=0),1])
            r_end = r_end.reshape([r_end.size(dim=0),1])
            
            q_start_score = ((q_start_pred == q_start).sum()/len(q_start_pred)).item()
            r_start_score = ((r_start_pred == r_start).sum()/len(r_start_pred)).item() 
            q_end_score = ((q_end_pred == q_end).sum()/len(q_end_pred)).item() 
            r_end_score = ((r_end_pred == r_end).sum()/len(r_end_pred)).item() 
            
            q_acc.append(q_start_score)
            q_acc.append(q_end_score)
            r_acc.append(r_start_score)
            r_acc.append(r_end_score)

            running_loss += loss.item()
            if batch_id % 30 == 0 and batch_id != 0:
                print('Validation Epoch {} Batch {} Loss {:.4f}'.format(
                    batch_id + 1, batch_id, running_loss / 30))
                running_loss = 0.0
        q_acc = sum(q_acc)/len(q_acc)
        r_acc = sum(r_acc)/len(r_acc)

        print("evaluate loss: ", loss)
        print("q-acc: ", q_acc)
        print("r-acc: ", r_acc)


# In[22]:


for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, leave=True)
    for batch_id, batch in enumerate(loop):
        # reset
        optim.zero_grad()


        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        q_start = batch['q_start'].to(device)
        r_start = batch['r_start'].to(device)
        q_end = batch['q_end'].to(device)
        r_end = batch['r_end'].to(device)


        # model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 1)
        q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)
        
        

        q_start_logits = q_start_logits.squeeze(-1).contiguous()
        r_start_logits = r_start_logits.squeeze(-1).contiguous()
        q_end_logits = q_end_logits.squeeze(-1).contiguous()
        r_end_logits = r_end_logits.squeeze(-1).contiguous()

        q_start_loss = loss_fct(q_start_logits, q_start)
        r_start_loss = loss_fct(r_start_logits, r_start)
        q_end_loss = loss_fct(q_end_logits, q_end)
        r_end_loss = loss_fct(r_end_logits, r_end)



        loss = q_start_loss + r_start_loss + q_end_loss + r_end_loss

        # calculate loss
        loss.backward()
        # update parameters
        optim.step()

        running_loss += loss.item()
        if batch_id % 50 == 0 and batch_id != 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                batch_id + 1, batch_id, running_loss / 50))
            running_loss = 0.0

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
    evaluate(valid_loader)


# In[23]:


model_name = 'aicup_model_bert_base_bilstm'
torch.save(model.state_dict(), model_name)


# ---
# # Inference

# In[ ]:


# model_name = 'aicup_model_bert_base_bilstm'
# model.load_state_dict(torch.load(model_name))


# In[1]:


df_test = pd.read_csv('Batch_answers - test_data(no_label).csv', encoding="ISO-8859-1")
df_test


# In[3]:


df_test[['q', 'r']] = df_test[['q', 'r']].apply(lambda x: x.str.strip('\"'))
df_test['r'] = df_test['s'] + ':' + df_test['r']
df_test


# ---
# ## Tokenize

# In[7]:


test_data_q = df_test['q'].tolist()
test_data_r = df_test['r'].tolist()


# In[8]:


test_encodings = tokenizer(test_data_q, test_data_r, truncation=True, padding=True)


# ---
# ## Dataset

# In[10]:


test_dataset = qrDataset(test_encodings)


# ---
# ## DataLoader

# In[12]:


test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# ---
# ## Predict

# In[14]:


from tqdm import tqdm

def predict(test_loader):
    predict_pos = []

    model.eval()

    q_sub_output, r_sub_output = [],[]

    loop = tqdm(test_loader, leave=True)
    for batch_id, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        # model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        q_start_logits, r_start_logits, q_end_logits, r_end_logits = torch.split(outputs, 1, 2)

        q_start_logits = q_start_logits.squeeze(-1).contiguous()
        r_start_logits = r_start_logits.squeeze(-1).contiguous()
        q_end_logits = q_end_logits.squeeze(-1).contiguous()
        r_end_logits = r_end_logits.squeeze(-1).contiguous()

        q_start_prdict = torch.argmax(q_start_logits, 1).cpu().numpy()
        r_start_prdict = torch.argmax(r_start_logits, 1).cpu().numpy()
        q_end_prdict = torch.argmax(q_end_logits, 1).cpu().numpy()
        r_end_prdict = torch.argmax(r_end_logits, 1).cpu().numpy()

        for i in range(len(input_ids)):
            predict_pos.append((q_start_prdict[i].item(), r_start_prdict[i].item(), q_end_prdict[i].item(), r_end_prdict[i].item()))

            q_sub = tokenizer.decode(input_ids[i][q_start_prdict[i]:q_end_prdict[i]+1])
            r_sub = tokenizer.decode(input_ids[i][r_start_prdict[i]:r_end_prdict[i]+1])
            
            q_sub_output.append(q_sub)
            r_sub_output.append(r_sub)
    
    return q_sub_output, r_sub_output, predict_pos


# In[15]:


q_sub_output, r_sub_output, predict_pos = predict(test_loader)


# In[16]:


def get_output_post_fn(test, q_sub_output, r_sub_output):
    q_sub, r_sub = [], []
    for i in range(len(test)):

        q_sub_pred = q_sub_output[i].split()
        r_sub_pred = r_sub_output[i].split()

        if q_sub_pred is None:
            q_sub_pred = []
        q_sub_error_index = q_sub_pred.index('[SEP]') if '[SEP]' in q_sub_pred else -1

        if q_sub_error_index != -1:
            q_sub_pred = q_sub_pred[:q_sub_error_index]

        temp = r_sub_pred.copy()
        if r_sub_pred is None:
            r_sub_pred = []
        else:
            for j in range(len(temp)):
                if temp[j] == '[SEP]':
                    r_sub_pred.remove('[SEP]')
                if temp[j] == '[PAD]':
                    r_sub_pred.remove('[PAD]')

        q_sub.append(' '.join(q_sub_pred))
        r_sub.append(' '.join(r_sub_pred))

    return q_sub, r_sub


# In[17]:


q_sub, r_sub = get_output_post_fn(df_test, q_sub_output, r_sub_output)


# In[18]:


df_test['q_sub'] = q_sub
df_test['r_sub'] = r_sub
df_test = df_test.set_index('id')
df_test


# ---
# # To CSV

# In[19]:


df_result = pd.read_csv('submission template.csv')
df_result = df_result.set_index('id')
df_result = pd.merge(df_result, df_test[['q_sub', 'r_sub']], left_index=True, right_index=True)
df_result = df_result.reset_index()
df_result = df_result.drop(['q', 'r'], axis=1)
df_result = df_result.rename(columns={"q_sub": "q", "r_sub": "r"})
# df_result[["q", "r"]] = df_result[["q", "r"]].astype(str)
df_result.update('"' + df_result[['q', 'r']].astype(str) + '"')

df_result.to_csv('submission_bert_base_bilstm.csv', encoding="utf-8", sep=',', index=False)


# In[20]:


df_result[:30]

