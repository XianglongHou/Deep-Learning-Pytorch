import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification

print("*************BERT**************")
## PART1: BERT
train_df = pd.read_csv("processed_data/train_df.csv")
test_df = pd.read_csv("processed_data/test_df.csv")

#1. encode text using bert tokenzier and pack, and later we will also use the packed dataset to train RNN(only input_ids and label)
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        label = row['target']
        #编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
# 1. text：输入文本，需要进行编码的字符串。
# 2. add_special_tokens：布尔值，表示是否在文本中添加特殊的开始和结束标记（例如，[CLS]和[SEP]）。
# 3. max_length：整数，表示编码后的序列的最大长度。如果输入文本的长度超过此值，将被截断；如果长度小于此值，将使用填充符号进行填充。
# 4. return_token_type_ids：布尔值，表示是否返回token_type_ids。在这个例子中，我们不需要token_type_ids，所以设置为False。
# 5. padding：字符串，表示填充策略。在这个例子中，我们使用'max_length'策略，这意味着所有序列都将被填充或截断到max_length的长度。
# 6. truncation：布尔值，表示是否对超过max_length的序列进行截断。
# 7. return_attention_mask：布尔值，表示是否返回注意力掩码。注意力掩码用于区分输入序列中的实际单词和填充符号。
# 8. return_tensors：字符串，表示返回的张量类型。在这个例子中，我们使用'pt'，表示返回PyTorch张量。
        #编码
        return {
            'input_ids': encoding['input_ids'].flatten(), #输入序列
            'attention_mask': encoding['attention_mask'].flatten(), #注意力掩码
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 2. read the model from folder
save_dir = "/model/shannonhow/Bert_and_RNN/model_BERT/"
tokenizer = BertTokenizer.from_pretrained(save_dir)
bert_model = BertForSequenceClassification.from_pretrained(save_dir, num_labels=2)

#3. 创建数据集实例
train_dataset = SentimentDataset(train_df, tokenizer, max_length=256) #256可以控制
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
test_dataset = SentimentDataset(test_df, tokenizer, max_length=256) #256可以控制
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=True)

#4. Fine-tune the BERT model on the Sentiment dataset
num_epochs = 4
device = 'cuda'
model = bert_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
for epoch in range(num_epochs):
    print("epoch:",epoch)
    count = 0
    for batch in train_dataloader:
        count+=1
        if count%78 == 0:
          print(count/391) #打印训练进度
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

#5. test accuracy
total_num = 0
correct_num = 0
for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    label = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    probs = torch.nn.functional.softmax(outputs[0], dim=1)
    predict_category = torch.argmax(probs,dim = 1)
    correct_num += torch.eq(predict_category,label).sum().item()
    total_num += len(batch['labels'])

print("total_test_num: ", total_num)
print("correct_num: ",correct_num)
print("Accuracy: ", correct_num/total_num)






print("*************RNN***********************")
# PART2: RNN model(with LSTM), we still use class SentimentDataset to encode and pack the dataset. 
# 1. RNN(with LSTM) class
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(embedded, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 2. 训练模型
vocab_size = tokenizer.vocab_size
input_size = vocab_size # 词汇表大小
hidden_size = 128
output_size = 2
num_layers = 1
model = RNN_LSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    count = 0
    print("epoch: ",epoch)
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count+=1
        if count%78 == 0:
            print(count/391,"Loss: ", loss.item())

#4.test
total_num = 0
correct_num = 0
for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    label = batch['labels'].to(device)
    outputs = model(input_ids)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    predict_category = torch.argmax(probs,dim = 1)
    correct_num += torch.eq(predict_category,label).sum().item()
    total_num += len(batch['labels'])

print("total_test_num: ", total_num)
print("correct_num: ",correct_num)
print("Accuracy: ", correct_num/total_num)