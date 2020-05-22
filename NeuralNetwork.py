import numpy as np
import pandas as pd
import transformers as trans
import torch
from sklearn import metrics
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from SiameseNetwork import Siemese


train_file = 'train.csv'
train_data = []
model_class, tokenizer_class, pretrained_weights = (trans.BertModel, trans.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
bert = model_class.from_pretrained('bert-base-uncased')
max_length = 128
batch_size = 128
HIDDEN_DIM = 512
NUM_LAYERS = 1
OUTPUT_DIM = 1
BIDIRECTIONAL = True
tokenized_text = []
attention_mask = []
model = Siemese(bert,HIDDEN_DIM,OUTPUT_DIM,NUM_LAYERS,BIDIRECTIONAL)
optimizer = optim.Adam(model.parameters())

# Process Training and Validation Data
def process_data():
    df = pd.read_csv(train_file)
    df[df.isnull().any(axis=1)]
    df = df.dropna(how='any')
    questions1 = df['question1']
    question2 = df['question2']
    output = df['is_duplicate']
    for i in question2.keys():
        train_data.append((questions1[i],question2[i]))

# Create Attention Mask for the Training Data
def create_mask():
    k = 0
    for (x,y) in train_data:
        tx = tokenizer.encode(x, add_special_tokens=True)
        ty = tokenizer.encode(y, add_special_tokens=True)
        if(len(tx) < max_length):
            tx = tx + [0]*(max_length-len(tx))
        if(len(ty) < max_length):
            ty = ty + [0]*(max_length-len(ty))

        if(len(tx) > max_length):
            tx = tx[:max_length]
        if(len(ty) > max_length):
            ty = ty[:max_length]
        ax = [float(i>0) for i in tx]
        ay = [float(i>0) for i in ty]

        tokenized_text.append((tx,ty))
        attention_mask.append((ax,ay))
        k += 1
        if(k % 10000 == 0): print("{0} Processed".format(k))

# Create Dataloader
def create_data_loaders():
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(tokenized_text, output.values, random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_mask, tokenized_text,random_state=2018, test_size=0.1)

    # create torch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels).float()
    validation_labels = torch.tensor(validation_labels).float()
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)


    # Create an iterator of our data with torch DataLoader 
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader,validation_dataloader

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# Train the Siamese Network 
def train(train_dataloader):
    i = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        data, mask, labels = batch
        question1 = data[:,0,:]
        question2 = data[:,1,:]
        question1_mask = mask[:,0,:]
        question2_mask = mask[:,1,:]

        predictions = model(question1,question2,question1_mask,question2_mask)
        acc = binary_accuracy(predictions, labels[:2])

        loss = criterion(predictions, labels)
        loss.backward()
        
        if(i % 500 == 0): print("i = {0}\t Accuracy = {1}\t Loss = {2}\t".format(loss.item(),acc.item()))

# Evaluate Network using validation set
def evaluate(validation_loader):
    pred = []
    true = []
    with torch.no_grad():

        for batch in validation_loader():
            data, mask, labels = batch
            question1 = data[:,0,:]
            question2 = data[:,1,:]
            question1_mask = mask[:,0,:]
            question2_mask = mask[:,1,:]

            predictions = model(question1,question2,question1_mask,question2_mask)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            pred.append(rounded_preds.numpy())
            true.append(labels.numpy())

    print('F1: {}'.format(metrics.f1_score(true, pred, average="samples")))
    print('Precision: {}'.format(metrics.precision_score(true, pred, average="samples")))
    print('Recall: {}'.format(metrics.recall_score(true, pred, average="samples")))


def main():
    process_data() # Process Data
    create_mask() # Create Attention Mask for training data
    train_dataloader,validation_dataloader = create_data_loaders() # Create Validation and Train Data Loaders
    train(train_dataloader) # Train Network
    evaluate(validation_dataloader) # Evaluate Network

main()




