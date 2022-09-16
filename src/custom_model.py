import numpy as np
import torch
import tqdm
import transformers

from sklearn.metrics import accuracy_score, f1_score

class BertClassifier(torch.nn.Module):
    def __init__(self, n_classes, bert_model_name, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_model_name)
        embd_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(embd_dim, n_classes)
        self.relu = torch.nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        _, cls_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            return_dict=False)
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)
        return final_output 

def train(model, train_dataset, val_dataset, batch_size, learning_rate, epochs):
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    for epoch in range(epochs):
        
        # Train 
        total_train_loss = 0
        
        for X, Y in tqdm.tqdm(train_dataloader):
            Y = Y.to(device)
            mask = X['attention_mask'].to(device)
            input_ids = X['input_ids'].squeeze(1).to(device)
            
            output = model(input_ids, mask)
            
            batch_train_loss = criterion(output, Y.long())
            total_train_loss += batch_train_loss.item()
            
            model.zero_grad()
            batch_train_loss.backward()
            optimizer.step()
        
        # Validate
        total_val_loss = 0
        
        with torch.no_grad():
            for X, Y in val_dataloader:
                Y = Y.to(device)
                mask = X['attention_mask'].to(device)
                input_ids = X['input_ids'].squeeze(1).to(device)
            
                output = model(input_ids, mask)
            
                batch_val_loss = criterion(output, Y.long())
                total_val_loss += batch_val_loss.item()
        
        # Show progress
        str_out = f"Epoch {epoch+1} | "
        str_out += f"Train Loss: {total_train_loss/len(train_dataset):.4f} | "
        str_out += f"Val Loss: {total_val_loss/len(val_dataset):.4f}"
        print(str_out)
    
    return model

def predict(model, test_dataset):
    test_dataloader = torch.utils.data.DataLoader(test_dataset)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with torch.no_grad():
        predictions = []
        for i, (X,Y) in enumerate(test_dataloader):
            Y = Y.to(device)
            mask = X['attention_mask'].to(device)
            input_ids = X['input_ids'].squeeze(1).to(device)
            output = model(input_ids, mask)
            predictions.append(output)
    
    predictions = torch.cat(predictions, axis=0)
    predictions = predictions.cpu().numpy()
    predictions = np.argmax(predictions, axis=1)
    
    return predictions

def compute_metrics(ytrue, ypred):
    f1 = f1_score(ytrue, ypred, average='weighted')
    acc = accuracy_score(ytrue, ypred)
    return {"accuracy":acc, "f1":f1}

