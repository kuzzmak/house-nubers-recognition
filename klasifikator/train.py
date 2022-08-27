import os
import numpy as np
import torch
import dataloader
import model
epochs = 15
batch_size = 128


def evaluate(name, eval_data, eval_model):
    with torch.no_grad():
        N = 73527
        correct = 0

        for batch_eval_idx, batch_eval_data in enumerate(eval_data):
            eval_inputs, eval_labels = batch_eval_data
            eval_inputs = torch.reshape(eval_inputs, (len(eval_labels), 3, 32, 32))
            eval_inputs = eval_inputs.float()
            eval_labels = torch.reshape(eval_labels, (len(eval_labels),))
            eval_labels = eval_labels.type(torch.LongTensor)
            eval_inputs, eval_labels = eval_inputs.to(device), eval_labels.to(device)

            eval_outputs = eval_model(eval_inputs)

            predicted_labels = torch.argmax(eval_outputs, dim=1)
            correct += (eval_labels == predicted_labels).sum().item()

        accuracy = (correct/N)*100
        print(name + "accuracy = %.2f" % accuracy)

train_data = dataloader.CustomDataset('./data/train_32x32.mat')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.resnet18()
model.to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30)

criterion = torch.nn.CrossEntropyLoss()

print("Number of epochs: ", epochs)

for epoch in range(epochs):
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, labels = batch_data
        inputs = torch.reshape(inputs, (len(labels),3,32,32))
        inputs = inputs.float()
        labels = torch.reshape(labels, (len(labels),))
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 30 == 0:
            print("epoch: {}, batch_loss: {}, idx: {}"
                  .format(epoch, loss.item(), batch_idx))

    scheduler.step()
    evaluate("Train ", train_loader, model)


torch.save(model.state_dict(), "/content/drive/MyDrive/NeuMre/model.pt")