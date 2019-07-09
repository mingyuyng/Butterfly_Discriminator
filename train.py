import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys

import dataloader as dl
import model as md

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


if len(sys.argv) != 4:
    print("Number of augments is wrong! Should get 4 but get %s instead \n" % (len(sys.argv)))
    exit(1)
else:
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    input_len = int(sys.argv[3])

    if input_len != 720 and input_len != 1440:
        print("Wrong length of input, can only be 720 or 1440")
        exit(1)

dataset = dl.dataLoader(in_path)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=200,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)
net = md.CNN(input_len).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([9.0]))
learning_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
print_interval = 10

num_epochs = 4

for epoch in range(num_epochs):
    for i, (frame) in enumerate(train_loader):

        x = frame['intensity'].type(torch.FloatTensor)
        y = frame['label'].type(torch.FloatTensor)
        uinput = Variable(x).to(device)
        utarget = Variable(y).to(device)
        pred_data = net(uinput)
        error_L2 = criterion(pred_data, utarget.unsqueeze(1))

        optimizer.zero_grad()
        error_L2.backward()
        optimizer.step()
        if i % print_interval == 0:
            get_error = extract(error_L2)[0]
            print("Epoch %s: Iter: %s err: %s \n" % (epoch, i, get_error))

    # if epoch % 2 == 0:
    #    Unet.eval()
    #    pred_val = []
    #    for i, (frame) in enumerate(val_loader):
    #        x = frame['intensity'].type(torch.FloatTensor)
    #        y = frame['label'].type(torch.FloatTensor)
    #        uinput = Variable(x)
    #        utarget = Variable(y)
    #        pred_data = torch.sigmoid(Unet(uinput)).squeeze(1)
    #        e = (pred_data > 0.5 - utarget).detach().numpy()
    #        pred_val.append(e)
    #    pred_val = np.stack(pred_val)
    #    miss = np.sum(pred_val == -1) / 500
    #    false_alarm = np.sum(pred_val == 1) / 7500
     #   print("Epoch %s: FA: %s MI: %s \n" % (epoch, false_alarm, miss))
    #    Unet.train()

torch.save(net.state_dict(), out_path)
