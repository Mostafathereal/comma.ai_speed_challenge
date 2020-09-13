import numpy as np
import torch
from inception_tl import *

cuda = torch.device("cuda")

speedNet = InceptionSpeed().to(cuda)
print(speedNet)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(speedNet.parameters(), lr = 0.001, betas = (0.9, 0.999))

inputs = torch.load("/home/mostafathereal/Desktop/comma.ai_speed_challenge/data1/train.pt")
labels = torch.load("/home/mostafathereal/Desktop/comma.ai_speed_challenge/data1/train_labels.pt")

inputs = inputs.type(torch.cuda.FloatTensor)

for epoch in range(30):

    running_loss = 0.0
    for i in range(len(labels)):

        optimizer.zero_grad()

        # print(np.shape((inputs[i])))
        speedNet.eval()
        outputs = speedNet(inputs[i])
        speedNet.train()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/len(labels)))


print("finished training")