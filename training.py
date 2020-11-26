import torch
from data import get_images_and_labels
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
PATH = "model/trained.txt"

if __name__ == "__main__":
    model = EfficientNet.from_name("efficientnet-b7")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train, test = get_images_and_labels()
    for epoch in range(2):  # loop over the dataset multiple times

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = train.get_images(), train.get_labels().iloc[:,1:2].to_numpy()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = 0.0
        print(labels)
        print(inputs[0])
        for i in range(1, len(inputs) + 1):
            outputs = model(inputs[i - 1])
            loss = criterion(outputs, labels[i])
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss = loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, 1, running_loss))

            loss += running_loss
            running_loss = 0.0
            outputs = []
        loss /= len(inputs)//180
    torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion
            }, PATH)
    print('Finished Training')
