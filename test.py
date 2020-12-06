import torch
from data import get_images_and_labels
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

PATH = "model/trained.txt"

if __name__ == "__main__":
    model = EfficientNet.from_pretrained("efficientnet-b7")
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train, test = get_images_and_labels()
    train_set = train.get_images()
    testloader = torch.utils.data.DataLoader(test.get_images(), batch_size=4,
                                              shuffle=False, num_workers=2)
    al = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        for i in range(len(outputs)):
            mx = 0
            get = 0
            for k in range(len(outputs[i])):
                if outputs[i][k] > mx:
                    mx = outputs[i][k]
                    get = k
            print(mx, k)
            if get == labels[i]:
                al += 1

    print(al/(len(testloader)*4))
