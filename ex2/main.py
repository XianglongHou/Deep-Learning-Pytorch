import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# load data
# Define the transformation to be applied to the dataset
# Define the transformation to be applied to the dataset

# load and split data
global_val_losses = []
global_train_losses = []


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download the CIFAR-10 dataset and apply the transformation
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    train_ratio = 0.85
    train_num = int(len(trainset) * train_ratio)
    trainset, valset = torch.utils.data.random_split(trainset, [train_num, len(trainset) - train_num])

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # cnn
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.dropout = nn.Dropout(p=0.4)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.bn4 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
            x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
            x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 128 * 4 * 4)
            x = self.dropout(x)
            x = nn.functional.relu(self.bn4(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def train_loop(dataloader, model, optimizer, criterion=nn.CrossEntropyLoss()):
        global global_train_losses
        model.train()
        running_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        global_train_losses.append(running_loss / len(dataloader))  # 据此估算交叉熵损失
        return running_loss / len(dataloader)

    def val_loop(dataloader, model, criterion=nn.CrossEntropyLoss()):
        global global_val_losses
        model.eval()  # 设置模型为评估模式, 不会自动求导
        total_loss = 0
        with torch.no_grad():  # 不需要计算梯度
            for inputs, labels in dataloader:
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
        global_val_losses.append(total_loss / len(dataloader))
        return total_loss / len(dataloader)

    def test_loop(dataloader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # index
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
        # print('测试集上的准确率： %d %%' % (100 * correct / total))

    model = CNN().to("cuda")
    best_model = model

    lr = 0.01
    num_epochs = 30
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # early stop

    best_val_loss = 1e10
    count = 0
    patience = 8
    epoch_time = 0
    # 训练CNN模型
    for epoch in range(num_epochs):
        run_loss = train_loop(train_loader, model, optimizer)
        val_loss = val_loop(val_loader, model)

        # 训练CNN模型

        # early stop

        if val_loss < best_val_loss or epoch < num_epochs / 4 :
            best_val_loss = val_loss
            best_model = model
            count = 0

        else:
            count += 1
            print(count)


        if count == patience:
            epoch_time = epoch
            break

        print("epoch:%d, train_loss: %.3f, val_loss: %.3f" % (epoch, run_loss, val_loss))

    # 测试CNN模型
    plt.plot(range(epoch_time+1), global_train_losses, label='train_loss')
    plt.plot(range(epoch_time+1), global_val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    print("loss on val_set: ", val_loop(val_loader, model))
    print("accuracy on val set: ", test_loop(val_loader, model))
    print("accuracy on test set: ", test_loop(test_loader, model))






if __name__ == '__main__':
    main()
