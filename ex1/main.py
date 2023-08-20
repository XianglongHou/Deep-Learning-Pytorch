import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# %% md
## Class FNN
# %%
class myFNN(nn.Module):
    def __init__(self, hidden_sizes=[10], activate_function='relu'):
        """

        :param hidden_sizes: numbers of nodes of each hidden layers (list)
        :param activate_function: 可选relu，sigmoid, tanh
        """
        super(myFNN, self).__init__()
        # 默认精度是float，要转化成double
        self.input_layer = nn.Linear(1, hidden_sizes[0], dtype=torch.double)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], dtype=torch.double))
        self.output_layer = nn.Linear(hidden_sizes[-1], 1, dtype=torch.double)
        self.activate_function = activate_function

    def forward(self, x):
        if self.activate_function == 'relu':
            x = torch.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = torch.relu(layer(x))
            x = self.output_layer(x)
            return x
        elif self.activate_function == 'sigmoid':
            x = torch.sigmoid(self.input_layer(x))
            for layer in self.hidden_layers:
                x = torch.sigmoid(layer(x))
            x = self.output_layer(x)
            return x
        elif self.activate_function == 'tanh':
            x = torch.tanh(self.input_layer(x))
            for layer in self.hidden_layers:
                x = torch.tanh(layer(x))
            x = self.output_layer(x)
            return x
        else:
            raise ValueError("you can only input activate function 'relu', 'sigmoid', 'tanh' ")


# %% md
## generate, split and data and minibatch
# %%
def func(x):
    return np.log2(x) + np.cos(np.pi * x / 2)


def load_data(N=1000, train_rate=0.7, val_rate=0.15, test_rate=0.15, batch_size=64):
    np.random.seed(10)
    x = np.random.uniform(1, 16, size=(N, 1))
    y = func(x)
    inputs = torch.tensor(x, device='cuda', dtype=torch.double)
    targets = torch.tensor(y, device='cuda', dtype=torch.double)
    dataset = torch.utils.data.TensorDataset(inputs, targets)

    num_train = int(N * train_rate)
    num_val = int(N * val_rate)
    num_test = int(N * test_rate)

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# %% md
## define one training/val/test loop
# %%
global_train_losses = []
global_val_losses = [1e3]


# 设置全局变量，方便可视化作图

def train_loop(data_loader, model, optimizer, criterion=nn.MSELoss()):
    global global_train_losses
    model.train()  # Batch Normalization和Dropout
    total_loss = 0

    for inputs, targets in data_loader:
        batch_size = len(inputs)
        optimizer.zero_grad()  # 在每个batch之前清除梯度，加速
        outputs = model(inputs)  # 前向计算
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        total_loss += loss.item() * batch_size
    print('Loss: {:.7f}'.format(total_loss / len(data_loader.dataset)))
    global_train_losses.append(total_loss / len(data_loader.dataset))


# 定义验证循环
def val_loop(data_loader, model, criterion=nn.MSELoss()):
    global global_val_losses
    model.eval()  # 设置模型为评估模式, 不会自动求导
    total_loss = 0
    with torch.no_grad():  # 不需要计算梯度
        for inputs, targets in data_loader:
            batch_size = len(inputs)
            # inputs = inputs.view(batch_size, 1)  # 这样才能输入到 输入层为1的网络中
            # targets = targets.view(batch_size, 1)
            outputs = model(inputs)  # 前向计算
            total_loss += criterion(outputs, targets).item() * batch_size
    global_val_losses.append(total_loss / len(data_loader.dataset))


# 定义测试循环
def test_loop(data_loader, model, criterion=nn.MSELoss()):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():  # 不需要计算梯度
        for inputs, targets in data_loader:
            batch_size = len(inputs)
            # inputs = inputs.view(batch_size, 1)
            # targets = targets.view(batch_size, 1)
            outputs = model(inputs)  # 前向计算
            total_loss += criterion(outputs, targets).item() * batch_size
    print('Test Loss: {:.7f}'.format(total_loss / len(data_loader.dataset)))


# %% md
## main training function
# %%
'''
这里调数据集大小
'''
N = 10000


def main_train(hidden_sizes=[128,128,128,128,128,128], batch_size=64, max_num_epochs=400,
               optimizer_type='Adam', lr=0.001,activate_function = 'tanh',  myplot=False):
    '''

    :param hidden_sizes:  隐藏层节点大小
    :param batch_size:  分成batch的大小
    :param num_epochs: epoch次数
    :param optimizer_type:  优化方式，目前只能输入‘Adam’和'SGD'两种
    :param lr: 学习率
    :param myplot: 是否可视化
    :return: 训练集上的误差，用于调节超参数
    '''
    # basic settings
    global global_train_losses
    global global_val_losses
    global_train_losses = []
    global_val_losses = [1e3]
    train_loader, val_loader, test_loader = load_data(N=N, batch_size=batch_size)
    model = myFNN(hidden_sizes=hidden_sizes,activate_function= activate_function).to('cuda')
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    #for early stop
    count = 0
    best_val_loss = 1e10
    best_model = None
    patience = 30
    epoch_num = 0

    #training
    for epoch in range(max_num_epochs):
        train_loop(train_loader, model, optimizer, criterion)
        val_loop(val_loader, model)
        epoch_num += 1
        # +早停 训练集上误差连续5次增加则停止循环
        if global_val_losses[-1] > best_val_loss:
            count += 1
        else:
            best_val_loss = global_val_losses[-1]
            # best_model = model
            count = 0
        # if count == patience:
        #     break

    # model = best_model

    print("epoch times", epoch_num)
    test_loop(test_loader, model, criterion)

    if myplot == True:
        with torch.no_grad():
            for inputs, targets in test_loader:
                plt.scatter(inputs.cpu(), targets.cpu(), s=12, color='red', label='target')
                plt.scatter(inputs.cpu(), model(inputs).cpu(),s=5, color='yellow', label='prediction')

        plt.legend(['target','prediction']);
        plt.show()

        plt.plot(range(epoch_num), global_train_losses, label='train_loss')
        global_val_losses.pop(0)
        plt.plot(range(epoch_num), global_val_losses, label='val_loss')
        plt.ylim(0,0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    return global_val_losses[-1]




# %% md
## parameter optimization
# %%
# 调参，从训练集上的误差和时间两方面考虑
import time


# 调参，从训练集上的误差和时间两方面考虑
import time


def parameter_optim():
    batch_sizes = [32, 64, 128]
    optimizer_types = ['SGD', 'Adam']
    hidden_sizes = [[512], [256, 256], [128, 128, 128, 128], [64, 64, 64, 64, 64, 64, 64, 64]]
    num_epochs = [100, 500, 1000, 3000]
    lrs = [0.1, 0.01, 0.001]
    activate_functions = ['tanh','relu','sigmoid']
    ## 层数和batch大小
    for activate_function in activate_functions:
        for hidden_size in hidden_sizes:
            print("*******************************")
            print('activate function=', activate_function, " hidden_sizes=", hidden_size)
            start_time = time.time()
            error = main_train(activate_function=activate_function,hidden_sizes=hidden_size)
            end_time = time.time()
            run_time = end_time - start_time
            print("run time：{:.2f} s".format(run_time))
            print("error on val_set:", error)

    for optimizer_type in optimizer_types:
        for lr in lrs:
            print("*******************************")
            print('optimizer types = ', optimizer_type, " learning rate=", lr)
            start_time = time.time()
            error = main_train(optimizer_type=optimizer_type, lr=lr)
            end_time = time.time()
            run_time = end_time - start_time
            print("run time：{:.2f} s".format(run_time))
            print("error on val_set:", error)

# %%
# parameter_optim()
# #%%
# parameter_optim()
# %%
if __name__ == '__main__':
    main_train(activate_function='tanh', myplot=True)

# %%
