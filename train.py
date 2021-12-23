from load_data import *
from building import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


epochs = 15
learning_rate = 0.01
momentum = 0.5
log_interval = 25

net = Model()
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]


def train(epoch_n):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out_put = net(data)
        loss = f.nll_loss(out_put, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_n, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch_n-1)*len(train_loader.dataset)))
            torch.save(net.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            out_put = net(data)
            test_loss += f.nll_loss(out_put, target, reduction='sum').item()
            prediction = out_put.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.data.view_as(prediction)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         test_loss, correct, len(test_loader.dataset),
         100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()


######################
# Show training plot #
######################
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='green')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


##################################
# Show test data with prediction #
##################################
with torch.no_grad():
    output = net(example_data)
for i in range(15):
    plt.subplot(5, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
