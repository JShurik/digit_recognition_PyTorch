import torch
import torchvision as tv

batch_size_train = 64
batch_size_test = 1000
torch.manual_seed(1)

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.1307,), (0.3081,))
                                   ])

train_loader = torch.utils.data.DataLoader(tv.datasets.MNIST(
    'LOADED_DATA', train=True, download=True, transform=transform),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(tv.datasets.MNIST(
    'LOADED_DATA', train=False, download=True, transform=transform),
     batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch, (example_data, example_targets) = next(examples)

###############################
# Show part of loaded dataset #
###############################

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_data.shape)
# fig = plt.figure()
# for i in range(9):
#    plt.subplot(3, 3, i+1)
#    plt.tight_layout()
#    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#    plt.title("Ground Truth: {}".format(example_targets[i]))
#    plt.xticks([])
#    plt.yticks([])
# plt.show()
