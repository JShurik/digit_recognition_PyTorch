from load_data import *
from building import *
import torch
import numpy as np
import matplotlib.pyplot as plt

model = Model()
model.load_state_dict(torch.load('results\model.pth'))


def predict_digit(img):
    img = np.array(img)
    img = tv.transforms.ToTensor()(img)
    img = img.unsqueeze(1)
    with torch.no_grad():
        model.eval()
        plt.title('crutch')
        plt.imshow(img.reshape(28, 28, 1))
        # plt.show()
        output = model(img)
        output = np.argmax(output)
        model.eval()
    return output
