import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from d2l import torch as d2l
from net import TinySSD
from config import get_config

config = get_config()

device = config['device']
net = TinySSD(num_classes = 1)
net = net.to(device)



def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

# 加载模型的状态字典
net.load_state_dict(torch.load('/home/extend2/user7/Hwj/深度学习代码练习/chapter13/checkpoints/tinyssd_model.pth'))
X = torchvision.io.read_image('/home/extend2/user7/Hwj/深度学习代码练习/img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
output = predict(X)
display(img, output.cpu(), threshold=0.9)
plt.savefig('/home/extend2/user7/Hwj/深度学习代码练习/chapter13/ssd/predicted_banana.jpg')
