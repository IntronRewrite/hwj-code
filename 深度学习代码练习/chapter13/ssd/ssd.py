import torch 
from torch import nn
from d2l import torch as d2l
from net import TinySSD
from config import get_config

config = get_config()

batch_size = config['batch_size']
device = config['device']

train_iter,_ =d2l.load_data_bananas(batch_size)
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction = 'none')

    
def calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
    batch_size,num_classes = cls_preds.shape[0],cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1,num_classes),cls_labels.reshape(-1)).reshape(batch_size,-1).mean(dim=1)
    bbox = bbox_loss(bbox_preds*bbox_masks,bbox_labels*bbox_masks).mean(dim=1)
    return cls +bbox

def cls_eval(cls_preds,cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds,bbox_labels,bbox_masks):
    return float((torch.abs(bbox_labels - bbox_preds)*bbox_masks).sum())

net = TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(),lr=0.2,weight_decay=5e-4)
num_epochs,timer = 20,d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net =net.to(device)
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    net.train()
    for features,target in train_iter:
        timer.start()
        trainer.zero_grad()
        X,Y = features.to(device),target.to(device)
        anchors,cls_preds,bbox_preds =net(X)
        bbox_labels,bbox_masks,cls_labels = d2l.multibox_target(anchors,Y)
        l = calc_loss(cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds,cls_labels),cls_labels.numel(),
                   bbox_eval(bbox_preds,bbox_labels,bbox_masks),bbox_labels.numel())
    cls_err,bbox_mae = 1 - metric[0]/metric[1],metric[2]/metric[3]
    animator.add(epoch+1,(cls_err,bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')

# 保存模型的状态字典
torch.save(net.state_dict(), '/home/extend2/user7/Hwj/深度学习代码练习/chapter13/checkpoints/tinyssd_model.pth')

