import torch

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],[0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
batch_size = 32
device = torch.device('cuda:0')

def get_config():
    return {
        "sizes": sizes,
        "ratios": ratios,
        "num_anchors": num_anchors,
        "batch_size": batch_size,
        "device": device
    }