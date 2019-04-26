from model import SqueezeNet
import numpy as np
import torch as t
from data.dataset import DogsCats
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# config
data_path = './data/'
batch_size = 32
gpu = True
device = t.device('cuda')
model_path = './checkpoints/squeezenet_epoch1.pth'
result_path = './result.csv'


@t.no_grad()
def test():
    """
    Test stage.
    :return: predicted results.
    """

    # model
    net = SqueezeNet()
    if model_path:  # if pretrained model file exists
        net.load_state_dict(t.load(model_path))
    if gpu and t.cuda.is_available():  # use gpu
        net.to(device)

    # data
    test_data = DogsCats(data_path, test=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    results = []
    for i, (img, id) in tqdm(enumerate(test_dataloader)):
        img = img.to(device)
        out = net(img)
        pred = np.argmax(t.nn.functional.softmax(out, dim=1).detach().cpu().numpy(), axis=1)

        batch_results = [(id_.item(), pred_) for id_, pred_ in zip(id, pred)]

        results += batch_results

    write_results(results, result_path)

    return results


def write_results(results, file_name):
    """
    Write the predicted results(type as 'id  label') to the path.
    :param results: predicted results.
    :param file_name: path to restore results.
    """
    with open(file=file_name, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == "__main__":
    test()

