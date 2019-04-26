import torch as t
from model import SqueezeNet
from data.dataset import DogsCats
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

# config
data_path = './data/'
batch_size = 32
gpu = True
device = t.device('cuda')
model_path = None  # './checkpoints/squeezenet_0416_20_53_03.pth'

epochs = 5
lr_decay = 0.5
weight_decay = 0e-5


def train():
    """
    Train stage.
    """

    # dataset
    train_data = DogsCats(data_path, train=True)
    val_data = DogsCats(data_path, train=False)
    train_dataloader = DataLoader(train_data, batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size,
                                shuffle=False, num_workers=4)

    # model
    net = SqueezeNet()
    if model_path:  # if pretrained model file exists
        net.load_state_dict(t.load(model_path))
    if gpu and t.cuda.is_available():  # use gpu
        net.to(device)

    lr = 0.001
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.net.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    # meters
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10

    # train
    for epoch in range(epochs):

        loss_meter.reset()

        for i, (img, label) in tqdm(enumerate(train_dataloader)):
            # train model
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = net(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

        t.save(net.state_dict(), './checkpoints/squeezenet_epoch{}.pth'.format(epoch))

        # validate and visualize
        val_accuracy = val(net, val_dataloader)
        print("epoch:{epoch}, lr:{lr}, loss:{loss}, val_accuracy:{val_accuracy}".format(
                epoch=epoch, lr=lr, loss=loss_meter.value()[0], val_accuracy=val_accuracy))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloader):
    """

    :param model:
    :param dataloader:
    :return:
    """
    model.eval()
    total = 0
    correct = 0
    for ii, (img, label) in tqdm(enumerate(dataloader)):
        img = img.to(device)
        out = model(img)

        _, predicted = t.max(out, 1)
        total += img.size(0)
        correct += predicted.cpu().eq(label).sum()

    acc = 1.0 * correct.numpy() / total

    return acc


if __name__ == "__main__":
    train()
