"""
FM模型
"""
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.load_data_util import *

class FmNet(nn.Module):
    """FM模型"""
    def __init__(self, vec_dim, feat_num):
        super(FmNet, self).__init__()
        self.vec_dim = vec_dim
        self.feat_num = feat_num
        self.linear = nn.Linear(self.feat_num, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.vec_dim, self.feat_num), requires_grad=False)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v.t())
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t())

        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return output

    def forward(self, x):
        output = torch.sigmoid(self.fm_layer(x))
        return output


EPOCH = 10
STEP_PRINT = 200
STOP_STEP = 2000

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
LAMDA = 1e-3

base, test = loadData()
VEC_DIM = 10
FEAT_NUM = base.shape[1]-1

fm = FmNet(VEC_DIM, FEAT_NUM)

optimizer = torch.optim.Adam(fm.parameters(), lr=LEARNING_RATE)
loss_func = nn.MSELoss()

def eval_acc(y_pred, y):
    pred_label = [int(p[0]) for p in y_pred]
    acc = accuracy_score(y, pred_label)
    return acc
def eval_auc(y_hat, y):
    auc = roc_auc_score(y, y_hat)
    return auc

def getValTest():
    """split validation data/ test data"""
    for valTest_x, valTest_y in getBatchData(test, batch_size=8000):
        val_x, val_y = valTest_x[:4000], valTest_y[:4000]
        test_x, test_y = valTest_x[4000:], valTest_y[4000:]
        return val_x, val_y, test_x, test_y


def train():
    """"""
    val_x, val_y, test_x, test_y = getValTest()
    step = 0
    last_improved = 0
    stop = False
    print("====== let's train =====")
    for epoch in range(EPOCH):
        print("EPOCH: {}".format(epoch + 1))
        for x_batch, y_batch in getBatchData(base, BATCH_SIZE):
            optimizer.zero_grad()
            train_x = torch.tensor(x_batch, dtype=torch.float)
            train_y = torch.tensor(y_batch, dtype=torch.float)

            v_x = torch.tensor(val_x, dtype=torch.float)

            y_pred = fm(train_x)

            train_loss = loss_func(y_pred, train_y)
            train_loss.backward()
            optimizer.step()


            if step % STEP_PRINT == 0:
                train_acc = eval_acc(y_pred.data.detach().numpy(), y_batch)
                y_val_pred = fm(v_x)
                val_acc = eval_acc(y_val_pred.data.detach().numpy(), val_y)

                msg = 'Iter: {0:>6}, Train acc: {1:>6.4}, Val acc: {2:>6.4}'
                print(msg.format(step, train_acc, val_acc))
            step += 1
            if step - last_improved > STOP_STEP:
                print("No optimization for a long time, auto-stopping...")
                stop = True
                break
        if stop:
            break

train()
