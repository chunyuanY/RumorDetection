import abc
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

seed = 0
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.patience = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss_func = nn.CrossEntropyLoss()

    @abc.abstractmethod
    def forward(self):
        pass


    def train_step(self, i, data):
        with torch.no_grad():
            batch_tid, batch_source, batch_replies, batch_y = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()
        logit = self.forward(batch_tid, batch_source, batch_replies)
        loss = self.loss_func(logit, batch_y)
        loss.backward()
        self.optimizer.step()

        corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
        accuracy = 100.0*corrects / len(batch_y)

        print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(i, loss.item(), accuracy, corrects, batch_y.size(0)))
        return loss, accuracy


    def fit(self, X_train_tid, X_train_source, X_train_replies, y_train,
                  X_dev_tid, X_dev_source, X_dev_replies, y_dev):

        if torch.cuda.is_available():
            self.cuda()

        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['reg'])

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train_source = torch.LongTensor(X_train_source)
        X_train_replies = torch.LongTensor(X_train_replies)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train_source, X_train_replies, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch+1, "/", self.config['epochs'])
            self.train()
            avg_loss, avg_acc = 0, 0
            for i, data in enumerate(dataloader):
                loss, accuracy = self.train_step(i, data)
                if i > 0 and i % 100 == 0:
                    self.evaluate(X_dev_tid, X_dev_source, X_dev_replies, y_dev)
                    self.train()

                avg_loss += loss.item()
                avg_acc += accuracy

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

            cnt = y_train.size(0) // batch_size + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))
            if epoch >= 10 and self.patience > 3:
                print("Reload the best model...")
                self.load_state_dict(torch.load(self.config['save_path']))
                now_lr = self.adjust_learning_rate(self.optimizer)
                print(now_lr)
                self.patience = 0

            self.evaluate(X_dev_tid, X_dev_source, X_dev_replies, y_dev)


    def adjust_learning_rate(self, optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr


    def evaluate(self, X_dev_tid, X_dev_source, X_dev_replies, y_dev):
        y_pred = self.predict(X_dev_tid, X_dev_source, X_dev_replies)
        acc = accuracy_score(y_dev, y_pred)
        print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))

        if acc > self.best_acc:
            self.best_acc = acc
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            self.patience += 1


    def predict(self, X_test_tid, X_test_source, X_test_replies):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid)
        X_test_source = torch.LongTensor(X_test_source)
        X_test_replies = torch.LongTensor(X_test_replies)

        dataset = TensorDataset(X_test_tid, X_test_source, X_test_replies)
        dataloader = DataLoader(dataset, batch_size=128)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_tid, batch_source, batch_replies = (item.cuda(device=self.device) for item in data)

            logits = self.forward(batch_tid, batch_source, batch_replies)
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
