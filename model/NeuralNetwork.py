import abc
import torch
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.patience = 0
        self.init_clip_max_norm = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def fit(self, X_train_tid, X_train, y_train,
                  X_dev_tid, X_dev, y_dev):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_func = nn.CrossEntropyLoss()
        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch+1, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            for i, data in enumerate(dataloader):
                with torch.no_grad():
                    batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)

                self.optimizer.zero_grad()
                logit = self.forward(batch_x_tid, batch_x_text)
                loss = loss_func(logit, batch_y)
                loss.backward()
                self.optimizer.step()

                corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                accuracy = 100*corrects/len(batch_y)

                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(i, loss.item(), accuracy, corrects, batch_y.size(0)))
                if i > 0 and i % 100 == 0:
                    self.evaluate(X_dev, y_dev)
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

            self.evaluate(X_dev_tid, X_dev, y_dev)


    def adjust_learning_rate(self, optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr


    def evaluate(self, X_dev_tid, X_dev, y_dev):
        y_pred = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            self.patience += 1


    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        X_test = torch.LongTensor(X_test).cuda()

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)

            logits = self.forward(batch_x_tid, batch_x_text)
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
