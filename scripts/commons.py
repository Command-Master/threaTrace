import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import NeighborSampler
import time, sys
import os.path as osp

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def show(*s):
    for x in s:
        print(str(x) + ' ', end='')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))


class TestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(TestDataset, self).__init__('/tmp/TestDataset')
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, concat=False):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 32, normalize=False, concat=concat)
        self.conv2 = SAGEConv(32, out_channels, normalize=False, concat=concat)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size)
        return F.log_softmax(x, dim=1)


def test(loader, model, mask, data, thre, fp, tn):
    model.eval()

    correct = 0
    total_loss = 0

    for data_flow in loader(mask):
        out = model(data.x.to(device), data_flow.to(device))
        pred = out.max(1)[1]
        pro = F.softmax(out, dim=1)
        pro1 = pro.max(1)
        for i in range(len(data_flow.n_id)):
            pro[i][pro1[1][i]] = -1
        pro2 = pro.max(1)
        for i in range(len(data_flow.n_id)):
            if pro1[0][i] / pro2[0][i] < thre:
                pred[i] = 100
        for i in range(len(data_flow.n_id)):
            if data.y[data_flow.n_id[i]] != pred[i]:
                fp.append(int(data_flow.n_id[i]))
            else:
                tn.append(int(data_flow.n_id[i]))
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()

    return total_loss / mask.sum().item(), correct / mask.sum().item()


def getFeature():
    feature_num = 0
    label_num = 0
    f = open('../models/feature.txt', 'r')
    for _ in f:
        feature_num += 1
    feature_num *= 2
    f.close()
    f = open('../models/label.txt', 'r')
    for _ in f:
        label_num += 1
    f.close()
    return feature_num, label_num


def detect(model_list, data, feature_num, label_num, thre, batch_size, update_benign_callback, raise_alert_callback):
    loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=batch_size, shuffle=True,
                             add_self_loops=True)
    fp = []
    tn = []

    test_acc = 0
    model = SAGENet(feature_num, label_num).to(device)
    for j in model_list:
        loop_num = 0
        base_model = str(j)
        while 1:
            model_path = '../models/' + base_model + '_' + str(loop_num)
            if not osp.exists(model_path): break
            model.load_state_dict(torch.load(model_path))
            fp.clear()
            tn.clear()
            loss, test_acc = test(loader, model, data.test_mask, data, thre, fp, tn)
            for i in tn:
                data.test_mask[i] = False
                update_benign_callback(i)
            if test_acc == 1: break
            loop_num += 1
        if test_acc == 1: break

    for i in fp:
        raise_alert_callback(i)


def extract_data(p, id_map_t):
    id_map = {}
    id_map_t.clear()
    ts = {}
    test_mask = []
    x = []
    y = []
    edge_s = []
    edge_e = []
    this_ts = 0
    try:
        node_num = int(p.stdout.readline())
    except Exception as e:
        show(f'Error: {e}')
        return None, 0
    if node_num == -1: return None, 0
    for i in range(node_num):
        line = bytes.decode(p.stdout.readline())
        line = list(map(int, line.strip('\n').split(' ')))
        id_map[line[0]] = i
        id_map_t[i] = line[0]
        y.append(line[1])
        if line[2] == 1:
            test_mask.append(True)
        else:
            test_mask.append(False)
        x.append(line[3:len(line) - 1])
        ts[i] = line[len(line) - 1] / 1000
        if ts[i] > this_ts: this_ts = ts[i]
    edge_num = int(p.stdout.readline())
    for i in range(edge_num):
        line = bytes.decode(p.stdout.readline())
        line = list(map(int, line.strip('\n').split(' ')))
        edge_s.append(id_map[line[0]])
        edge_e.append(id_map[line[1]])
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index, test_mask=test_mask, train_mask=test_mask)
    dataset = TestDataset([data])
    data = dataset[0]
    return data, this_ts


def train(model, loader, data, device, optimizer):
    model.train()

    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()
