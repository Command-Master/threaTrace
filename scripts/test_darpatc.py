from torch_geometric.data import NeighborSampler

from commons import SAGENet, device, test

from data_process_test import *

from resource import getrusage, RUSAGE_SELF

import argparse, os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='0')
parser.add_argument('--scene', type=str, default='')
parser.add_argument('--pretrained', action='store_true')
args = parser.parse_args()
assert args.scene in ['cadets', 'trace', 'theia', 'fivedirections']

if args.pretrained:
    model_path = f'../example_models/darpatc/{args.scene}/'
    output_path = f'../outputs/{args.scene}_pretrained/'
else:
    model_path = f'../outputs/{args.scene}_train/'
    output_path = f'../outputs/{args.scene}_test/'

show(f'Resource usage start: {getrusage(RUSAGE_SELF)}')

thre_map = {"cadets": 1.5, "trace": 1.0, "theia": 1.5, "fivedirections": 1.0}
b_size = 5000
nodeA = []
path = '../graphchi-cpp-master/graph_data/darpatc/' + args.scene + '_test.txt'
graphId = 1
show('Start testing graph ' + str(graphId) + ' in model ' + str(args.model))
data1, feature_num, label_num, adj, adj2, nodeA, _nodeA, _neibor = MyDatasetA(model_path, path, args.model)

dataset = TestDatasetA(data1, args.scene)
data = dataset[0]

loader = NeighborSampler(data, size=[1.0, 1.0], num_hops=2, batch_size=b_size, shuffle=False, add_self_loops=True)

model = SAGENet(feature_num, label_num).to(device)

thre = thre_map[args.scene]

loop_num = 0
model_map = {0: 0}
for j in range(1):
    test_acc = 0
    args.model = model_map[j]
    while 1:
        if loop_num > 100: break
        model_path_ = model_path + 'model_' + str(loop_num)
        if not osp.exists(model_path_):
            loop_num += 1
            continue
        model.load_state_dict(torch.load(model_path_))

        fp = []
        tn = []
        loss, test_acc = test(loader, model, data.test_mask, data, thre, fp, tn)
        show(str(loop_num) + '  loss:{:.4f}'.format(loss) + '  acc:{:.4f}'.format(test_acc) + '  fp:' + str(len(fp)))
        for i in tn:
            data.test_mask[i] = False
        if test_acc == 1: break
        loop_num += 1
    if test_acc == 1: break
fw = open(output_path + 'alarm.txt', 'w')
fw.write(str(len(data.test_mask)) + '\n')
for i in range(len(data.test_mask)):
    if data.test_mask[i]:
        fw.write('\n')
        fw.write(str(i) + ':')
        neibor = set()
        if i in adj.keys():
            for j in adj[i]:
                neibor.add(j)
                if not j in adj.keys(): continue
                for k in adj[j]:
                    neibor.add(k)
        if i in adj2.keys():
            for j in adj2[i]:
                neibor.add(j)
                if not j in adj2.keys(): continue
                for k in adj2[j]:
                    neibor.add(k)

        for j in neibor:
            fw.write(' ' + str(j))

fw.close()

show('Finish testing graph ' + str(graphId) + ' in model ' + str(args.model))

show(f'Resource usage: {getrusage(RUSAGE_SELF)}')