import argparse

parser = argparse.ArgumentParser()
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

print(model_path)

f_gt = open(model_path + 'groundtruth_nodeId.txt', 'r')
f_alarm = open(output_path + 'alarm.txt', 'r')

eps = 1e-10

f = open(model_path + 'id_to_uuid.txt', 'r')
node_map = {}
for line in f:
    line = line.strip('\n').split(' ')
    node_map[int(line[0])] = line[1]
f.close()

gt = {}
for line in f_gt:
    gt[int(line.strip('\n').split(' ')[0])] = 1

ans = []
ans_f = {}

for line in f_alarm:
    if line == '\n': continue
    if not ':' in line:
        tot_node = int(line.strip('\n'))
        for i in range(tot_node):
            ans.append('tn')
        for i in gt:
            ans[i] = 'fn'
        continue
    line = line.strip('\n')
    a = int(line.split(':')[0])
    b = line.split(':')[1].strip(' ').split(' ')
    flag = 0
    for i in b:
        if i == '': continue
        if int(i) in gt.keys():
            ans[int(i)] = 'tp'
            flag = 1

    if a in gt.keys():
        ans[a] = 'tp'
    else:
        if flag == 0:
            ans[a] = 'fp'

tn = 0
tp = 0
fn = 0
fp = 0
for i in ans:
    if i == 'tp': tp += 1
    if i == 'tn': tn += 1
    if i == 'fp': fp += 1
    if i == 'fn': fn += 1
print(tp, fp, tn, fn)
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
fscore = 2 * precision * recall / (precision + recall + eps)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-Score: ', fscore)
