import os
from subprocess import Popen, PIPE

from commons import *

alert_thre = 300
id_map_t = {}
anomaly_node = {}
final_anomaly = []
this_ts = 0

def update_benign(k):
    k = id_map_t[k]
    if k in anomaly_node.keys():
        anomaly_node.pop(k)


def raise_alert(k):
    now_ts = this_ts
    k = id_map_t[k]
    if k in anomaly_node.keys():
        if now_ts - anomaly_node[k] > alert_thre:
            anomaly_node.pop(k)
            final_anomaly.append(k)
    else:
        anomaly_node[k] = now_ts


def main():
    global this_ts

    model_list = []
    feature_num, label_num = getFeature()
    f = open('models_list.txt', 'r')
    for line in f:
        temp = line.strip(' \n').split(' ')
        if temp[0] == '': continue
        temp2 = [int(i) for i in temp]
        model_list.append(temp2)
    f.close()

    ss = sys.argv[1]
    batch_size = int(sys.argv[2])
    graph_id = sys.argv[3]
    scene = int(int(graph_id) / 100) + 1
    thre = float(sys.argv[4])

    minfp = 10000
    for this_model in model_list:
        p = Popen(
            '../graphchi-cpp-master/bin/example_apps/test file ../graphchi-cpp-master/graph_data/gdata filetype edgelist stream_file ../graphchi-cpp-master/graph_data/streamspot/' + str(
                scene) + '/' + graph_id + '.txt batch ' + ss, shell=True, stdin=PIPE, stdout=PIPE)
        while 1:
            data, this_ts = extract_data(p, id_map_t)
            if data is None:
                break
            anomaly_node.clear()
            final_anomaly.clear()
            detect(this_model, data, feature_num, label_num, thre, batch_size, update_benign, raise_alert)
        if len(anomaly_node.keys()) + len(final_anomaly) < minfp: minfp = len(anomaly_node.keys())
    show(str(graph_id) + ' finished. fp: ' + str(minfp))


if __name__ == "__main__":
    graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
    os.environ['GRAPHCHI_ROOT'] = graphchi_root
    main()
