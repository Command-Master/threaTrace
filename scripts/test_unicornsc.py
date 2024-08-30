import os
from subprocess import Popen, PIPE

from commons import *

alert_thre = 300
id_map_t = {}
anomaly_node = {}
this_ts = 0


def update_benign(k):
    k = id_map_t[k]
    if k in anomaly_node.keys():
        anomaly_node.pop(k)


def raise_alert(k):
    k = id_map_t[k]
    if not k in anomaly_node.keys():
        anomaly_node[k] = this_ts


def real_raise_alert():
    for k in anomaly_node.keys():
        if this_ts - anomaly_node[k] > alert_thre:
            return False
    return True


def main():
    global alert_thre
    global this_ts

    thre = 2.0
    first_alert = True
    batch_size = 0
    model_list = []
    feature_num, label_num = getFeature()
    f = open('models_list.txt', 'r')
    for line in f:
        model_list.append(line.strip('\n'))
    f.close()
    if len(sys.argv) > 1: ss = sys.argv[1]
    fw = open('pid.txt', 'w')
    fw.write(str(os.getpid()) + '\n')

    if len(sys.argv) > 2: batch_size = int(sys.argv[2])
    if len(sys.argv) > 3: graph_id = sys.argv[3]
    if len(sys.argv) > 4: thre = float(sys.argv[4])
    if len(sys.argv) > 5: alert_thre = float(sys.argv[5])
    p = Popen(
        '../graphchi-cpp-master/bin/example_apps/test file ../graphchi-cpp-master/graph_data/gdata filetype edgelist stream_file ../graphchi-cpp-master/graph_data/unicornsc/' + graph_id + '.txt batch ' + ss,
        shell=True, stdin=PIPE, stdout=PIPE)
    fw.write(str(p.pid + 1) + '\n')
    fw.close()
    while 1:
        data, this_ts = extract_data(p, id_map_t)
        if data is None:
            break
        if first_alert:
            detect(model_list, data, feature_num, label_num, thre, batch_size, update_benign, raise_alert)
            first_alert = real_raise_alert()

    show(str(graph_id) + ' finished. fp: ' + str(len(anomaly_node.keys())))


if __name__ == "__main__":
    graphchi_root = os.path.abspath(os.path.join(os.getcwd(), '../graphchi-cpp-master'))
    os.environ['GRAPHCHI_ROOT'] = graphchi_root
    main()
