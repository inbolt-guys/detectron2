import json
import os
import matplotlib.pyplot as plt

path = "/home/clara/detectronDocker/outputs/RCNN_RGB_class_agnostic_and_normalized"

metrics_file_path = path + "/metrics.json"
show = True
with open(metrics_file_path) as metrics:
    lines = metrics.readlines()
    d = {}
    all_keys = []
    interesting_keys = ["eta_seconds", "fast_rcnn/cls_accuracy", "fast_rcnn/false_negative", 
            "fast_rcnn/fg_cls_accuracy", "loss_box_reg", "loss_cls", "loss_mask", "loss_rpn_cls", "loss_rpn_loc", "mask_rcnn/accuracy", 
            "mask_rcnn/false_negative", "mask_rcnn/false_positive", "total_loss"]
    for l in lines[190:]:
        data = json.loads(l)
        for k, v in data.items():
            if k not in d:
                d[k] = []
                all_keys.append(k)
            d[k].append(v)
    print("all keys")
    for k in all_keys: print(k) 
    for k in interesting_keys:
        plt.plot(d[k])
        plt.title(k)
        plt.savefig(path + "/" + k.replace("/", "_") + ".png")
        if show:
            plt.show()
        else:
            plt.clf()
    total_loss = d["total_loss"]
    """a = [sorted(total_loss[i:i+1000])[499] for i in range(len(total_loss)-1000)]
    k = "median_total_loss_over_1000_iterations"
    plt.plot(a)
    plt.title(k)
    plt.savefig(path + "/" + k.replace("/", "_") + ".png")"""
    if show:
        plt.show()
    else:
        plt.clf()
