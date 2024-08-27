# import some common libraries
import numpy as np
import torch
import pickle as pkl

path = "/home/clara/detectronDocker/outputs/early_fusion_DRCNN-FPN-attention-relu_full_training_from_scratch/model_final.pth"
file = open(path,'rb')
#file = open("/home/clara/detectronDocker/outputs/model_RGBD5000/model_final.pth", "rb")
#file = open("/home/clara/Downloads/model_final_f10217.pkl",'rb')
#model = pkl.load(file)
backbone = 0
other = 0
total = 0
resnet1 = 0
resnet2 = 0
fusion = 0
sizes = {}
if path.endswith(".pth"):
    model = torch.load(file, map_location=torch.device("cpu"))

    for k in model["model"].keys():
        v = model["model"][k]
        if k.startswith("backbone"):
            backbone+=torch.numel(v)
            total += torch.numel(v)
            if "resnet1" in k:
                resnet1 += torch.numel(v)
                print(k, torch.numel(v), torch.numel(model["model"][k.replace("resnet1", "resnet2")]))
            elif "resnet2" in k:
                resnet2 += torch.numel(v)
            elif "fusion" in k:
                fusion += torch.numel(v)
        else:
            other+=torch.numel(v)
            total += torch.numel(v)
        sizes[k] = torch.numel(v)
        #print(k, torch.numel(v))
elif path.endswith(".pkl"):
    model = pkl.load(file)
    for k in model["model"].keys():
        v = model["model"][k]
        if "fusion_layer" in k and "bias" in k:
            print(k, torch.sum(torch.abs(v)))
        if k.startswith("backbone"):
            backbone+=v.size
            total += v.size
            if "resnet1" in k:
                resnet1 += v.size
            elif "resnet2" in k:
                resnet2 += v.size
            elif "fusion" in k:
                fusion += v.size
        else:
            other+=v.size
            total += v.size
        print(k, v.size)


print(f"backbone: {backbone} ({backbone/total*100}%)")
print(f"resnet1: {resnet1} ({resnet1/total*100}%)")
print(f"resnet2: {resnet2} ({resnet2/total*100}%)")
print(f"fusion: {fusion} ({fusion/total*100}%)")
print(f"other: {other} ({other/total*100}%)")
print(f"total: {total} ({100}%)")


"""for k, v in sizes.items():
    print(f"{k}: {v} ({v/total*100}%)")"""