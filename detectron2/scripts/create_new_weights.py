# import some common libraries
import numpy as np
import pickle as pkl
from copy import deepcopy

def avg_stem_weights(weights, outDim=1):
    return np.hstack((np.sum(weights, axis=1, keepdims=True),)*outDim)
def greyscale_stem_weigts(weights):
    return np.average(weights, axis=1, weights=[0.299, 0.587, 0.114])

    
file = open("/home/clara/Downloads/model_final_f10217.pkl",'rb')
model = pkl.load(file)

modelRGBD = deepcopy(model)
modelRGBN = deepcopy(model)
modelGD = deepcopy(model)
modelGN = deepcopy(model)

for k in model["model"].keys():
    v = model["model"][k]
    ks = k.split(".")
    if "stem.conv1.weight" in k:
        modelRGBD["model"].pop(k)
        modelRGBD["model"]["backbone.fpnRGB."+".".join(ks[1:])] = v
        modelRGBD["model"]["backbone.fpnDepth."+".".join(ks[1:])] = avg_stem_weights(v)
    elif "backbone" in k:
        modelRGBD["model"].pop(k)
        modelRGBD["model"]["backbone.fpnRGB."+".".join(ks[1:])] = v
        modelRGBD["model"]["backbone.fpnDepth."+".".join(ks[1:])] = v

for k in model["model"].keys():
    v = model["model"][k]
    ks = k.split(".")
    if "stem" in k:
        print(k, model["model"][k].shape)

with open("/home/clara/Downloads/rgbgrey.pkl", 'wb') as handle:
    pkl.dump(modelRGBD, handle, protocol=pkl.HIGHEST_PROTOCOL)

