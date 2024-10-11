# First import the library
import pyrealsense2 as rs

# Import Numpy for easy array manipulation
import numpy as np

# Import OpenCV for easy image rendering
import cv2
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image


def include_depth_in_dataset(
    imagesPath="/home/clara/datasets/rgb/train",
    depthsPath="/home/clara/datasets/depth/train",
    outPath="/home/clara/datasets",
    datasetFunction="train",
    extension="png",
    saveDepth=True,
    showAll=True,
    saveRGBD=True,
    colorFroms=[cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2YUV, cv2.COLOR_RGB2BGR],
    colorTos=[cv2.COLOR_HLS2RGB, cv2.COLOR_YUV2RGB, cv2.COLOR_BGR2RGB],
    colorFormatNames=["HLS", "YUV", "BGR"],
    canals=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
):
    assert len(colorFroms) == len(colorTos) == len(canals) == len(colorFormatNames)

    for c in range(len(colorFormatNames)):
        cfn = colorFormatNames[c]
        for c2 in canals[c]:
            if not os.path.isdir(outPath + "/" + cfn + str(c2)):
                os.mkdir(outPath + "/" + cfn + str(c2))
            if not os.path.isdir(outPath + "/" + cfn + str(c2) + "/" + datasetFunction):
                os.mkdir(outPath + "/" + cfn + str(c2) + "/" + datasetFunction)

    if saveDepth:
        if not os.path.isdir(outPath + "/" + "depth_normalized"):
            os.mkdir(outPath + "/" + "depth_normalized")
        if not os.path.isdir(outPath + "/" + "depth_normalized" + "/" + datasetFunction):
            os.mkdir(outPath + "/" + "depth_normalized" + "/" + datasetFunction)

    if saveRGBD:
        if not os.path.isdir(outPath + "/" + "RGBD"):
            os.mkdir(outPath + "/" + "RGBD")
        if not os.path.isdir(outPath + "/" + "RGBD" + "/" + datasetFunction):
            os.mkdir(outPath + "/" + "RGBD" + "/" + datasetFunction)

    if showAll:
        if not os.path.isdir(outPath + "/" + "all"):
            os.mkdir(outPath + "/" + "all")
        if not os.path.isdir(outPath + "/" + "all" + "/" + datasetFunction):
            os.mkdir(outPath + "/" + "all" + "/" + datasetFunction)

    imagesPaths = sorted(glob.glob(imagesPath + "/*." + extension))
    depthsPaths = sorted(glob.glob(depthsPath + "/*." + extension))

    depthTimestamps = [0] * len(depthsPaths)
    for i in range(len(depthsPaths)):
        dp = depthsPaths[i]
        depthName = os.path.basename(dp)
        ts = os.path.splitext(depthName)[0]
        depthTimestamps[i] = float(ts)
    depthTimestamps = np.array(depthTimestamps)

    closestDepthIndex = [0] * len(imagesPaths)
    for i in range(len(imagesPaths)):
        ip = imagesPaths[i]
        imageName = os.path.basename(ip)
        ts = os.path.splitext(imageName)[0]
        closestDepthIndex[i] = np.argmin(np.abs(depthTimestamps - float(ts)))

    for i in range(len(imagesPaths)):
        ip = imagesPaths[i]
        dp = depthsPaths[closestDepthIndex[i]]
        name = os.path.basename(ip)
        print(name)
        image = (plt.imread(ip) * 255).astype(np.uint8)
        image = np.array(Image.open(ip))
        depth = plt.imread(dp)
        depth = np.array(Image.open(dp))
        normalized_depth = normalize_depth(depth)

        colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(normalized_depth), cv2.COLORMAP_JET)

        if showAll:
            totalImage = np.hstack(
                (
                    image,
                    np.dstack((normalized_depth, normalized_depth, normalized_depth)),
                    colorized_depth,
                )
            )
        if saveDepth:
            Image.fromarray(normalized_depth).save(
                outPath + "/" + "depth_normalized" + "/" + datasetFunction + "/" + name
            )
            # plt.imsave(outPath+"/"+"depth_normalized"+"/"+datasetFunction+"/"+name, normalized_depth, cmap='gray')
        if saveRGBD:
            RGBD = np.dstack((image, np.expand_dims(normalized_depth, axis=2)))
            Image.fromarray(RGBD).save(
                outPath + "/" + "RGBD" + "/" + datasetFunction + "/" + name + ".png"
            )
        for c in range(len(colorFroms)):
            if showAll:
                lineImage = np.empty(shape=(image.shape[0], 0, 3))
            colorFrom = colorFroms[c]
            colorTo = colorTos[c]
            for c2 in canals[c]:
                depthImage = add_depth_to_image(image, normalized_depth, colorFrom, colorTo, c2)
                # plt.imsave(outPath+"/"+colorFormatNames[c]+str(c2)+"/"+datasetFunction+"/"+name, depthImage)
                Image.fromarray(depthImage).save(
                    outPath
                    + "/"
                    + colorFormatNames[c]
                    + str(c2)
                    + "/"
                    + datasetFunction
                    + "/"
                    + name
                )
                if showAll:
                    lineImage = np.hstack((lineImage, depthImage))
            if showAll:
                totalImage = np.vstack((totalImage, lineImage))
        if showAll:
            Image.fromarray(totalImage.astype(np.uint8)).save(
                outPath + "/" + "all" + "/" + datasetFunction + "/" + name
            )
            # plt.imsave(outPath+"/"+"all"+"/"+datasetFunction+"/"+name, totalImage.astype(np.uint8))


def add_depth_to_image(image, normalized_depth, colorFrom, colorTo, canal):
    """
    image.shape = (l, h, 3)
    depth.shape = (l, h)
    out.shape = (l, h, 3)
    """
    newImage = cv2.cvtColor(np.copy(image), colorFrom)
    newImage[:, :, canal] = normalized_depth
    newImage = cv2.cvtColor(newImage, colorTo)
    return newImage.astype(np.uint8)


def normalize_depth(
    depth,
    reverse=True,
    percentileForMin=5,
    percentileForMax=85,
    scaleFactorForMin=1 / 1.5,
    valueWhenLessThanMin=255,
    valueWhenMoreThanMax=255,
    valueWhenNoValue=255,
):
    noValueMask = depth == 0

    max_depth = np.percentile(depth, percentileForMax)
    min_depth = (
        np.percentile(depth[np.logical_not(noValueMask)], percentileForMin) * scaleFactorForMin
    )

    depth_normalized = np.copy(depth)

    lessThanMinMask = np.logical_and(depth_normalized < min_depth, np.logical_not(noValueMask))
    moreThanMaxMask = depth_normalized > max_depth

    depth_normalized = (depth_normalized - min_depth) / (max_depth - min_depth) * 255

    depth_normalized[noValueMask] = valueWhenNoValue
    depth_normalized[lessThanMinMask] = valueWhenLessThanMin
    depth_normalized[moreThanMaxMask] = valueWhenMoreThanMax

    if reverse:
        depth_normalized = 255 - depth_normalized

    return depth_normalized.astype(np.uint8)


"""def include_depth_in_dataset(imagesPath="/home/clara/datasets/rgb/train", depthsPath = "/home/clara/datasets/depth/train", outPath = "/home/clara/datasets/",
                             imageFormat = (640, 480),
                             colorFroms = [cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2YUV, cv2.COLOR_RGB2BGR], 
                             colorTos = [cv2.COLOR_HLS2RGB, cv2.COLOR_YUV2RGB, cv2.COLOR_BGR2RGB],
                             colorFormatNames = ["HLS", "YUV", "BGR"],
                             canals = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]):
    assert len(colorFroms) == len(colorTos) == len(canals) == len(colorFormatNames)

    for cfn in colorFormatNames:
        os.mkdir(outPath+cfn)

    imagesPaths = glob.glob(imagesPath+"/*.png")
    depthsPaths = glob.glob(depthsPath+"/*.png")

    outImages = np.empty(shape=(len(colorFrom))+imageFormat+(3))
    for i in range(len(imagesPaths)):
        image = images[:, :, :, i]
        depth = depths[:, :, i]
        normalized_depth = normalize_depth(depth)
        modification = 0
        for c in range(len(colorFroms)):
            colorFrom = colorFroms[c]
            colorTo = colorTos[c]
            for c2 in canals[c]:
                depthImage = add_depth_to_image(image, normalized_depth, colorFrom, colorTo, c2)
                outImages[modification, i, :, :, :] = depthImage
                modification+=1"""

"""recordings = glob.glob("/home/clara/depth_aligned/recordings/*")

for rec in recordings:

    include_depth_in_dataset(imagesPath=rec+"/color",
                         depthsPath=rec+"/depth_aligned_mm",
                         outPath="/home/clara/depth_aligned/test/all",
                         extension="tif")"""

# include_depth_in_dataset()
rec = "/home/clara/depth_aligned/recordings/ec1b58ee-0aa7-4117-9669-fe94847c8411"
include_depth_in_dataset(
    imagesPath=rec + "/color",
    depthsPath=rec + "/depth_aligned_mm",
    outPath="/home/clara/depth_aligned/test/ec1b58ee-0aa7-4117-9669-fe94847c8411",
    extension="tif",
)
