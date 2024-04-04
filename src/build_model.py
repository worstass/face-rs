import argparse
import csv
import logging
from os import path as osp
import sys
import shutil

import numpy as np

import tvm
from tvm import te
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_runtime, cc
from PIL import Image
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
import tensorflow.compat.v1 as tf1
from tensorflow.python.platform import gfile

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Resnet build example")
aa = parser.add_argument
aa("--build-dir", type=str, required=True, help="directory to put the build artifacts")
aa("--batch-size", type=int, default=1, help="input image batch size")
aa(
    "--opt-level",
    type=int,
    default=3,
    help="level of optimization. 0 is unoptimized and 3 is the highest level",
)
aa("--target", type=str, default="llvm", help="target context for compilation")
aa("--image-shape", type=str, default="3,224,224", help="input image dimensions")
aa("--image-name", type=str, default="cat.png", help="name of input image to download")
args = parser.parse_args()

build_dir = args.build_dir
batch_size = args.batch_size
opt_level = args.opt_level
target = tvm.target.create(args.target)
image_shape = tuple(map(int, args.image_shape.split(",")))
data_shape = (batch_size,) + image_shape

tf1_ml_models = (
    # VGGFace2 training set, 0.9965 LFW accuracy
    ('20180402-114759', '1im5Qq006ZEV_tViKh3cgia_Q4jJ13bRK', (1.1817961, 5.291995557), 0.4),
    # CASIA-WebFace training set, 0.9905 LFW accuracy
    ('20180408-102900', '100w4JIUz44Tkwte9F-wEH0DOFsY-bPaw', (1.1362496, 5.803152427), 0.4),
    # CASIA-WebFace-Masked, 0.9873 LFW, 0.9667 LFW-Masked (orig model has 0.9350 on LFW-Masked)
    ('inception_resnetv1_casia_masked', '1FddVjS3JbtUOjgO0kWs43CAh0nJH2RrG', (1.1145709, 4.554903071), 0.6)
)

tf2_ml_models = (
    ('inception_v3_on_mafa_kaggle123', '1nhmv4Pd8nnV8XHv6vlf6RCpwQLow78zS'),
)

mxnet_ml_models = (
    ('mobilenet_v2_on_mafa_kaggle123', '1DYUIroNXkuYKQypYtCxQvAItLnrTTt5E'),
    ('resnet18_on_mafa_kaggle123', '1A3fNrvgrJqMw54cWRj47LNFNnFvTjmdj')
)
# detection models
# ml_models = (
#     ('retinaface_mnet025_v1', '1ggNFFqpe0abWz6V1A82rnxD6fyxB8W2c'),
#     ('retinaface_mnet025_v2', '1EYTMxgcNdlvoL1fSC8N1zkaWrX75ZoNL'),
#     ('retinaface_r50_v1', '1LZ5h9f_YC5EdbIZAqVba9TKHipi90JBj'),
# )
# recog models 
# ml_models = (
#     ('arcface_mobilefacenet', '17TpxpyHuUc1ZTm3RIbfvhnBcZqhyKszV', (1.26538905, 5.552089201), 200),
#     ('arcface_r100_v1', '11xFaEHIQLNze3-2RUV1cQfT-q6PKKfYp', (1.23132175, 6.602259425), 400),
#     ('arcface_resnet34', '1ECp5XrLgfEAnwyTYFEhJgIsOAw6KaHa7', (1.2462842, 5.981636853), 400),
#     ('arcface_resnet50', '1a9nib4I9OIVORwsqLB0gz0WuLC32E8gf', (1.2375747, 5.973354538), 400),
#     ('arcface-r50-msfdrop75', '1gNuvRNHCNgvFtz7SjhW82v2-znlAYaRO', (1.2350148, 7.071431642), 400),
#     ('arcface-r100-msfdrop75', '1lAnFcBXoMKqE-SkZKTmi6MsYAmzG0tFw', (1.224676, 6.322647217), 400),
#     # CASIA-WebFace-Masked, 0.9840 LFW, 0.9667 LFW-Masked (orig mobilefacenet has 0.9482 on LFW-Masked)
#     ('arcface_mobilefacenet_casia_masked', '1ltcJChTdP1yQWF9e1ESpTNYAVwxLSNLP', (1.22507105, 7.321198934), 200),
# )


def _embedding_calculator(model_file):
    with tf1.Graph().as_default() as graph:
        graph_def = tf1.GraphDef()
        with gfile.FastGFile(model_file, 'rb') as f:
            model = f.read()
        graph_def.ParseFromString(model)
        tf1.import_graph_def(graph_def, name='')

def build(target_dir):
    """ Compiles resnet18 with TVM"""
    # Download the pretrained model in MxNet's format.
    block = get_model("resnet18_v1", pretrained=True)

    shape_dict = {"data": (1, 3, 224, 224)}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    # Add softmax to do classification in last layer.
    func = mod["main"]
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

    # save the model artifacts
    deploy_lib = osp.join(target_dir, "deploy_lib.o")
    lib.save(deploy_lib)
    cc.create_shared(osp.join(target_dir, "deploy_lib.so"), [osp.join(target_dir, "deploy_lib.o")])

    with open(osp.join(target_dir, "deploy_graph.json"), "w") as fo:
        fo.write(graph)

    with open(osp.join(target_dir, "deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(params))


def download_img_labels():
    """ Download an image and imagenet1k class labels for test"""
    from mxnet.gluon.utils import download

    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "synset.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")

    with open(synset_path) as fin:
        synset = eval(fin.read())

    with open(synset_name, "w") as f:
        for key in synset:
            f.write(synset[key])
            f.write("\n")

    return synset


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def get_cat_image():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    shutil.copyfile(img_path, "cat.png")
    img = Image.open(img_path).resize((224, 224))
    return transform_image(img)


def test_build(build_dir):
    """ Sanity check with the cat image we download."""
    graph = open(osp.join(build_dir, "deploy_graph.json")).read()
    lib = tvm.runtime.load_module(osp.join(build_dir, "deploy_lib.so"))
    params = bytearray(open(osp.join(build_dir, "deploy_param.params"), "rb").read())
    input_data = get_cat_image()
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.load_params(params)
    module.run(data=input_data)
    out = module.get_output(0).asnumpy()
    top1 = np.argmax(out[0])
    synset = download_img_labels()
    print("TVM prediction top-1:", top1, synset[top1])


if __name__ == "__main__":
    logger.info("Compiling the model to graph runtime.")
    build(build_dir)
    logger.info("Testing the model's predication on test data.")
    test_build(build_dir)