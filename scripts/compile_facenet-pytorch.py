import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import PNet, ONet, RNet
import os

def export2onnx(out_dir):
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    torch.onnx.export(pnet, torch.randn(1,3,96,96), os.path.join(out_dir, "pnet.onnx"),
                      export_params=True)
    torch.onnx.export(rnet, torch.randn(1,3,48,16), os.path.join(out_dir, "rnet.onnx"),
                      export_params=True)
    torch.onnx.export(onet, torch.randn(1,3,96,32), os.path.join(out_dir, "onet.onnx"),
                      export_params=True)
    image = torch.randn(1,3,256,256)
    net = InceptionResnetV1(pretrained="vggface2")
    torch.onnx.export(net, image, os.path.join(out_dir, "20180402-114759.onnx"), export_params=True)
    net = InceptionResnetV1(pretrained="casia-webface")
    torch.onnx.export(net, image, os.path.join(out_dir, "20180408-102900.onnx"), export_params=True)


def compile(out_dir):
    # image = Image.open(os.path.join(os.path.dirname(__file__), "multiface.jpg"))
    # image = pil_to_tensor(image)
    # image = image.unsqueeze(0)
    # # image = image.permute(0,2,3,1)
    # image = image.to(torch.float)


    # mod, params = relay.frontend.from_pytorch(model, [("imgs", image.shape)])


    # target = tvm.target.Target("llvm", host="llvm")
    # dev = tvm.cpu(0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target=target, params=params)
    #     temp = tvm.utils.tempdir()
    #     path_lib = temp.relpath("deploy_lib.tar")
    #     lib.export_library("compiled_lib.so")

    # # load it back as a runtime
    # lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")
    # # Call the library factory function for default and create
    # # a new runtime.Module, wrap with graph module.
    # gmod = graph_executor.GraphModule(lib["default"](dev))
    # # use the graph module.
    # gmod.set_input("x", data)
    # gmod.run()

    # dtype = "float32"
    # m = graph_executor.GraphModule(lib["default"](dev))
    # # Set inputs
    # m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # # Execute
    # m.run()
    # # Get outputs
    # tvm_output = m.get_output(0)


if __name__ == "__main__":
    export2onnx(os.path.join(os.path.dirname(__file__), "../data/models/temp"))