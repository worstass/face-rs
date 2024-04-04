import sys
import tvm
import tvm.relay as relay
import torch
from tvm.contrib import graph_executor
from facenet_pytorch import MTCNN


def compile():
    input_shape = [1, 3, 224, 224]
    img = torch.randn(input_shape)
    # img = torch.ones(3,4,2,2)
    save_path = None
    return_prob = False
    model = torch.jit.trace(MTCNN(), img).eval()

    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(model,  shape_list)

    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
        temp = tvm.utils.tempdir()
        path_lib = temp.relpath("deploy_lib.tar")
        lib.export_library("compiled_lib.so")

    # dtype = "float32"
    # m = graph_executor.GraphModule(lib["default"](dev))
    # # Set inputs
    # m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # # Execute
    # m.run()
    # # Get outputs
    # tvm_output = m.get_output(0)


if __name__ == "__main__":
    compile()