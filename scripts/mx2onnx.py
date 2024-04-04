import sys
from mxnet.onnx import export_model

if __name__ == "__main__":
    export_model(sys.argv[1], sys.argv[2], input_shape=[(0,0,0,0)])
