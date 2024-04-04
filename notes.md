## 20180402-114759
```
python3 src/pb2savedmodel.py ./data/models/temp/20180402-114759/20180402-114759/20180402-114759.pb ./data/models/temp/20180402-114759-savedmodel

# NotImplementedError: The following operators are not implemented: {'RandomUniform', 'FIFOQueueV2', 'QueueDequeueUpToV2'}
# savedmodel v1
tvmc compile --target llvm --model-format pb --output ./data/models/converted/20180402-114759.tar ./data/models/temp/20180402-114759/20180402-114759/20180402-114759.pb

# input: 
#     name: batch_size | tensor: int32[?]
#     name: phase_train | tensor: boolean[?]
# not working
python3 -m tf2onnx.convert --graphdef ./data/models/temp/20180402-114759/20180402-114759/20180402-114759.pb --inputs batch_size:0,phase_train:0 --outputs embeddings:0 --output ./data/models/temp/20180402-114759.onnx

# not working
python3 -m tf2onnx.convert --saved-model ./data/models/temp/20180402-114759-savedmodel --inputs batch_size:0,phase_train:0 --outputs embeddings:0 --output ./data/models/temp/20180402-114759.onnx

```
## 20180402-114759-vggface2.pth
```
# PyTorch v0.1.10
tvmc compile --target llvm --model-format pytorch --output ./data/models/converted/20180402-114759-pt.tar --input-shapes "data:[1,3,96,96]"  ./data/models/downloaded/20180402-114759-vggface2.pth
```

## 20180408-102900
```
# savedmodel v1
# NotImplementedError: The following operators are not implemented: {'RandomUniform', 'FIFOQueueV2', 'QueueDequeueUpToV2'}
tvmc compile --target llvm --model-format pb --output ./data/models/20180408-102900.tar ./data/models/20180408-102900/20180408-102900/20180408-102900.pb 

python3 -m tf2onnx.convert --input ./data/models/20180408-102900/20180408-102900/20180408-102900.pb --inputs input:0,phase_train:0 --outputs embeddings:0 --opset 10 --output ./data/models/temp/20180408-102900.onnx


tvmc compile --target llvm --model-format onnx --output ./data/models/converted/2d106det.tar --input-shapes "data:[1,3,96,96]" ./data/models/downloaded/2d106det.onnx
```

## 20180408-102900 (from facenet-pytorch)
```
tvmc compile --target llvm --model-format onnx --output ./data/models/converted/pnet.zip --input-shapes "input.1:[1,3,96,96]"  ./data/models/temp/pnet.onnx
tvmc compile --target llvm --model-format onnx --output ./data/models/converted/rnet.zip --input-shapes "input.1:[1,3,48,16]"  ./data/models/temp/rnet.onnx
tvmc compile --target llvm --model-format onnx --output ./data/models/converted/onet.zip --input-shapes "input.1:[1,3,96,32]"  ./data/models/temp/onet.onnx

tvmc compile --target llvm --model-format onnx --output ./data/models/converted/20180408-102900.tar --input-shapes "input.1:[1,3,630,1200]"  ./data/models/temp/20180408-102900.onnx
```

## 2021.10.16_lr0.005_decay0.5 (inception_v3_on_mafa_kaggle123)
```
# SavedModel V2
#
# TensorFlow Saved Model v1 - v175 - serve
# inputs: 
#     name: serving_default_input_1 | tensor: float32[-1,100,100,3]
#     name: saver_filename | tensor: string
# 
python3 -m tf2onnx.convert --saved-model ./data/models/temp/2021.10.16_lr0.005_decay0.5 --output ./data/models/temp/2021.10.16_lr0.005_decay0.5.onnx
tvmc compile --target llvm --model-format onnx --output data/models/converted/2021.10.16_lr0.005_decay0.5.tar --input-shapes "data:[3,100,100]" ./data/models/temp/2021.10.16_lr0.005_decay0.5.onnx
```

## inception_resnetv1_casia_masked
```
# NotImplementedError: The following operators are not implemented: {'RandomUniform', 'QueueDequeueUpToV2', 'FIFOQueueV2'}
tvmc compile --target llvm --model-format pb --output ./data/models/converted/inception_resnetv1_casia_masked.tar ./data/models/temp/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked.pb

python3 -m tf2onnx.convert --input ./data/models/temp/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked.pb --inputs input:0,phase_train:0 --outputs embeddings:0 --opset 13 --output ./data/models/temp/inception_resnetv1_casia_masked.onnx

python3 -m tf2onnx.convert --input ./data/models/20180408-102900/20180408-102900/20180408-102900.pb  --inputs input:0,phase_train:0 --outputs embeddings:0 --opset 8 --output target.onnx

python3 -m tf2onnx.convert --input ./data/models/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked.pb --inputs input:0,phase_train:0 --outputs embeddings:0 --opset 8 --output target.onnx


tvmc compile --target llvm --model-format pb --output data/models/inception_resnetv1_casia_masked.tar --input-shapes "data:[1,3,96,96]" ./data/models/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked/inception_resnetv1_casia_masked.pb

```

## retinaface-R50
```
python3 src/mxnet_compile.py ./data/models/temp/retinaface-R50/R50-symbol.json  ./data/models/temp/retinaface-R50/R50-0000.params ./data/models/converted/retinaface-R50.tar


python3 src/mxnet_compile.py
```

## arcface-mnet 
```
# MXNet v1.0.0
tvmc compile --target llvm --model-format pytorch --output ./data/models/converted/arcface-mnet.tar --input-shapes "data:[1,3,96,96]" ./data/models/downloaded/arcface-mnet.pth
  
python3 src/mxnet_compile.py
```


## buffalo_l
```

tvmc compile --target llvm --model-format onnx --output data/models/converted/w600k_r50.tar --input-shapes "input.1:[1,3,112,112]" data/models/temp/buffalo_l/w600k_r50.onnx

tvmc compile --target llvm --model-format onnx --output data/models/converted/det_10g.tar --input-shapes "input.1:[1,3,224,224]" data/models/temp/buffalo_l/det_10g.onnx

tvmc compile --target llvm --model-format onnx --output data/models/converted/1k3d68.tar --input-shapes "data:[1,3,192,192]" data/models/temp/buffalo_l/1k3d68.onnx

tvmc compile --target llvm --model-format onnx --output data/models/converted/2d106det.tar --input-shapes "data:[1,3,192,192]" data/models/temp/buffalo_l/2d106det.onnx


tvmc tune --target llvm  --output data/models/converted/genderage-autotuner_records.json  data/models/temp/buffalo_l/genderage.onnx
tvmc compile --target llvm --model-format onnx --output data/models/converted/genderage.tar --input-shapes "data:[1,3,96,96]" data/models/temp/buffalo_l/genderage.onnx

```

## mtcnn
```
tvmc compile --target llvm --model-format onnx --output data/models/converted/mtcnn.tar --input-shapes "data:[1,3,160,160]" data/models/temp/mtcnn.onnx
tvmc compile --target llvm --model-format onnx --output data/models/converted/mymodel.tar --input-shapes "input.1:[1,1,32,32]" data/models/temp/mymodel.onnx
```
