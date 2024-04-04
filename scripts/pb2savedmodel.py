#!/usr/bin/env python3

import sys
# import tensorflow as tf
# import tensorflow.compat.v1 as tf1
import tensorflow._api.v2.compat.v1 as tf1
tf1.disable_v2_behavior()

def convert(model_file, out_path):
    with tf1.Graph().as_default() as graph:
        graph_def = tf1.GraphDef()
        with tf1.io.gfile.GFile(model_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        _ = tf1.import_graph_def(graph_def, name='')
        _ = tf1.global_variables_initializer()
        builder = tf1.saved_model.builder.SavedModelBuilder(out_path)
        with tf1.Session(graph=graph) as sess:
            builder.add_meta_graph_and_variables(sess,
                                                 [tf1.saved_model.tag_constants.SERVING],
                                                #  strip_default_attrs=True,
                                                 signature_def_map={tf1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf1.saved_model.signature_def_utils.predict_signature_def(inputs={'batch_size': graph.get_tensor_by_name("batch_size:0"), 'phase_train': graph.get_tensor_by_name("phase_train:0")},
                                                                    outputs={ 'embeddings': graph.get_tensor_by_name("embeddings:0") })},
                                                #  assets_collection=None
                                                )
        builder.save()

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])