import numpy as np
import tensorflow as tf

meta_path = './nr/data-train/model-0.0.ckpt.meta' # Your .meta file
model_path = './nr/data-train/'

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

    graph = tf.get_default_graph()

    vars_ = [v for key in graph.get_all_collection_keys() for v in graph.get_collection_ref(key)][1:]

    var_sizes =  [np.product(list(map(int, v.get_shape())))*v.dtype.size for v in vars_]
    print(sum(var_sizes)/(1024**2), 'MB')