from __future__ import absolute_import, print_function, division

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TensorboardLogger(object):

    def __init__(self, log_path, tf_graph):
        self.logger = tf.summary.FileWriter(log_path, tf_graph)

    def add_summary(self, value, name, step_num):
        self.logger.add_summary(tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)]),
                                global_step=step_num)
