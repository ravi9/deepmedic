import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.concat import Concat

class Concat5d(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('concat', dict(op='ConcatV2')),
            ],
            edges=[
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        concat = match['concat']
        if concat["name"] != "net/concat_2":
            return

        inp0 = concat.in_port(0).get_source().node
        inp1 = concat.in_port(1).get_source().node

        shape = np.array([-1, 50, 29*29*29], dtype=np.int32)
        shapeNode = Const(graph, {'value': shape}).create_node()
        reshape0 = Reshape(graph, dict(name=inp0.name + '/reshape')).create_node([inp0, shapeNode])
        reshape1 = Reshape(graph, dict(name=inp1.name + '/reshape')).create_node([inp1, shapeNode])
        new_concat = Concat(graph, dict(name=concat.name + "/new", axis=1)).create_node([reshape0, reshape1])

        shape = np.array([-1, 100, 29, 29, 29], dtype=np.int32)
        shapeNode = Const(graph, {'value': shape}).create_node()
        restore = Reshape(graph, dict(name=inp0.name + '/reshape')).create_node([new_concat, shapeNode])

        concat.out_port(0).get_connection().set_source(restore.out_port(0))
