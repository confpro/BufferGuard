import pandas as pd
from tree_sitter import Language, Parser
import networkx as nx
from collections import OrderedDict
import os
import tree_sitter_c as ts_c
from tree_sitter import Language, Parser
import networkx as nx


unsafe_functions = [
    'strcpy', 'strcat', 'strncpy', 'strncat',
    'sprintf', 'vsprintf', 'snprintf', 'vsnprintf',
    'scanf', 'vscanf', 'sscanf', 'vsscanf', 'fscanf', 'vfscanf',
    'memcpy', 'memmove', 'memset',
    'fgets', 'fread', 'read'
]

C_LANGUAGE = Language(ts_c.language())
parser = Parser()
parser.language = C_LANGUAGE


def extract_ast_structure(node, parent=None, graph=None, node_counter=[1], c_code=None):
    if graph is None:
        graph = nx.DiGraph()

    node_id = node_counter[0]
    node_counter[0] += 1
    node_text = c_code[node.start_byte:node.end_byte]

    graph.add_node(node_id, label=node_text, tree_sitter_node=node, node_id=node_id)

    if parent is not None:
        graph.add_edge(parent, node_id)

    for child in node.children:
        extract_ast_structure(child, parent=node_id, graph=graph, node_counter=node_counter, c_code=c_code)

    return graph

def check_for_unsafe_functions(node, c_code):
    if node.type == 'call_expression':
        function_name_node = next((child for child in node.children if child.type == 'identifier'), None)
        if function_name_node:
            function_name = c_code[function_name_node.start_byte:function_name_node.end_byte]
            if function_name in unsafe_functions:
                parameters = []
                for child in node.children:
                    if child.type == 'argument_list':
                        for param in child.children:
                            if param.type != 'comma':
                                parameters.append(param)
                return node, parameters
    for child in node.children:
        unsafe_nodes = check_for_unsafe_functions(child, c_code)
        if unsafe_nodes:
            return unsafe_nodes
    return None


def get_adjacent_nodes(node, graph):
    adjacent_nodes = []
    for neighbor in graph.neighbors(node):
        adjacent_nodes.append(graph.nodes[neighbor])
    for parent in graph.predecessors(node):
        adjacent_nodes.append(graph.nodes[parent])

    return adjacent_nodes
def process_func_before(func_before_code):
    tree = parser.parse(bytes(func_before_code, 'utf-8'))
    graph = extract_ast_structure(tree.root_node, c_code=func_before_code)

    focus_nodes = []
    for node_id, data in graph.nodes(data=True):
        if 'tree_sitter_node' in data:
            tree_sitter_node = data['tree_sitter_node']
            unsafe_node = check_for_unsafe_functions(tree_sitter_node, func_before_code)
            if unsafe_node:
                call_node, parameters = unsafe_node
                focus_nodes.append(node_id)
                for param in parameters:
                    param_text = func_before_code[param.start_byte:param.end_byte]
                    param_id = [key for key, value in graph.nodes(data=True) if value['tree_sitter_node'] == param][0]
                    focus_nodes.append(param_id)
                    adjacent_nodes = get_adjacent_nodes(param_id, graph)
                    for adj_node in adjacent_nodes:
                        focus_nodes.append(adj_node['node_id'])
                call_adjacent_nodes = get_adjacent_nodes(node_id, graph)
                for adj_node in call_adjacent_nodes:
                    focus_nodes.append(adj_node['node_id'])
    focus_nodes = list(OrderedDict.fromkeys(focus_nodes))

    focus_nodes_content = " ".join([graph.nodes[node_id]['label'] for node_id in focus_nodes])

    return focus_nodes_content

def process_csv(input_file, output_file):
    with pd.read_csv(input_file, chunksize=1000) as reader:
        for chunk in reader:

            chunk['slice_code'] = chunk['func_before'].apply(process_func_before)

            chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            print("1000")

input_file = '***.csv'
output_file = '***.csv'

process_csv(input_file, output_file)
