# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:39:49 2017

@author: SzMike
"""

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from cntk import load_model
from cntk.logging.graph import get_node_outputs


#####################################################
#####################################################
# helpers to print all node names
def dfs_walk(node, visited):
    if node in visited:
        return
    visited.add(node)
    print("visiting %s"%node.name)
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visited)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visited)

#####################################################
#####################################################

model_file=r'c:\Users\picturio\Documents\Projects\WISH\models\IMAGENET\AlexNetBS.model'

dfs_walk(z, set())
    # use this to print all node names of the model (and knowledge of the model to pick the correct one)

node_outputs = get_node_outputs(load_model(model_file))
for out in node_outputs: print("{0} {1}".format(out.name, out.shape))