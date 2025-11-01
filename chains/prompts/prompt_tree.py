from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from anytree import Node, RenderTree, PreOrderIter


class LeafNode:
    """Terminal node that sets task_id"""
    def __init__(self, name: str, task_id: int):
        self.name = name
        self.task_id = task_id


class FwdNode:
    """Forward reference to another node"""
    def __init__(self, node_name: str):
        self.node_name = node_name


class DecisionNode:
    """Metadata for a decision node"""
    def __init__(
        self,
        node_name: str,
        variable_name: str,
        description: str,
        prompt: str,
        true_branch: Union[LeafNode, FwdNode],
        false_branch: Union[LeafNode, FwdNode]
    ):
        self.node_name = node_name
        self.variable_name = variable_name
        self.description = description
        self.prompt = prompt
        self.true_branch = true_branch
        self.false_branch = false_branch


class PromptTree:
    """
    Tree-based API for building classification/routing pipelines.
    
    Uses anytree internally for tree representation.
    
    Usage:
        tree = PromptTree()
        tree.add_node(
            node_name="decision",
            variable_name="is_true",
            description="...",
            prompt="...",
            true_branch=LeafNode("yes", task_id=1),
            false_branch=FwdNode("next_decision")
        )
        tree.compile()  # Build the internal tree representation
        pipeline = tree.to_pipeline()
    """
    
    def __init__(self):
        self.node_specs = {}  # Store node specifications before compilation
        self.root = None  # Root of the anytree
        self.compiled = False
        self._node_map = {}  # Map from node_name to anytree Node
        
    def add_node(
        self,
        node_name: str,
        variable_name: str,
        description: str,
        prompt: str,
        true_branch: Union[LeafNode, FwdNode],
        false_branch: Union[LeafNode, FwdNode],
        is_root: bool = False
    ):
        """Add a decision node specification"""
        self.node_specs[node_name] = DecisionNode(
            node_name=node_name,
            variable_name=variable_name,
            description=description,
            prompt=prompt,
            true_branch=true_branch,
            false_branch=false_branch
        )
        if is_root or len(self.node_specs) == 1:
            self.root_name = node_name
        self.compiled = False  # Invalidate compilation
        
    def compile(self):
        """Build the anytree representation from node specifications"""
        if not self.node_specs:
            raise ValueError("No nodes have been added to the tree")
            
        # Create anytree nodes for all decision nodes
        self._node_map = {}
        for name, spec in self.node_specs.items():
            self._node_map[name] = Node(name, data=spec)
            
        # Set root
        if not hasattr(self, 'root_name'):
            # If no root specified, use the first node or find one that's not referenced
            referenced = set()
            for spec in self.node_specs.values():
                if isinstance(spec.true_branch, FwdNode):
                    referenced.add(spec.true_branch.node_name)
                if isinstance(spec.false_branch, FwdNode):
                    referenced.add(spec.false_branch.node_name)
            
            roots = set(self.node_specs.keys()) - referenced
            if roots:
                self.root_name = list(roots)[0]
            else:
                # Fall back to first node if there's a cycle
                self.root_name = list(self.node_specs.keys())[0]
                
        self.root = self._node_map[self.root_name]
        
        # Build tree structure by processing branches
        # First pass: identify which nodes should be parented
        parented_nodes = set()

        for name, spec in self.node_specs.items():
            if isinstance(spec.true_branch, FwdNode):
                parented_nodes.add(spec.true_branch.node_name)
            if isinstance(spec.false_branch, FwdNode):
                parented_nodes.add(spec.false_branch.node_name)

        # Second pass: build the tree structure
        for name, spec in self.node_specs.items():
            node = self._node_map[name]

            # Process true branch
            if isinstance(spec.true_branch, LeafNode):
                # Create a terminal node for the leaf
                leaf = Node(f"{name}_true_leaf", parent=node,
                           data=spec.true_branch, branch_type="true")
            elif isinstance(spec.true_branch, FwdNode):
                # Link to another decision node by parenting it
                target_name = spec.true_branch.node_name
                if target_name in self._node_map:
                    target_node = self._node_map[target_name]
                    # Only parent if it doesn't already have a parent
                    if target_node.parent is None:
                        target_node.parent = node
                        target_node.branch_type = "true"
                    else:
                        # Create a reference node if already parented elsewhere
                        ref = Node(f"{name}_true_ref", parent=node,
                                 data=FwdNode(target_name), branch_type="true")

            # Process false branch
            if isinstance(spec.false_branch, LeafNode):
                # Create a terminal node for the leaf
                leaf = Node(f"{name}_false_leaf", parent=node,
                           data=spec.false_branch, branch_type="false")
            elif isinstance(spec.false_branch, FwdNode):
                # Link to another decision node by parenting it
                target_name = spec.false_branch.node_name
                if target_name in self._node_map:
                    target_node = self._node_map[target_name]
                    # Only parent if it doesn't already have a parent
                    if target_node.parent is None:
                        target_node.parent = node
                        target_node.branch_type = "false"
                    else:
                        # Create a reference node if already parented elsewhere
                        ref = Node(f"{name}_false_ref", parent=node,
                                 data=FwdNode(target_name), branch_type="false")
                    
        self.compiled = True
    

            
    def visualize(self):
        """visualization using anytree's RenderTree"""
        if not self.compiled:
            self.compile()

        print("\nAnytree Representation:")
        print("=" * 50)
        for pre, _, node in RenderTree(self.root):
            if isinstance(node.data, DecisionNode):
                branch = getattr(node, 'branch_type', None)
                if branch:
                    print(f"{pre}[{branch}] -> {node.name} ({node.data.variable_name})")
                else:
                    print(f"{pre}{node.name} ({node.data.variable_name})")
            elif isinstance(node.data, LeafNode):
                branch = getattr(node, 'branch_type', '?')
                print(f"{pre}[{branch}] -> task_id={node.data.task_id}")
            elif isinstance(node.data, FwdNode):
                branch = getattr(node, 'branch_type', '?')
                print(f"{pre}[{branch}] -> {node.data.node_name}")
