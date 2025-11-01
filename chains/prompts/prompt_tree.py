from typing import Union
from pydantic import BaseModel
from anytree import Node, RenderTree
import json


class LeafNode(BaseModel):
    """Terminal node that sets task_id"""
    name: str
    task_id: int

    def __init__(self, name: str = None, task_id: int = None, **data):
        if name is not None and task_id is not None:
            super().__init__(name=name, task_id=task_id, **data)
        else:
            super().__init__(**data)


class FwdNode(BaseModel):
    """Forward reference to another node"""
    node_name: str

    def __init__(self, node_name: str = None, **data):
        if node_name is not None:
            super().__init__(node_name=node_name, **data)
        else:
            super().__init__(**data)


class DecisionNode(BaseModel):
    """Metadata for a decision node"""
    node_name: str
    variable_name: str
    description: str
    prompt: str
    true_branch: Union[LeafNode, FwdNode]
    false_branch: Union[LeafNode, FwdNode]


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
                Node(f"{name}_true_leaf", parent=node,
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
                        Node(f"{name}_true_ref", parent=node,
                             data=FwdNode(target_name), branch_type="true")

            # Process false branch
            if isinstance(spec.false_branch, LeafNode):
                # Create a terminal node for the leaf
                Node(f"{name}_false_leaf", parent=node,
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
                        Node(f"{name}_false_ref", parent=node,
                             data=FwdNode(target_name), branch_type="false")
                    
        self.compiled = True
    
    def export(self):
        """Export tree to JSON string"""
        if not self.compiled:
            self.compile()

        def node_to_dict(node):
            """Recursively convert anytree Node to dictionary"""
            result = {
                'name': node.name,
                'data': node.data.model_dump() if isinstance(node.data, BaseModel) else node.data,
                'branch_type': getattr(node, 'branch_type', None)
            }

            if node.children:
                result['children'] = [node_to_dict(child) for child in node.children]

            return result

        return json.dumps(node_to_dict(self.root), indent=2)
            
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

    async def execute(self, chain, initial_fields: dict = None) -> tuple[int, dict]:
        """
        Execute the decision tree starting from root.

        Args:
            chain: PromptChain instance for LLM execution
            initial_fields: Optional initial prev_fields

        Returns:
            tuple: (task_id, all_decisions) where all_decisions contains
                   all variable_name: bool pairs from the traversal
        """
        if not self.compiled:
            self.compile()

        # Initialize chain with any starting fields
        if initial_fields:
            chain = chain.set_prev_fields({**chain.prev_fields, **initial_fields})

        decisions = {}  # Track all decisions made during traversal
        current_node = self.root

        # Traverse until we hit a leaf
        while True:
            node_data = current_node.data

            # If we've reached a leaf node, return its task_id
            if isinstance(node_data, LeafNode):
                return node_data.task_id, decisions

            # If it's a forward reference, resolve it
            if isinstance(node_data, FwdNode):
                current_node = self._node_map[node_data.node_name]
                continue

            # Must be a DecisionNode - execute it
            if isinstance(node_data, DecisionNode):
                # Execute the prompt to get a boolean decision
                decision = await self._execute_decision(
                    chain,
                    node_data,
                    decisions
                )

                # Store the decision
                decisions[node_data.variable_name] = decision

                # Update chain with this decision
                chain = chain.set_prev_fields({
                    **chain.prev_fields,
                    node_data.variable_name: decision
                })

                # Follow the appropriate branch
                current_node = self._follow_branch(current_node, decision)
            else:
                raise ValueError(f"Unknown node type: {type(node_data)}")

    async def _execute_decision(
        self,
        chain,
        node_data: DecisionNode,
        prev_decisions: dict
    ) -> bool:
        """
        Execute a decision node's prompt and return boolean result.

        Uses a simple Yes/No prompt with structured output.
        """
        # Define response format for boolean decisions
        class Decision(BaseModel):
            reasoning: str
            decision: bool

        # Render the prompt with any previous decisions in context
        from chains.prompts.prompt_chain import replace_strs
        context = {**chain.prev_fields, **prev_decisions}
        rendered_prompt = replace_strs(node_data.prompt, context)

        # Execute the prompt
        result_chain = await (
            chain
            .prompt(rendered_prompt)
            .with_structure(Decision)
            .generate()
        )

        # Extract the decision
        decision_obj = result_chain.response_list[-1]
        return decision_obj.decision

    def _follow_branch(self, decision_node, decision: bool):
        """
        Follow the true or false branch from a decision node.

        Returns the next anytree Node to process.
        """
        # Find children by branch_type
        for child in decision_node.children:
            branch_type = getattr(child, 'branch_type', None)
            if decision and branch_type == 'true':
                return child
            elif not decision and branch_type == 'false':
                return child

        raise ValueError(f"No {'true' if decision else 'false'} branch found for node {decision_node.name}")


# ============================================================================
# Usage Example
# ============================================================================

async def example_usage():
    """Example of using PromptTree with async execution"""
    from chains.prompts.prompt_chain import PromptChain

    # Build a simple decision tree
    tree = PromptTree()

    tree.add_node(
        node_name="sentiment_check",
        variable_name="is_positive",
        description="Check if the input has positive sentiment",
        prompt="Is the following text positive in sentiment? {{input_text}}",
        true_branch=LeafNode(name="positive", task_id=1),
        false_branch=LeafNode(name="negative", task_id=2),
        is_root=True
    )

    tree.visualize()

    # Execute the tree directly
    chain = PromptChain()

    task_id, decisions = await tree.execute(
        chain,
        initial_fields={"input_text": "I love this product!"}
    )

    print(f"\nRouted to task_id: {task_id}")
    print(f"Decisions made: {decisions}")
    # Output: Routed to task_id: 1
    #         Decisions made: {'is_positive': True}


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
