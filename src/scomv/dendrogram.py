from scipy.cluster.hierarchy import ClusterNode
from typing import List
from scipy.cluster.hierarchy import to_tree

def dendrogram2newick(
    node: ClusterNode, parent_dist: float, leaf_names: List[str], newick: str = ""
) -> str:
    """Convert scipy dendrogram tree to newick format tree

    Args:
        node (ClusterNode): Tree node
        parent_dist (float): Parent distance
        leaf_names (List[str]): Leaf names
        newick (str): newick format string (Used in recursion)

    Returns:
        str: Newick format tree
    """
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{(parent_dist - node.dist):.2f}{newick}"
    else:
        if len(newick) > 0:
            newick = f"):{(parent_dist - node.dist):.2f}{newick}"
        else:
            newick = ");"
        newick = dendrogram2newick(node.left, node.dist, leaf_names, newick)
        newick = dendrogram2newick(node.right, node.dist, leaf_names, f",{newick}")
        newick = f"({newick}"
        return newick