import sun.reflect.generics.tree.Tree;

import java.util.ArrayList;

/**
 * Created by kavin on 4/10/15.
 */
public class TreeNode {

    public boolean is_leaf_node;
    public ID3DecisionTree.Attribute attribute;
    public String node_label;
    public int node_type;
    public int class1_instances;
    public int class2_instances;
    public double threshold;
    public ArrayList<TreeNode> node_children;
    public int node_level;
}
