import sun.reflect.generics.tree.Tree;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by kavin on 4/10/15.
 */
public class ID3DecisionTree {

    private static String class_labels[];
    private static ArrayList<Attribute> attribute_list = null;
    private static ArrayList<HashMap<String, String>> instances = null;
    private static int num_attributes = 0;

    private static double getlog2(double value){
        if(value == 0.0) {
            return 0.0;
        }
        return Math.log(value)/Math.log(2.0);
    }

    private static TreeNode constructTree(ArrayList<HashMap<String, String>> data_instance_list, int node_level, int m){
        int class1_count = 0;
        int class2_count = 0;

        for(int i = 0; i < data_instance_list.size(); i++){
            if(data_instance_list.get(i).get("class_label").equalsIgnoreCase(class_labels[0])){
                class1_count++;
            }else if(data_instance_list.get(i).get("class_label").equalsIgnoreCase(class_labels[1])){
                class2_count++;
            }
        }
        double prob_1 = (double)class1_count/(double)data_instance_list.size();
        double prob_2 = (double)class2_count/(double)data_instance_list.size();
        double total_entropy = - ((prob_1*getlog2(prob_1)) + (prob_2*getlog2(prob_2)));

        if(total_entropy == 0.0){
            TreeNode treeNode = new TreeNode();
            treeNode.is_leaf_node = true;
            treeNode.attribute = null;
            treeNode.node_children = null;
            treeNode.threshold = 0.0;
            treeNode.node_type = -1;
            treeNode.node_level = node_level;
            treeNode.class1_instances = class1_count;
            treeNode.class2_instances = class2_count;
            if(class1_count >= class2_count){
                treeNode.node_label = class_labels[0];
            }else{
                treeNode.node_label = class_labels[1];
            }
            return treeNode;
        }

        if(data_instance_list.size() < m || data_instance_list.size() == 0){
            TreeNode treeNode = new TreeNode();
            treeNode.is_leaf_node = true;
            treeNode.attribute = null;
            treeNode.node_children = null;
            treeNode.threshold = 0.0;
            treeNode.node_type = -1;
            treeNode.node_level = node_level;
            treeNode.class1_instances = class1_count;
            treeNode.class2_instances = class2_count;
            if(class1_count >= class2_count){
                treeNode.node_label = class_labels[0];
            }else{
                treeNode.node_label = class_labels[1];
            }
            return treeNode;
        }

        double[] attribute_entropy = new double[num_attributes];
        for(int i = 0; i < num_attributes; i++){
            if(attribute_list.get(i).attribute_type == 0){
                //Nominal
                attribute_entropy[i] = get_nominal_attribute_entropy(data_instance_list, i);

            }else if(attribute_list.get(i).attribute_type == 1){
                //Continuous
                attribute_entropy[i] = get_continuous_attribute_entropy(data_instance_list, i).entropy;
            }
            /*System.out.println("Attr: " + attribute_list.get(i).attribute_name +
              "  entropy: " + attribute_entropy[i]);*/
        }

        int min_entropy_index = 0;
        double min_entropy = attribute_entropy[0];

        for(int j = 1 ;j<attribute_entropy.length;j++){
            if(attribute_entropy[j] < min_entropy){
                min_entropy = attribute_entropy[j];
                min_entropy_index = j;
            }
        }

        Attribute attr = attribute_list.get(min_entropy_index);
        //System.out.println("Splitting at node level : " + node_level + " on attribute " + attr.attribute_name);
        if(attr.attribute_type == 0){
            //Nominal
            TreeNode split_node = new TreeNode();
            split_node.threshold = -1;
            split_node.node_label = "";
            split_node.is_leaf_node = false;
            split_node.node_type = 0;
            split_node.attribute = attr;
            split_node.node_level = node_level;
            split_node.class1_instances = class1_count;
            split_node.class2_instances = class2_count;
            split_node.node_children = new ArrayList<>(attr.attribute_values.length);
            for(int i = 0; i < attr.attribute_values.length; i++){
                ArrayList<HashMap<String, String>> data = new ArrayList<>();
                for(int j = 0 ;j < data_instance_list.size(); j++){
                    if(data_instance_list.get(j).get(attr.attribute_name).equalsIgnoreCase(attr.attribute_values[i])){
                        data.add(data_instance_list.get(j));
                    }
                }
                split_node.node_children.add(constructTree(data, node_level+1, m));
            }
            return split_node;
        }else{
            //continuous
            int attr_num = -1;
            TreeNode split_node = new TreeNode();
            for(int i = 0; i < attribute_list.size(); i++){
                if(attribute_list.get(i).attribute_name.equalsIgnoreCase(attr.attribute_name)){
                    attr_num = i;
                    break;
                }
            }
            split_node.threshold = get_continuous_attribute_entropy(data_instance_list,attr_num).threshold;
            split_node.node_label = "";
            split_node.is_leaf_node = false;
            split_node.node_type = 0;
            split_node.attribute = attr;
            split_node.node_level = node_level;
            split_node.class1_instances = class1_count;
            split_node.class2_instances = class2_count;
            split_node.node_children = new ArrayList<>(2);
            ArrayList<HashMap<String, String>> less_than_data = new ArrayList<>();
            ArrayList<HashMap<String, String>> greater_than_data = new ArrayList<>();

            for(int j = 0 ;j < data_instance_list.size(); j++) {
                if (Double.parseDouble(data_instance_list.get(j).get(attr.attribute_name)) <= split_node.threshold) {
                    less_than_data.add(data_instance_list.get(j));
                } else {
                    greater_than_data.add(data_instance_list.get(j));
                }
            }
                split_node.node_children.add(constructTree(less_than_data, node_level+1, m));
                split_node.node_children.add(constructTree(greater_than_data, node_level+1, m));
            return split_node;
        }
    }

    private static double getEntropy(double numerator, double denominator) {
        if (numerator == 0 || denominator == 0) {
            return 0;
        }
        return (numerator / denominator) * getlog2(numerator / denominator);
    }


    private static double get_nominal_attribute_entropy(ArrayList<HashMap<String, String>> data_instance_list, int attribute_num){
        double entropy = 0.0;

        int labelAttrCount = attribute_list.get(attribute_num).attribute_values.length;
        String[] attributeValues = attribute_list.get(attribute_num).attribute_values;

        for (int i = 0; i < labelAttrCount; i++) {
            int labelSubCountFirstClass = 0;
            int labelSubCountSecondClass = 0;
            int labelCount = 0;
            for (int j = 0; j < data_instance_list.size(); j++) {
                if (data_instance_list.get(j).get(attribute_list.get(attribute_num).attribute_name).equalsIgnoreCase(attributeValues[i])) {
                    if (data_instance_list.get(j).get("class_label").equalsIgnoreCase(class_labels[0])){
                        labelSubCountFirstClass++;
                    } else if (data_instance_list.get(j).get("class_label").equalsIgnoreCase(class_labels[1])){
                        labelSubCountSecondClass++;
                    }
                    labelCount++;
                }
            }
            double subEntropy =
                    getEntropy(labelSubCountFirstClass, labelCount) +
                            getEntropy(labelSubCountSecondClass, labelCount);
            entropy += ((double) labelCount / data_instance_list.size()) * subEntropy;

        }
        return -entropy;

    }

    private static ContinuousAttribute get_continuous_attribute_entropy(ArrayList<HashMap<String, String>> data_instance_list, int attribute_num){
        double[] attribute_values = new double[data_instance_list.size()];

        for(int i = 0; i < data_instance_list.size(); i++){
            attribute_values[i] = Double.parseDouble(data_instance_list.get(i).get(attribute_list.get(attribute_num).attribute_name));
        }
        Arrays.sort(attribute_values);
        TreeSet<Double> values_without_duplicates = new TreeSet<Double>();
        for(int i = 0; i < attribute_values.length; i++){
            values_without_duplicates.add(attribute_values[i]);
        }

        int k = 0;
        for(double d : values_without_duplicates){
            attribute_values[k++] = d;
        }
        double[] possible_candidate_splits = new double[values_without_duplicates.size()];
        for(int i = 0; i < values_without_duplicates.size()-1; i++){
            possible_candidate_splits[i] = (attribute_values[i] + attribute_values[i+1])/2.0;
        }
        /*if (attribute_list.get(attribute_num).attribute_name.equalsIgnoreCase("thalach")) {
            System.out.println();
            System.out.println(Arrays.toString(possible_candidate_splits));
        }*/
        attribute_values = new double[data_instance_list.size()];

        for(int i = 0; i < data_instance_list.size(); i++){
            attribute_values[i] = Double.parseDouble(data_instance_list.get(i).get(attribute_list.get(attribute_num).attribute_name));
        }

        double entropy[] = new double[possible_candidate_splits.length];


        for(int i = 0; i < possible_candidate_splits.length; i++){

            int l_c1_cnt = 0;
            int g_c1_cnt = 0;
            int l_c2_cnt = 0;
            int g_c2_cnt = 0;
            int l_count = 0;
            int g_count = 0;
            for(int j = 0; j < data_instance_list.size(); j++){
                if(attribute_values[j] <= possible_candidate_splits[i]){
                    l_count++;
                    if(data_instance_list.get(j).get("class_label").equalsIgnoreCase(class_labels[0])){
                        l_c1_cnt++;
                    }else if(data_instance_list.get(j).get("class_label").equalsIgnoreCase(class_labels[1])){
                        l_c2_cnt++;
                    }
                }else{
                    g_count++;
                    if(data_instance_list.get(j).get("class_label").equalsIgnoreCase(class_labels[0])){
                        g_c1_cnt++;
                    }else if(data_instance_list.get(j).get("class_label").equalsIgnoreCase(class_labels[1])){
                        g_c2_cnt++;
                    }
                }
            }

            double prob_1 = 0.0, prob_2 = 0.0, prob_pos_1 = 0.0;
            double prob_pos_2 = 0.0, prob_neg_1 = 0.0, prob_neg_2 = 0.0;
            if (l_count != 0) {
                prob_1 = (double) l_count / (double) data_instance_list.size();
                if (l_c1_cnt != 0) {
                    prob_pos_1 = (double) l_c1_cnt / (double) l_count;
                } else {
                    prob_pos_1 = 0.0;
                }

                if (l_c2_cnt != 0) {
                    prob_pos_2 = (double) l_c2_cnt / (double) l_count;
                } else {
                    prob_pos_2 = 0.0;
                }
            } else {
                prob_1 = 0.0;
                prob_pos_1 = 0.0;
                prob_pos_2 = 0.0;
            }

            if (g_count != 0) {
                prob_2 = (double) g_count / (double) data_instance_list.size();
                if (g_c1_cnt != 0) {
                    prob_neg_1 = (double) g_c1_cnt / (double) g_count;
                } else {
                    prob_neg_1 = 0.0;
                }

                if (g_c2_cnt != 0) {
                    prob_neg_2 = (double) g_c2_cnt / (double) g_count;
                } else {
                    prob_neg_2 = 0.0;
                }
            } else {
                prob_2 = 0.0;
                prob_neg_1 = 0.0;
                prob_neg_2 = 0.0;
            }

            double log2prob_pos_1 = 0.0, log2prob_pos_2 = 0.0, log2prob_neg_1 = 0.0, log2prob_neg_2 = 0.0;
            if (prob_pos_1 == 0.0) {
                log2prob_pos_1 = 0.0;
            } else {
                log2prob_pos_1 = getlog2(prob_pos_1);
            }
            if (prob_pos_2 == 0.0) {
                log2prob_pos_2 = 0.0;
            } else {
                log2prob_pos_2 = getlog2(prob_pos_2);
            }
            if (prob_neg_1 == 0.0) {
                log2prob_neg_1 = 0.0;
            } else {
                log2prob_neg_1 = getlog2(prob_neg_1);
            }
            if (prob_neg_2 == 0.0) {
                log2prob_neg_2 = 0.0;
            } else {
                log2prob_neg_2 = getlog2(prob_neg_2);
            }
            entropy[i] = -((prob_1 * ((prob_pos_1 * log2prob_pos_1) + (prob_pos_2 * log2prob_pos_2))) + (prob_2 * ((prob_neg_1 * log2prob_neg_1) + (prob_neg_2 * log2prob_neg_2))));
        }
        double min_entropy = entropy[0];
        int min_entropy_index = 0;

        for(int i = 1; i < entropy.length; i++){
            if(entropy[i] < min_entropy){
                min_entropy = entropy[i];
                min_entropy_index = i;
            }
        }

        ContinuousAttribute continuousAttribute = new ContinuousAttribute();
        continuousAttribute.entropy = min_entropy;
        continuousAttribute.threshold = possible_candidate_splits[min_entropy_index];
        return continuousAttribute;
    }

    private static void print_tree(TreeNode node){
        int nodeLevel = node.node_level;

        if (node.node_children != null) {
            for (int j = 0; j < node.node_children.size(); j++) {
                int class1 = 0;
                int class2 = 0;
                if (node.node_level > 0) {
                    for (int i = 0; i < nodeLevel; i++) {
                        System.out.print("|");
                        System.out.print("\t");
                    }
                }
                class1 = node.class1_instances;
                class2 = node.class2_instances;
                //System.out.println("Class 1 instances : " + class1);
                //System.out.println("Class 2 instances : " + class2);
                System.out.print(node.attribute.attribute_name);
                if (node.attribute.attribute_type == 0) {
                    System.out.print(" = ");
                    System.out.print(node.attribute.attribute_values[j]);
                } else if (node.attribute.attribute_type == 1) {
                    if (node.node_children.indexOf(node.node_children.get(j)) == 0) {
                        System.out.print(" <= ");
                    } else if (node.node_children.indexOf(node.node_children.get(j)) == 1) {
                        System.out.print(" > ");
                    }
                    System.out.print(node.threshold);
                }

                if (node.node_children.get(j).is_leaf_node) {
                    System.out.print(": ");
                    if (node.node_children.get(j).node_label == null) {
                        System.out.println(class_labels[0]);
                    } else {
                        System.out.println(node.node_children.get(j).node_label);
                    }
                } else {
                    System.out.println();
                    print_tree(node.node_children.get(j));
                }
            }
        }

    }

    public static void main(String args[]){
        String train_file = args[0];
        String test_file = args[1];
        int m = Integer.parseInt(args[2]);

        TreeNode root = train_decision_tree(train_file, m);
        test_decision_tree(test_file, root);

    }

    private static void test_decision_tree(String test_file, TreeNode tree_node){
        ArffLoader arffLoader = new ArffLoader();
        File input_file = new File(test_file);
        try {
            arffLoader.setFile(input_file);
            Instances data_instance = arffLoader.getDataSet();

            HashMap<String, String> data_hash = new HashMap<String, String>();
            ArrayList<HashMap<String, String>> test_instances = new ArrayList<>(data_instance.numInstances());
            for(int k = 0; k < data_instance.numInstances(); k++){
                data_hash = new HashMap<>();
                for(int l = 0; l < data_instance.numAttributes(); l++){
                    if(l == data_instance.numAttributes()-1){
                        data_hash.put("class_label", data_instance.instance(k).stringValue(l));
                    }else {
                        if (attribute_list.get(l).attribute_type == 0) {
                            data_hash.put(attribute_list.get(l).attribute_name, data_instance.instance(k).stringValue(l));
                        } else if (attribute_list.get(l).attribute_type == 1) {
                            data_hash.put(attribute_list.get(l).attribute_name, String.valueOf(data_instance.instance(k).value(l)));
                        }
                    }
                }
                test_instances.add(data_hash);
            }
            /*for(int i = 0; i < test_instances.size(); i++){
                for(int j = 0; j < attribute_list.size(); j++){
                    System.out.println(attribute_list.get(j).attribute_name + " : " +
                            test_instances.get(i).get((attribute_list.get(j).attribute_name)));
                }
                    System.out.println("class label : "  +
                            test_instances.get(i).get(("class_label")));
            }*/

            int numCorrectlyClassified = 0;
            System.out.println("<Predictions for the Test Set Instances>");

            for (int i = 0; i < test_instances.size(); i++) {
                String predictedClassLabel = "";
                String actualClassLabel = test_instances.get(i).get("class_label");
                System.out.print(String.format("%3d", i+1) + ": Actual: " + actualClassLabel + " Predicted: ");
                predictedClassLabel = predict_instance_class(tree_node, test_instances.get(i));
                if (predictedClassLabel.equalsIgnoreCase(actualClassLabel)) {
                    numCorrectlyClassified++;
                }
                System.out.println(predictedClassLabel);
            }
            System.out.println("Number of correctly classified: " + numCorrectlyClassified +
                    " Total number of test instances: " + test_instances.size());


        }catch (IOException e){
            System.out.println(e.getMessage());
        }
    }

    private static String predict_instance_class(TreeNode node, HashMap<String, String> data){
        Attribute attribute = node.attribute;
        if (node.is_leaf_node) {
            return node.node_label;
        } else {
            if (attribute.attribute_type == 1) {
                double splitValue = node.threshold;
                if (Double.parseDouble(data.get(attribute.attribute_name)) <= splitValue) {
                    return (predict_instance_class(node.node_children.get(0), data));
                } else if (Double.parseDouble(data.get(attribute.attribute_name)) > splitValue) {
                    return (predict_instance_class(node.node_children.get(1), data));
                }
            } else if (attribute.attribute_type == 0) {
                String instanceAttributeValue = data.get(attribute.attribute_name);
                int attributeIndex = -1;
                for (int l = 0; l < attribute.attribute_values.length; l++) {
                    if (instanceAttributeValue.equalsIgnoreCase(attribute.attribute_values[l])) {
                        attributeIndex = l;
                        break;
                    }
                }
                return (predict_instance_class(node.node_children.get(attributeIndex), data));
            }
        }
        return "";
    }


    private static TreeNode train_decision_tree(String train_file, int m){
        ArffLoader arffLoader = new ArffLoader();
        File input_file = new File(train_file);
        try {
            arffLoader.setFile(input_file);
            Instances data_instance = arffLoader.getDataSet();
            class_labels = new String[2];
            class_labels[0] = data_instance.attribute(data_instance.numAttributes() - 1).value(0);
            class_labels[1] = data_instance.attribute(data_instance.numAttributes() - 1).value(1);

            //System.out.println(class_labels[0]);
            //System.out.println(class_labels[1]);

            attribute_list = new ArrayList<Attribute>(data_instance.numInstances()-1);

            for(int i = 0; i < data_instance.numAttributes()-1; i++){
                Attribute attribute = new Attribute();
                attribute.attribute_name = data_instance.attribute(i).name();
                if(data_instance.attribute(i).isNominal()) {
                    attribute.attribute_type = 0; //Nominal
                    attribute.attribute_values = new String[data_instance.attribute(i).numValues()];
                    for(int j = 0; j < data_instance.attribute(i).numValues(); j++){
                        attribute.attribute_values[j] = data_instance.attribute(i).value(j);
                    }
                }else if(data_instance.attribute(i).isNumeric()) {
                    attribute.attribute_type = 1; //Numeric
                }
//                System.out.println("Attribute name : " + attribute.attribute_name);
//                System.out.println("Attribute type : " + attribute.attribute_type);
                /*if(attribute.attribute_type == 0) {
                    for (int l = 0; l < attribute.attribute_values.length; l++) {
                        System.out.println("Attribute value : " + attribute.attribute_values[l]);
                    }
                }*/
                attribute_list.add(attribute);
            }
            num_attributes = data_instance.numAttributes()-1;

            HashMap<String, String> data_hash = new HashMap<String, String>();
            instances = new ArrayList<>(data_instance.numInstances());
            for(int k = 0; k < data_instance.numInstances(); k++){
                data_hash = new HashMap<>();
                for(int l = 0; l < data_instance.numAttributes(); l++){
                    if(l == data_instance.numAttributes()-1){
                        data_hash.put("class_label", data_instance.instance(k).stringValue(l));
                    }else {
                        if (attribute_list.get(l).attribute_type == 0) {
                            data_hash.put(attribute_list.get(l).attribute_name, data_instance.instance(k).stringValue(l));
                        } else if (attribute_list.get(l).attribute_type == 1) {
                            data_hash.put(attribute_list.get(l).attribute_name, String.valueOf(data_instance.instance(k).value(l)));
                        }
                    }
                }
                instances.add(data_hash);
            }
            /*for(int i = 0; i < instances.size(); i++){
                for(int j = 0; j < attribute_list.size(); j++){
                    System.out.println(attribute_list.get(j).attribute_name + " : " +
                            instances.get(i).get((attribute_list.get(j).attribute_name)));
                }
                    System.out.println("class label : "  +
                            instances.get(i).get(("class_label")));
            }*/
            int root_node_level = 0;
            TreeNode root_node = constructTree(instances,root_node_level, m);
            print_tree(root_node);
            return root_node;
        }catch (IOException e){
            System.out.println(e.getMessage());
        }
        return null;
    }

    public static class Attribute{
        public String attribute_name;
        public int attribute_type;
        public String[] attribute_values;
    }

    public static class ContinuousAttribute{
        public double entropy;
        public double threshold;
    }
}
