__author__ = 'kavin'
import inputparser
import math
import sys
import operator

global training_set
global testing_set
attributes_list = []
attributes_in_order = []
class_labels = []
training_set_formatted = []
test_attributes_list = []
testing_set_formatted =[]
prob_list1 = 0
prob_list2 = 0
prob_class_1 = 0
prob_class_2 = 0
class1 = 0
class2 = 0


def get_data():
    global training_set_formatted, testing_set_formatted, attributes_list, class_labels, attributes_in_order
    attributes_list,attributes_in_order,class_labels = inputparser.getAttributes(training_set)
    training_set_formatted = inputparser.getData(training_set, attributes_list, attributes_in_order)
    testing_set_formatted = inputparser.getData(testing_set, attributes_list, attributes_in_order)
    return

def find_prob_train(attr,attr_value):

    global attributes_list
    attr_value_count_1 =0
    attr_value_count_2 =0

    for inst in training_set_formatted:
        for attribute in inst:
            if (attr == attribute and attr_value == inst[attribute] and inst["class"] == class_labels[1]):
                    attr_value_count_1 += 1
            elif (attr == attribute and attr_value == inst[attribute] and inst["class"] == class_labels[0]):
                     attr_value_count_2 += 1

    num1 = attr_value_count_1 + 1
    den1 = class1 + len(attributes_list[attr])

    num2 = attr_value_count_2 + 1
    den2 = class2 + len(attributes_list[attr])

    prob_attr1 = float(num1)/float(den1)
    prob_attr2 = float(num2)/float(den2)

    return prob_attr1, prob_attr2

def naive_bayes():
    global testing_set_formatted
    count =0
    for val in attributes_in_order:
        if val != "class":
            print val + " " + "class"
    print "\n"
    for inst in testing_set_formatted:
        attribute_prob_list1 = []
        attribute_prob_list2 = []
        new_prob1 = 1
        new_prob2 = 1
        for attr in inst:
            if attr != "class":
                prob_attr1, prob_attr2 = find_prob_train(attr,inst[attr])
                attribute_prob_list1.append(prob_attr1)
                attribute_prob_list2.append(prob_attr2)
        for i in attribute_prob_list1:
            new_prob1 = i * new_prob1
        for i in attribute_prob_list2:
            new_prob2 = new_prob2 * i

        overall_prob1 = float(new_prob1) * float(prob_class_1)/ (float(new_prob1) * float(prob_class_1) + float(new_prob2) * float(prob_class_2))
        overall_prob2 = float(new_prob2) * float(prob_class_2)/ (float(new_prob1) * float(prob_class_1) + float(new_prob2) * float(prob_class_2))

        if (overall_prob1 > overall_prob2):
            print class_labels[1] + " " + inst["class"], overall_prob1
            if (class_labels[1] == inst["class"]):
                count+=1
        else:
            print class_labels[0] + " " + inst["class"], overall_prob2
            if (class_labels[0] == inst["class"]):
                count+=1
    print "\n"
    print count

def tan_prob_find(at_1,a,at_2,b,c):
    c1=0
    c2=0
    c3=0
    for inst in training_set_formatted:
        if (a == inst[at_1] and b == inst[at_2] and c == inst["class"]):
            c1+=1
        if (a == inst[at_1] and c == inst["class"]):
            c2+=1
        if (b == inst[at_2] and c == inst["class"]):
            c3+=1
    tan_num1 = c1+1
    tan_den1 = len(training_set_formatted) + (len(attributes_list[at_1]) * len(attributes_list[at_2]) * len(class_labels))
    tan_prob1 = float(tan_num1)/float(tan_den1)

    tan_den2 = class_count(c) + (len(attributes_list[at_1]) * len(attributes_list[at_2]))
    tan_prob2 = float(tan_num1)/float(tan_den2)

    tan_num2 = c2 + 1
    tan_den3 = class_count(c) + len(attributes_list[at_1])
    tan_prob3 = float(tan_num2)/float(tan_den3)

    tan_num3 = c3 + 1
    tan_den4 = class_count(c) + len(attributes_list[at_2])
    tan_prob4 = float(tan_num3)/float(tan_den4)

    new_prob = tan_prob1 * math.log((tan_prob2/(tan_prob3 * tan_prob4)),2)

    return new_prob



def class_count(c):
    c_count = 0
    for inst in training_set_formatted:
        if inst["class"] == c:
            c_count+=1
    return c_count

def prob_tan(inst1,attr,parent_list):

    var1 = 0
    var2 = 0
    variable1 = 0
    variable2 = 0
    length = len(attributes_list[attr])
    for train_val in training_set_formatted:
        if (train_val[attr] == inst1[attr] and train_val["class"] == class_labels[0]):
            found = 0
            for parent in parent_list:
                if parent != -1:
                    if train_val[attributes_in_order[parent]] == inst1[attributes_in_order[parent]]:
                        found += 1
            if found == len(parent_list) or parent == -1:
                var1 += 1

        if (train_val["class"] == class_labels[0]):
            found = 0
            for parent in parent_list:
                if parent != -1:
                    if train_val[attributes_in_order[parent]] == inst1[attributes_in_order[parent]]:
                        found += 1
            if found == len(parent_list) or parent == -1:
                variable1 += 1

        if (train_val[attr] == inst1[attr] and train_val["class"] == class_labels[1]):
            found = 0
            for parent in parent_list:
                if parent != -1:
                    if train_val[attributes_in_order[parent]] == inst1[attributes_in_order[parent]]:
                        found += 1
            if found == len(parent_list) or parent == -1:
                var2 += 1

        if (train_val["class"] == class_labels[1]):
            found = 0
            for parent in parent_list:
                if parent != -1:
                    if train_val[attributes_in_order[parent]] == inst1[attributes_in_order[parent]]:
                        found += 1
            if found == len(parent_list) or parent == -1:
                variable2 += 1

    numerator1 = var1 + 1
    denominator1 = variable1 + length
    p_num1 = float(numerator1)/float(denominator1)

    numerator2 = var2 + 1
    denominator2 = variable2 + length
    p_num2 = float(numerator2)/float(denominator2)

    return p_num1, p_num2


def tan_bayes():

    global testing_set_formatted, final_prob1, final_prob2
    newlist = []
    index_list = []
    oldlist = attributes_in_order
    count =0
    spanning_tree = {}
    spanning_tree_name = {}
    big_probability = []
    # for val in attributes_in_order:
    #     if val != "class":
    #         print val + " " + "class"
    # print "\n"

    for i in range(0,len(attributes_in_order)-1):
        edge_list = []
        for j in range(0,len(attributes_in_order)-1):
            probability = []
            edge_value = 0
            at_1 = attributes_in_order[i]
            at_2 = attributes_in_order[j]
            if (i == j):
                edge_list.append(-1.0)
            else:
                for a in attributes_list[at_1]:
                    for b in attributes_list[at_2]:
                        for c in class_labels:
                            probability.append(tan_prob_find(at_1,a,at_2,b,c))

                for val in probability:
                    edge_value = val + edge_value

                edge_list.append(edge_value)

        big_probability.append(edge_list)

    newlist.append(attributes_in_order[0])

    for val in range(0,len(attributes_in_order)-2):
        edges = {}
        max_edges = []
        for vertex_1 in newlist:
            for vertex_2 in oldlist:
                if (vertex_1 != "class" and vertex_2 != "class" and vertex_1!=vertex_2 and vertex_2 not in newlist):
                    vertex_1_index = attributes_in_order.index(vertex_1)
                    vertex_2_index = attributes_in_order.index(vertex_2)
                    edges[(vertex_1,vertex_2)] = big_probability[vertex_1_index][vertex_2_index]

        max_edges = max(edges.iteritems(),key = operator.itemgetter(1))[0]
        maxedgevalue = max(edges.iteritems(),key = operator.itemgetter(1))[1]

        newlist.append(max_edges[1])

        spanning_tree[(attributes_in_order.index(max_edges[0]), attributes_in_order.index(max_edges[1]))] = maxedgevalue
        spanning_tree_name[max_edges[0],max_edges[1]] = maxedgevalue

    parent_child = {}

    parent_child[0] = [-1]

    for item in spanning_tree:
        parent_child[item[1]] = [item[0]]

    for name in attributes_in_order:
         if name != "class":
             string_l = []
             for val in parent_child[attributes_in_order.index(name)]:
                 if val != -1:
                     string_l.append(attributes_in_order[val])
             string = " ".join(string_l)
             print str(name) + " " +str(string) + " " + "class"


    print "\n"
    prob_class_x, prob_class_y = get_prob()

    for inst1 in testing_set_formatted:
        final_prob1 = []
        final_prob2 = []
        prob_1 = 1
        prob_2 = 1
        for attr in inst1:
            if (attr != "class"):
                parent = attributes_in_order.index(attr)
                probability_1, probability_2 = prob_tan(inst1,attr,parent_child[parent])
                final_prob1.append(probability_1)
                final_prob2.append(probability_2)


        for value in final_prob1:
            prob_1 *= value
        for value in final_prob2:
            prob_2 *= value

        reqd_prob1 = (float(prob_1) * float(prob_class_x)) / ((float(prob_1) * float(prob_class_x)) + (float(prob_2) * float(prob_class_y)))
        reqd_prob2 = 1.0 - float(reqd_prob1)

        if (reqd_prob1 > reqd_prob2):
            print class_labels[0] + " " + inst1["class"] + " " + str(reqd_prob1)
            if (class_labels[0] == inst1["class"]):
                count+=1
        else:
            print class_labels[1] + " " + inst1["class"] + " " + str(reqd_prob2)
            if (class_labels[1] == inst1["class"]):
                count+=1
    print "\n"
    print count
    return


def get_prob():

    class1a = 0
    class2a = 0

    for inst in training_set_formatted:
        if inst["class"] == class_labels[0]:
            class1a+=1
        elif inst["class"] == class_labels[1]:
            class2a+=1
    prob_classa = float(class1a+1)/float(len(training_set_formatted)+2)
    prob_classb = float(class2a+1)/float(len(training_set_formatted)+2)
    return prob_classa, prob_classb

def class_prob():
    global training_set_formatted
    global class1
    global class2
    global prob_class_1
    global prob_class_2

    for inst in training_set_formatted:
        if inst["class"] == class_labels[1]:
            class1+=1
        elif inst["class"] == class_labels[0]:
            class2+=1
    prob_class_1 = float(class1+1)/float(len(training_set_formatted))
    prob_class_2 = float(class2+1)/float(len(training_set_formatted))
    return



def main():

    global training_set,testing_set
    program_args = sys.argv
    training_set = program_args[1]
    testing_set = program_args[2]
    mode = program_args[3]
    get_data()
    if (mode == "n"):
        class_prob()
        naive_bayes()
    elif (mode == "t"):
        tan_bayes()
    else:
        print "Usage : bayes <train-set-file> <test-set-file> <n|t>"

if __name__ == "__main__":
    main()
