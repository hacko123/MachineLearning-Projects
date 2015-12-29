__author__ = 'kavin'
# This file is to extract the attributes, labels and its values from the input training or testing set
# NO LIBRARIES USED, so have to grep through every line and using regex(re) take out the desired data
import re


#using the basic while loop to read and grep through lines
#@attribute 'fbs' { t, f}
def getAttributes(training_set_temp):
    attributes_in_order = []
    attributes_list = {}
    class_labels = []
    #opening the arff file to read the attributes
    with open(training_set_temp, 'r') as file_open:
        while True:
            #reading the line
            string_temp = file_open.readline()
            #print(string_temp)
            #if it starts with @attribute grep for the values inside attributes
            #@attribute 'fbs' { t, f}
            if string_temp.startswith("@attribute"):
                rowwisesplit = string_temp.split(" ")
                #removing the '' from the values using regex
                #print(rowwisesplit[1])
                rowwisesplit[1] = re.sub(r"^'",'',rowwisesplit[1])
                rowwisesplit[1] = re.sub(r"'$",'',rowwisesplit[1])
                #print(rowwisesplit[1])
                attributes_in_order.append(rowwisesplit[1])
                #if the values are nominal, then look for '{' and split it
                if "{" in string_temp:
                    #print "Nominal values, so splitting it up : {0}".format(string_temp)
                    val = "".join((re.findall(r'\{(.*?)\}', string_temp)[0]).split())
                    val = re.sub(r"'",'',val)
                    attributes_list[rowwisesplit[1]] = val.split(",")
                else:
                    attributes_list[rowwisesplit[1]] = 0
            #if the attributes are over, break the while
            elif not string_temp.startswith("@") and not string_temp.startswith("%"):
                break
    #print("attributes_in_order")
    #print(attributes_in_order)
    #print("attributes_list")
    #print(attributes_list)
    class_labels = attributes_list['class'] or attributes_list['Class']
    return(attributes_list,attributes_in_order,class_labels)

#50,female,atyp_angina,120,244,f,normal,162,no,1.1,up,0,normal,negative
def getData(training_set_temp,attributes_list,attributes_in_order):
    training_set_formatted = []
    with open(training_set_temp, 'r') as file_open:
        while True:
            string_temp = file_open.readline()
            #check to see EOF
            if string_temp == '':
                break
            else:
                #50,female,atyp_angina,120,244,f,normal,162,no,1.1,up,0,normal,negative
                if not string_temp.startswith("@") and not string_temp.startswith("%"):
                    newdict = {}
                    #removing new line
                    string_temp = re.sub(r"\n", '', string_temp)
                    string_temp = re.sub(r"'", '', string_temp)
                    string_temp = re.sub(r"\r", '', string_temp)
                    #splitting it
                    rowwisesplit = string_temp.split(",")
                    #typecasting the float and str values
                    for i, val in enumerate(attributes_in_order):
                        rowwisesplit[i] = float(rowwisesplit[i]) if (attributes_list[val] == 0) else rowwisesplit[i]
                    #adding the labels to the attributes in order
                    for i in range(len(attributes_in_order)):
                        newdict[attributes_in_order[i]] = rowwisesplit[i]
                    #append the dict to the list traning_set_formatted
                    training_set_formatted.append(newdict)
    return training_set_formatted
