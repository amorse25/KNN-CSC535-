""" 
Program: KNN.py
Programmed By: Adam Morse
Description: An implementation of the KNN classification algorithm using
    weighted voting.  
Slate Folder: Morse1
"""

import numpy as np
import csv
import math
import operator 


""" Function to load the csv training file """
def loadDatasetTrain(filename, trainingSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)         # Grab each line in the file
        dataset = list(lines)               # Store in a list
        dataset = dataset[1:]               # Ignore the first row in the file
        for x in range(len(dataset)-1):     
            for y in range(len(dataset[x])): 
                dataset[x][y] = float(dataset[x][y])        # Convert to Float
            trainingSet.append(dataset[x])                  # Store in trainingSet

""" Function to load the csv test file """       
def loadDatasetTest(filename, testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)         # Grab each line in the file
        dataset = list(lines)               # Store in a list
        dataset = dataset[1:]               # Ignore the first row in the file
        for x in range(len(dataset)-1):
            for y in range(len(dataset[x])):
                dataset[x][y] = float(dataset[x][y])        # Convert to Float
            testSet.append(dataset[x])                      # Store in testSet

""" Function to compute the Euclidean Distance """ 
def euclideanDistance(testValue, trainingValue, length):
    distance = 0
    for x in range(length):
        distance += pow((testValue[x] - trainingValue[x]), 2)
    return math.sqrt(distance)

""" Function to compute the nearest neighbors to a test sample instance """ 
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)      # Calculate the distance between instances
        distances.append((trainingSet[x], dist))        # Store in distances array
    distances.sort(key=operator.itemgetter(1))          # Sort from greatest to least
    
    neighbors = []                                      # Store neighbor (distance, class values)
    weightedVotes = np.zeros(k)                         # Array of zeros, length of k
    totalSum = 0.0                                      # Store total sum for class instances
    
    for x in range(k):
        weightedVotes[x] += pow((1.0 / distances[x][1]), 2)         # Use weighted voting to determine class
        totalSum += weightedVotes[x]                                # Grab total sum found at each class iteration

    for i in range(len(weightedVotes)):                             # Divide by total sum at each class iteration
        weightedVotes[i] = weightedVotes[i] / totalSum
        neighbors.append((distances[i][0], weightedVotes[i]))       # Store values based on distance, class
    return neighbors
        

""" Function to compute the best class based on votes (Weighted Voting) """ 
def getVotes(neighbors):     
    classVotes = dict()
    totalClassVotes = dict()

    for i in neighbors:
        if i[0][0] in classVotes:
            classVotes[i[0][0]].append(i[1])            # Append votes to dictionary: (class, distances)
        else:
            classVotes[i[0][0]] = [i[1]]
    for key in classVotes:
        totalClassVotes[key] = sum(classVotes[key])     # Sum the total distances for each class iteration found

    sortedVotes = sorted(totalClassVotes.items(), key=operator.itemgetter(1), reverse=True)     # Sort from greatest to least
    return sortedVotes[0][0]                # Return the class with the greatest sum
        
""" Function to compute the Accuracy of sampling K values """ 
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:         # Grab the amount of correctly computed classes
            correct += 1
    return correct

""" Function to find the best value for K | Takes a very long time to compute.
    Sampling part of the training set didnt seem to yield accurate results, 
    decided to do the whole training set                                  """
def findBestK(k):
    kValues = dict()
    kSet = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    trainingSet = []        # Store training values    
    testSet = []            # Store test values
    predictions = []        # Store prediction values
    
    loadDatasetTrain('MNIST_train.csv', trainingSet)        # Load training set file
    loadDatasetTest('MNIST_test.csv', testSet)        # Load training set file
    
    for i in kSet:
        print("k:", i)
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], i)        # Grab the nearest neighbors
            result = getVotes(neighbors)                                # Grab the best class
            predictions.append(result)                                  # Store into predictions array to use for accuracy
        accuracy = getAccuracy(testSet, predictions)                    # Grab accuracy of k sampling
        kValues[i] = accuracy                                           # Store accuracy values for k into dictionary
        predictions = []
        print(kValues)
    sortedKValues = sorted(kValues.items(), key=operator.itemgetter(1), reverse=True)
    print("sorted", sortedKValues)
    return sortedKValues[0][0]                                          # Return highest accuracy rate of k
        
""" Main """ 
def main():
    if(__name__ == "__main__"):
        
        trainingSet = []        # Store training values    
        testSet = []            # Store test values
        predictions = []        # Store prediction values
        
        
        """ Function call to determine best k value. This function takes a long time to run as it goes through
        all 50 test samples for each k iteration. 
        
        #k = 1
        #k = findBestK(k)   # Function call to find best k value | k = 7 && k = 9 yield best results: 5 missed | 89.79%
        
        """
        
        
        k = 7
        loadDatasetTrain('MNIST_train.csv', trainingSet)        # Load training set file
        loadDatasetTest('MNIST_test.csv', testSet)              # Load test set file
        print("K = ", k)
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)        # Grab the nearest neighbors
            result = getVotes(neighbors)                                # Grab the best class
            predictions.append(result)                                  # Store into predictions array to use for accuracy
            print("Desired class: " + repr(testSet[x][0]) + " computed class: " + repr(result))
            
        accuracy = getAccuracy(testSet, predictions)                    # Grab accuracy of k sampling
        print('Accuracy rate: ', repr((accuracy * 100.0 ) /float(len(testSet)))  + '%')         # Correct samples (Accuracy rate)
        print("Number of misclassified test samples: ", float(len(testSet)) - accuracy)         # Incorrect samples (Missed samples)
        print("Total number of test samples: ", len(testSet) + 1)       # Add one because popped header row initially
	
main()       


#---------------------------------End of Program-------------------------------     
    
    
    
    
    
    