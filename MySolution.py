import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import minimize
from pulp import *

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes
        self.binaryClassifiers = []
        self.lp_class = {}
        self.output_class = {}
    
    def train_binary_classifier(self, trainX, binary_trainY):
        L = 10
        # initial_params = [a, b] but zeroes
        initial_params = np.zeros(trainX.shape[1]+1)

        # objective function
        def linear_svm_objective(params, x, s, L):
            a = params[:-1]
            b = params[-1]
            hinge_loss = np.maximum(1 - s * (np.dot(x, a) + b), 0)
            return np.sum(hinge_loss) + np.dot(a, a) * L

        # Solve optimization problem
        svm_optimizer = minimize(linear_svm_objective, x0=initial_params, args=(trainX, binary_trainY, L))

        # Extract the optimized parameters
        a_optimal = svm_optimizer.x[:-1]
        b_optimal = svm_optimizer.x[-1]

        return a_optimal, b_optimal
    
    def train(self, trainX, trainY):
        count = 0
        for i in range(trainY.shape[0]):
            if trainY[i] not in self.lp_class:
                self.lp_class[trainY[i]] = count
                self.output_class[count] = trainY[i]
                count += 1
            
            trainY[i] = self.lp_class[trainY[i]]


        # Initializes and trains K(K-1)/2 linear classifiers.
        for i in range(self.K):
            for j in range(i+1, self.K):
                binary_trainY = (trainY == i).astype(int) - (trainY == j).astype(int)  # turns trainY to take value 1 if equal to i, -1 if equal to j, 0 if neither
                classifier = self.train_binary_classifier(trainX, binary_trainY)
                self.binaryClassifiers.append(((i,j,classifier)))
                
    def predict(self, testX):
        # Make predictions using all binary classifiers
        predY = np.zeros((testX.shape[0], self.K))

        for i, j, classifier in self.binaryClassifiers:
            a = classifier[0]
            b = classifier[1]
            soln = np.dot(testX, a) + b

            predY[:, i] += (soln > 0).astype(int)
            predY[:, j] += (soln <= 0).astype(int)

        out = np.argmax(predY, axis=1)
        for i in range(out.shape[0]):
            out[i] = self.output_class[out[i]]

        return out


        # Return the predicted class labels of the input data (testX)
        # return predY
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K):
        self.K = K  # number of classes
        self.cluster_centers = None
        self.labels = None

    def cosine_similarity(self, a,b):
        return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))


        
    def train(self, trainX):
        N = trainX.shape[0]
        M = trainX.shape[1]
        # keep taking K random points from dataset without repetition to act as cluster center until their similarities are small
        self.cluster_centers = trainX[np.random.choice(N, size=self.K, replace=False)]
        count = 0
        while self.cosine_similarity(self.cluster_centers[0],self.cluster_centers[1]) > 0.5 or self.cosine_similarity(self.cluster_centers[0],self.cluster_centers[2]) > 0.5 or self.cosine_similarity(self.cluster_centers[2],self.cluster_centers[1]) > 0.5:
            count += 1
            self.cluster_centers = trainX[np.random.choice(N, size=self.K, replace=False)]


        # Create variables for each item i and cluster j
        y = LpVariable.dicts('item_cluster', ((i, j) for i in range(N) for j in range(self.K)), lowBound = 0,upBound=1, cat='Integer')

        # Create an ILP problem
        prob = LpProblem("ClusteringProblem", LpMinimize)

        # Objective function
        prob += lpSum( np.linalg.norm(trainX[i,:]- self.cluster_centers[j,:]) *y[(i,j)] for j in range(self.K) for i in range(N))
        
        # Constraints
        for i in range(N):
            prob += lpSum(y[(i,j)] for j in range(self.K)) == 1

        # Solve LP
        prob.solve(PULP_CBC_CMD(msg=0))


        # Helper function that returns the new cluster centers by finding the mean
        def get_cluster_center():
            output = []
            for j in range(self.K):
                count = 0
                center = np.zeros(M)
                for i in range(N):
                    if value(y[(i,j)]) == 1:
                        center = center + trainX[i, :]
                        count += 1
                output.append(center / count)
            return np.array(output)
        
        # Helper function that returns the maximum l2 difference between 2 lists of vectors
        def max_l2_diff(list1, list2):
            maximum = 0
            for i in range(len(list1)):
                x = np.linalg.norm(list1[i] - list2[i])
                if maximum < x:
                    maximum = x
            return maximum


        new_centers = get_cluster_center()
        convergence_epsilon = 0.000001

        while max_l2_diff(new_centers,self.cluster_centers) > convergence_epsilon:
            self.cluster_centers = new_centers
            prob.objective = lpSum(np.linalg.norm(trainX[i,:] - self.cluster_centers[j,:]) *y[(i,j)] for j in range(self.K) for i in range(N))
            prob.solve(PULP_CBC_CMD(msg=0))
            new_centers = get_cluster_center()
        
        # Update and return the cluster labels of the training data (trainX)
        self.labels = np.array([[value(y[i,j]) for j in range(self.K)] for i in range(N)])
        self.labels = np.argmax(np.array([[value(y[i,j]) for j in range(self.K)] for i in range(N)]), axis=1)

        return self.labels
    
    
    def infer_cluster(self, testX):
        similarity = np.zeros((testX.shape[0], self.K))
        for i, center in enumerate(self.cluster_centers):
            for j in range(testX.shape[0]):
                similarity[j, i] = self.cosine_similarity(testX[j,:], center)
        return np.argmax(similarity, axis=1)
    

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(self, cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label

    def cosine_similarity(self, a,b):
        return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

    def select(self, trainX):
        N = trainX.shape[0]
        L = int(N * self.ratio)
        K = 3
        clustering = MyClustering(K)
        clustering.train(trainX)


        # Define variables y_i = {0,1}, 1 if want the label, 0 if not
        y = LpVariable.dicts('item_cluster', (i for i in range(N)), lowBound = 0,upBound=1, cat='Integer')

        prob = LpProblem("LabelSelection", LpMaximize)

        # Objective function
        prob += lpSum(self.cosine_similarity(trainX[i,:], clustering.cluster_centers[j]) * y[j] for i in range(N) for j in range(K))

        # Constraints
        prob += lpSum(y[i] for i in range(N)) == L

        prob.solve(PULP_CBC_CMD(msg=0))

        output = np.zeros(N)
        for i in range(N):
            output[i] = value(y[i])
        # Return an index list that specifies which data points to label
        return np.where(output == 1)[0]

    