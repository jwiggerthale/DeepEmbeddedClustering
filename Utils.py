'''
Implementation of different tools for DEC
Description see below
'''


import numpy as np
import torch
from typing import Optional
from Modules import LinearSumAssignment


'''
Function which calculates accuracy of clustering using linear sum assignment
Call with: 
	y_true: list of true clusters, points belong to
	y_predicted: list of predicted cluster centers
	cluster_number: number of clusters (Optionan[int])
		--> Can be calcukated from input if None
Returns: 
	Dictionary which allows reassignment of clusters
	Accuracy of clustering
'''
def cluster_accuracy(y_true, 
		     y_predicted, 
		     cluster_number: Optional[int] = None):
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assumption: labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    acc = LinearSumAssignment((count_matrix.max() - count_matrix).reshape(-1))
    accuracy = acc / y_predicted.size
    return accuracy 

'''
Calculate target distribution for KL divergence loss 
--> See 3.1.3 Equation 3 of Xie/Girshick/Farhadi
Call with batch 
Returns target distribution
'''
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / torch.sum(batch, 0) #torch.sum gives sum of every column of tensor (shape = num_cols)
    return (weight.t() / torch.sum(weight, 1)).t()



        
