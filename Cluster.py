'''
This code implements two classes: 
	ClusterAssignment
	DEC
--> Description see below
'''


import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional



'''
ClusterAssignment
Soft assignment according to 3.1.1. in Xie/Girshick/Farhadi
Use Student t-distribution to measure similarity between feature vector and centroid
Call with: 
	cluster_number: Number of cluster you use (int)
	embedding_dimension: Number of features in latent feature space (int)
	alpha: degree of freedom in t-distribution (float)
	cluster_centers: cluster centers to initialize distribution (Optional[torch.Tensor])
Forward pass: 
	Call with batch of feature vectors (torch.Tensor)
	Soft assignment for a batch of feature vectors
	Returns batch of assignments for each cluster
	Low value is better
'''
class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)# Sum of every row for predictios - cluster_center[i]
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)



'''
Module whicih combines all the parts to the DEC model acording to Xie/Girshick/Farhadi
	--> Autoencder + ClusterAssignment
Call with: 
	cluster_number: number of clusters (int)
	hidden_dimension: number of features in bottleneck of autoencoder (int)
	encoder: autoencoder (torch.nn.Module)
	alpha: degree of freedom in t-distribution (float)
Forward pass: 
	Calls forward pass of encoder on batch 
        Computes cluster assignmentfor results
	Call with: 
		batch: Tensor of samples from dataset (torch.Tensor)
	Forward pass returns batch of assignments for each cluster
'''
class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
    ):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.assignment(self.encoder(batch)[0])
    
