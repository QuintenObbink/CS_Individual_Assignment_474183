# CS_Individual_Assignment_474183
This GitHub directory contains the code to the algorithm that was used while making the individual assignment for the course Computer Science for Business Analytics. 
As a result the following paper was written: "Increasing the Efficiency of MSMP: Preselection Criteria Revised". 

The code contains the implementation of a scalable duplicate detection algorithm, using Locality Sensitive Hashing (LSH) and Agglomerative clustering.
The dataset used for this research contains product descriptions of 1624 TV's across four different webshops. The product descriptions contain information on the webshop, title of the product, the modelID and a set of key-value pairs. The modelID's are only used for evaluation of the algorithm.
To keep the code as clean as possible, a description of each notable function in the code will be provided in this README.

The code itself works sequentially, so each function requires input obtained through previous functions. If the code is not ran in the given order, errors can occur.

_dataPrep_: Function for cleaning the data, removing irregularities and creating the same product representation for each product description. It also performs the first step of the LSH algorithm, which is the construction of the vocabulary. The vectors for the eventual preselection criteria are also constructed here.

_minHash_: Function to perform the minhashing-step of the LSH algorithm. Constructs the sparse matrix, as well as the signature matrix. 

_candidateBands_: Function to perform the banding-step of the LSH algorithm. The signature matrix is split into the specified rows and bands, and candidates are extracted based on similar signatures in these bands.

_distanceMatrix_: Function to calculate the Jaccard- and Cosine-distance measures. Here the preselection criteria are used. The preselection criteria are added manually, so when you want to run the algorithm for only the shop and brand criteria, remove the 'inches[i]==inches[j]'-criterion from the if-statement in row 204.

_clusterPairing_: Function to perform the Agglomerative hierarchical clustering algorithm, based on the calculated distance matrices, with a specified distance threshold. Next to this, this function also extracts the predicted pairs from the clustering algorithm, as well as the real pairs from the modelID's obtained from the dataset.

_performance_: Function to calculate all performance measures needed for evaluation.

_totalEvaluation_: An encompassing function containing all functions above, to make the coding for the bootstrapping procedure more compact.

_bootstrap_: Function for the bootstrapping procedure. Its inputs are the dataset, desired row and band combination, the calculated distance threshold and the number of bootstraps. It returns the average across the performance measures for both distance measures.

The end of the code includes the actual calling of the bootstrap function for all combinations of parameters that were needed. 
This is done, such that when running the entire code, the results are all there in a comprehensive manner.
