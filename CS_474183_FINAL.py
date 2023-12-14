# Importing packages
import json
import pandas as pd
import re
import numpy as np
from numpy.linalg import norm
from random import shuffle
import sys
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import math
from sklearn.model_selection import train_test_split

#%%
# Loading the json file
json = json.load(open("TVs-all-merged.json"))

# Splitting the data from the json file and turning into a DataFrame
temp = []
val = list(json.values())
for i in val:
    for j in i:
        temp.append(j)
df = pd.DataFrame(temp)  

#%% 
def dataPrep(df):
    titles = df['title']
    modelid = df['modelID']
    shop = df['shop']
    
    # Replace all instances related to 'inch', 'hertz' and 'diagonal' with the same word
    for i in range(len(titles)): 
        titles[i] = titles[i].lower()
        titles[i] = titles[i].replace('"','inch')
        titles[i] = titles[i].replace('\"','inch')
        titles[i] = titles[i].replace('inches','inch')
        titles[i] = titles[i].replace(' inch','inch')
        
        titles[i] = titles[i].replace('hertz','hz')
        titles[i] = titles[i].replace(' hz','hz')
    
        titles[i] = titles[i].replace('diagonal','diag')
        titles[i] = titles[i].replace('diagonal size','diag')
        titles[i] = titles[i].replace('diag.','diag') 
        titles[i] = titles[i].replace('diagonally','diag')
    
    # Clean up some special characters/instances    
    titles_clean = []    
    for i in range(len(titles)):
        alt = re.sub(r'[^\w\s]','',titles[i])
        alt = alt.split()
        titles_clean.append(alt)
    
    # Creating the vocabulary
    uniqueTitleWords = []
    for i in range(len(titles_clean)):
        for j in range(len(titles_clean[i])):
            if titles_clean[i][j] not in uniqueTitleWords:
                uniqueTitleWords.append(titles_clean[i][j])
    
    # Removing the words relating to the specific website out of the title
    uniqueTitleWords.remove('neweggcom')            
    uniqueTitleWords.remove('best')
    uniqueTitleWords.remove('buy')
    
    # Cleaning the modelIDs
    modelID_clean = []
    for i in range(len(modelid)):
        temp = re.sub(r'[^\w\s]','',modelid[i])
        temp = temp.upper()
        modelID_clean.append(temp)
    
    # Creating a vector for the brand preselection criterion
    brands_all = ['Panasonic', 'Samsung', 'Sharp', 'Coby', 'LG', 'Sony',
            'Vizio', 'Dynex', 'Toshiba', 'HP', 'Supersonic', 'Elo',
            'Proscan', 'Westinghouse', 'SunBriteTV', 'Insignia', 'Haier',
            'Pyle', 'RCA', 'Hisense', 'Hannspree', 'ViewSonic', 'TCL',
            'Contec', 'NEC', 'Naxa', 'Elite', 'Venturer', 'Philips',
            'Open Box', 'Seiki', 'GPX', 'Magnavox', 'Hello Kitty', 'Naxa', 'Sanyo',
            'Sansui', 'Avue', 'JVC', 'Optoma', 'Sceptre', 'Mitsubishi', 'CurtisYoung', 'Compaq',
            'UpStar', 'Azend', 'Contex', 'Affinity', 'Hiteker', 'Epson', 'Viore', 'VIZIO',
            'SIGMAC','Craig','ProScan', 'Apple']
    
    for i in range(len(brands_all)):
        brands_all[i] = brands_all[i].lower()
        
    brands = np.zeros((len(titles_clean)))
    for i in range(len(titles_clean)):
        for j in range(len(brands_all)):
            if brands_all[j] in titles_clean[i]:
                brands[i] = j + 1
                break
        
    # Creating a vector for the screen size (inches) preselection criterion
    small_inch = 5
    large_inch = 90
    
    inches_all = []   
    for j in range(small_inch, large_inch+1):
        temp_inch = "{}inch".format(j)
        inches_all.append(temp_inch)
    
    inches = np.zeros((len(titles_clean)))
    for i in range(len(titles_clean)):
        for j in range(len(inches_all)):
            if inches_all[j] in titles_clean[i]:
                inches[i] = j + small_inch # small_inch addition so that it corresponds with the actual inch size
                break
    
    return uniqueTitleWords, titles_clean, shop, modelID_clean, brands, inches

#%%
def minHash(uniqueTitleWords, titles_clean, r, b):
    # Constructing the sparse matrix
    binary_matrix = np.zeros((len(uniqueTitleWords),len(titles_clean)))
    for k in range(len(uniqueTitleWords)):
        for l in range(len(titles_clean)):
            if uniqueTitleWords[k] in titles_clean[l]:
                binary_matrix[k][l] = 1
               
    # Create a vector with numbers 1 to h in random order
    def random_vec(h):
        hash_ex = list(range(1, h+1))
        shuffle(hash_ex)
        return hash_ex
    
    hashfunctions = r*b
    rand_vec1 = random_vec(hashfunctions)
    rand_vec2 = random_vec(hashfunctions)
    p = 1999 # Some prime number greater than the length of the vocabulary
    
    # MinHashing for the signature matrix
    signature_matrix = np.ones((hashfunctions,len(binary_matrix[0]))) * sys.maxsize
    for w in range(len(binary_matrix)):
        hash_value = []
        for i in range(hashfunctions):
            temp_hash = (rand_vec1[i] * w + rand_vec2[i]) % p
            hash_value.append(temp_hash)
            
        for c in range(len(binary_matrix[0])):
            if binary_matrix[w][c] == 1:
                for j in range(hashfunctions):
                    if hash_value[j] < signature_matrix[j][c]:
                        signature_matrix[j][c] = hash_value[j]

    return binary_matrix, signature_matrix           

#%%
def candidateBands(signature_matrix, r, b):
    # Convert vector of integers to string    
    def int_to_str(signature):
        newstring = [str(sig) for sig in signature]
        appended_string = ''.join(newstring)
        return appended_string
    
    # Split signature matrix in b bands of r rows
    bands = np.split(signature_matrix, b)
    bands_upd = []
    for i in range(0,b):
        bands_temp = []
        for j in range(len(bands[i][0])):
            sig_temp = bands[i][:,j]
            string_sig = int_to_str(sig_temp)
            string_sig = string_sig.replace('.0','')
            bands_temp.append(string_sig)
        bands_upd.append(bands_temp)
    
    # Construct binary candidate matrix
    cl = len(bands_upd[0])
    candidates = np.zeros((cl,cl)) # 1624x1624
    
    for i in range(cl):
        for j in range(i+1, cl):
            for k in range(0, len(bands_upd)):
                if bands_upd[k][i] == bands_upd[k][j]:
                    candidates[i][j] = 1
                    candidates[j][i] = 1
                    break                

    return candidates
#%%
def distanceMatrix(titles_clean, binary_matrix, candidates, shop, brands, inches):         
    # Jaccard distance measure
    def jaccard(A,B):
        union = set(A) | set(B)
        intersect = set(A) & set(B)
        Jacc_sim = len(intersect)/len(union)
        Jacc_dist = 1 - Jacc_sim
        return Jacc_dist
    
    # Cosine distance measure 
    def cosine(A,B):
        Cosine_sim = (np.dot(A,B))/(norm(A)*norm(B))
        Cosine_dist = 1 - Cosine_sim
        return Cosine_dist
    
    # Construct distance matrix using preselection criteria, set other distances to infinity
    # Removal/addition of "inches[i]==inches[j]" criterion obtains results for the two different models
    distance_matrix_Jacc = np.ones((len(titles_clean), len(titles_clean))) 
    distance_matrix_Cosine = np.ones((len(titles_clean), len(titles_clean)))     
    for i in range(len(candidates)):
        for j in range(i, len(candidates)):
            if candidates[i][j] == 1 and shop[i] != shop[j] and brands[i] == brands[j] and inches[i] == inches[j]:
                temp_Jacc = jaccard(titles_clean[i], titles_clean[j])
                distance_matrix_Jacc[i][j] = temp_Jacc
                distance_matrix_Jacc[j][i] = temp_Jacc
                
                temp_Cosine = cosine(binary_matrix[:,i], binary_matrix[:,j])
                distance_matrix_Cosine[i][j] = temp_Cosine
                distance_matrix_Cosine[j][i] = temp_Cosine
            else:
                distance_matrix_Jacc[i][j] = sys.maxsize
                distance_matrix_Jacc[j][i] = sys.maxsize
                
                distance_matrix_Cosine[i][j] = sys.maxsize
                distance_matrix_Cosine[j][i] = sys.maxsize

    return distance_matrix_Jacc, distance_matrix_Cosine

#%%
def clusterPairing(distance_matrix_Jacc, distance_matrix_Cosine, modelID_clean, t):
    # Agglomerative hierarchical clustering with complete linkage and distance threshold t
    def clustering(dis_matrix, t):
        agglom_cluster = AgglomerativeClustering(n_clusters=None, metric='precomputed', 
                                                                     linkage='complete', distance_threshold=t)
        ac_fit = agglom_cluster.fit(dis_matrix)
        labels = ac_fit.labels_
        n_cl = ac_fit.n_clusters_
        return labels, n_cl
    
    labels_Jacc, n_cl_Jacc = clustering(distance_matrix_Jacc, t)
    labels_Cosine, n_cl_Cosine = clustering(distance_matrix_Cosine, t)
    
    # Extract all predicted duplicate pairs
    def pairExtract(labels, n_cl):
        pairs = []
        for i in range(0, n_cl):
            temp_pairs = []
            for j in range(len(labels)):
                if labels[j] == i:
                    temp_pairs.append(j)
            comb = list(combinations(temp_pairs,2))
            pairs.append(comb)
        
        pairs_flat = [item for sublist in pairs for item in sublist]  
        
        return pairs_flat
    
    pairs_Jacc = pairExtract(labels_Jacc, n_cl_Jacc)
    pairs_Cosine = pairExtract(labels_Cosine, n_cl_Cosine)
    
    # Extract all real pairs from the actual ModelID's
    pairs_ID_temp = []
    for i in range(len(modelID_clean)):
        temp_id = []
        for j in range(len(modelID_clean)):
            if modelID_clean[i] == modelID_clean[j]:
                temp_id.append(j)
        comb_id = list(combinations(temp_id,2))   
        pairs_ID_temp.append(comb_id)  
        
    pairs_ID = [item for sublist in pairs_ID_temp for item in sublist]
    pairs_ID = list(set(pairs_ID))

    return pairs_Jacc, pairs_Cosine, pairs_ID

#%%
def performance(pairs_Jacc, pairs_Cosine, pairs_ID, distance_matrix_Jacc, distance_matrix_Cosine):
    # Calculate confusion matrix measures
    def confusionMatrix(pairs, pairsID):
        TP_list = []
        FP_list = []
        for i in range(len(pairs)):
            if pairs[i] in pairsID:
                TP_list.append(pairs[i])
            else:
                FP_list.append(pairs[i])
        
        TP = len(TP_list)
        FP = len(FP_list)
        FN = len(pairsID) - TP
                
        return TP, FP, FN
    
    TP_Jacc, FP_Jacc, FN_Jacc = confusionMatrix(pairs_Jacc, pairs_ID)
    TP_Cosine, FP_Cosine, FN_Cosine = confusionMatrix(pairs_Cosine, pairs_ID)
    
    # Calculate final performance measures
    def performanceMeasures(TP, FP, FN, dis_matrix):
        #Pair Quality   = number of duplicates found    /   number of comparisons made
        cm = np.count_nonzero(dis_matrix < 1) / 2 
        PQ = TP / cm
            
        #Pair Completeness  = number of duplicates found    /   total number of duplicates
        PC = TP / len(pairs_ID)
        
        #Precision and recall
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        
        #F1-measure
        F1 = 2 / ((1/precision) + (1/recall))
        
        #F1*-measure = harmonic mean between PC and PQ
        F1star = 2 / ((1/PQ) + (1/PC))
        
        # Fractions of comparisons = number of comparisons made / total number of possible comparisons
        cm_total = (len(dis_matrix) * (len(dis_matrix)-1)) / 2
        FoC = cm / cm_total
        
        return PQ, PC, F1, F1star, FoC
    
    perf_Jacc = performanceMeasures(TP_Jacc, FP_Jacc, FN_Jacc, distance_matrix_Jacc)
    perf_Cosine = performanceMeasures(TP_Cosine, FP_Cosine, FN_Cosine, distance_matrix_Cosine)

    return perf_Jacc, perf_Cosine
#%%
def totalEvaluation(df, r, b, t):    
    # Cleaning and preparing the data
    uniqueTitleWords, titles_clean, shop, modelID_clean, brands, inches = dataPrep(df)
    
    # Creating the binary and signature matrix
    binary_matrix, signature_matrix = minHash(uniqueTitleWords, titles_clean, r, b)
    
    # Constructing the candidate matrix
    candidates = candidateBands(signature_matrix, r, b)
    
    # Calculating the distance matrices for two distance measures: Jaccard and Cosine
    distance_matrix_Jacc, distance_matrix_Cosine = distanceMatrix(titles_clean, binary_matrix, candidates, 
                                                                  shop, brands, inches)
    
    # Perform Agglomerative Clustering and extracting the predicted duplicate pairs and the real pairs
    pairs_Jacc, pairs_Cosine, pairs_ID = clusterPairing(distance_matrix_Jacc, distance_matrix_Cosine,
                                                        modelID_clean, t)
    
    # Calculate and return all performance measurs
    perf_Jacc, perf_Cosine = performance(pairs_Jacc, pairs_Cosine, pairs_ID, distance_matrix_Jacc,
                                         distance_matrix_Cosine)
    
    return perf_Jacc, perf_Cosine
#%%
def thresCalc(r, b):
    # Formula for calculating the distance measure threshold
    t = (1/b)**(1/r)
    
    return 1-t

# Calculate threshold for different combinations of rows and bands
thresholds = np.zeros(10)
for i in range(len(thresholds)):
    x = i+1
    y = math.floor(800/(x)) 
    thresholds[i] = thresCalc(x,y)

#%%
def bootstrap(df, r, b, t, num_bootstraps):
    avgPerf_Jacc = np.zeros(5)
    avgPerf_Cosine = np.zeros(5)
    
    # Bootstrapping procedure
    for i in range(num_bootstraps):
        dfBoot = df.sample(n=len(df), replace=True)
        boot = train_test_split(dfBoot, test_size=0.33)[0].reset_index(drop=True)
        
        perf_Jacc_boot, perf_Cosine_boot = totalEvaluation(boot, r, b, t)
            
        avgPerf_Jacc[0] = avgPerf_Jacc[0] + perf_Jacc_boot[0]
        avgPerf_Jacc[1] = avgPerf_Jacc[1] + perf_Jacc_boot[1]
        avgPerf_Jacc[2] = avgPerf_Jacc[2] + perf_Jacc_boot[2]
        avgPerf_Jacc[3] = avgPerf_Jacc[3] + perf_Jacc_boot[3]
        avgPerf_Jacc[4] = avgPerf_Jacc[4] + perf_Jacc_boot[4]
        
        avgPerf_Cosine[0] = avgPerf_Cosine[0] + perf_Cosine_boot[0]
        avgPerf_Cosine[1] = avgPerf_Cosine[1] + perf_Cosine_boot[1]
        avgPerf_Cosine[2] = avgPerf_Cosine[2] + perf_Cosine_boot[2]
        avgPerf_Cosine[3] = avgPerf_Cosine[3] + perf_Cosine_boot[3]
        avgPerf_Cosine[4] = avgPerf_Cosine[4] + perf_Cosine_boot[4]
    
    # Calculate the average across all bootstraps
    avgPerf_Jacc = avgPerf_Jacc / num_bootstraps
    avgPerf_Cosine = avgPerf_Cosine / num_bootstraps
    
    return avgPerf_Jacc, avgPerf_Cosine
#%%
## Bootstrapping for the model WITH the 'inches' preselection criterion
# Bootstrapping function calling for the different parameters
num_bootstraps = 5
B1 = bootstrap(df, 1, 800, thresholds[0], num_bootstraps)
B2 = bootstrap(df, 2, 400, thresholds[1], num_bootstraps)
B3 = bootstrap(df, 3, 266, thresholds[2], num_bootstraps) 
B4 = bootstrap(df, 4, 200, thresholds[3], num_bootstraps)
B5 = bootstrap(df, 5, 160, thresholds[4], num_bootstraps)
B6 = bootstrap(df, 6, 133, thresholds[5], num_bootstraps)
B7 = bootstrap(df, 7, 114, thresholds[6], num_bootstraps)
B8 = bootstrap(df, 8, 100, thresholds[7], num_bootstraps)
B9 = bootstrap(df, 9, 88, thresholds[8], num_bootstraps)
#B10 = bootstrap(df, 10, 80, thresholds[9], num_bootstraps) | Not included due to unfeasibility

# Creating a DataFrame, for each distance measure, with the all bootstrap results for each performance measure 
df_perf_Jacc = pd.DataFrame((B1[0],B2[0],B3[0],B4[0],B5[0],B6[0],B7[0],B8[0],B9[0]),
                            index = ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            columns=['PQ', 'PC', 'F1', 'F1star', 'FoC'])
df_perf_Cosine = pd.DataFrame((B1[1],B2[1],B3[1],B4[1],B5[1],B6[1],B7[1],B8[1],B9[1]),
                            index = ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            columns=['PQ', 'PC', 'F1', 'F1star', 'FoC'])

#%%
## Bootstrapping for the model WITHOUT the 'inches' preselection criterion
# Bootstrapping function calling for the different parameters
num_bootstraps = 5
B1_old = bootstrap(df, 1, 800, thresholds[0], num_bootstraps)
B2_old = bootstrap(df, 2, 400, thresholds[1], num_bootstraps)
B3_old = bootstrap(df, 3, 266, thresholds[2], num_bootstraps) 
B4_old = bootstrap(df, 4, 200, thresholds[3], num_bootstraps)
B5_old = bootstrap(df, 5, 160, thresholds[4], num_bootstraps)
B6_old = bootstrap(df, 6, 133, thresholds[5], num_bootstraps)
B7_old = bootstrap(df, 7, 114, thresholds[6], num_bootstraps)
B8_old = bootstrap(df, 8, 100, thresholds[7], num_bootstraps)
B9_old = bootstrap(df, 9, 88, thresholds[8], num_bootstraps)
#B10 = bootstrap(df, 10, 80, thresholds[9], num_bootstraps) | Not included due to unfeasibility

# Creating a DataFrame, for each distance measure, with the all bootstrap results for each performance measure 
df_perf_Jacc_old = pd.DataFrame((B1_old[0],B2_old[0],B3_old[0],B4_old[0],B5_old[0],B6_old[0],B7_old[0],
                             B8_old[0],B9_old[0]),
                            index = ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            columns=['PQ', 'PC', 'F1', 'F1star', 'FoC'])
df_perf_Cosine_old = pd.DataFrame((B1_old[1],B2_old[1],B3_old[1],B4_old[1],B5_old[1],B6_old[1],B7_old[1],
                               B8_old[1],B9_old[1]),
                            index = ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
                            columns=['PQ', 'PC', 'F1', 'F1star', 'FoC'])
