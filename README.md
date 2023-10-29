# Multi-Domain-Active-Learning
This repo is from the paper: Katsarou, Katerina, Roxana Jeney, and Kostas Stefanidis. "MUTUAL: Multi-Domain Sentiment Classification via Uncertainty Sampling." Proceedings of the 38th ACM/SIGAPP Symposium on Applied Computing. 2023.

MUTUAL is a technique designed to learn both general and domain-specific sentence embeddings, which are further enriched by being context-sensitive through the incorporation of an attention mechanism. In this research, we introduce a model constructed from a stacked BiLSTM Autoencoder with an embedded attention layer to craft these specialized sentence embeddings. Using the Jensen-Shannon (JS) distance metric, we identify and select the general sentence embeddings from the four domains most closely related to our target domain. Subsequently, these selected embeddings, along with domain-specific ones, are combined and passed through a dense neural layer during training. Complementing this, we also present an active learning strategy. It commences by leveraging the elliptic envelope method to weed out outliers from an unlabeled dataset, which MUTUAL then classifies. From this classification, the data points with the highest uncertainty are chosen for labeling, grounded on the least confidence measure.

Multi Domain Sentiment Classification using Active Learning 
## Getting Started
For running the framework, this work recommends creating a new virtual environment which uses the python version 3.7.7.
Afterwards, install the packages in the requirements.txt of the requirements_files directory to get started. Using anaconda, the commands look like this:
```bash
conda create -n myenv python=3.7.7
conda activate myenv
conda install --file requirements_files/requirements.txt
```
The requirements directory also contains a requirements_ae.txt which can come in handy when training the autoencoder separately (e.g. on a GPU-dedicated server). The requirements_bert.txt in the directory can be used if the BERT experiment of this research is desired to be reproduced. 

If the entire framework is desired to be executed, the data set [1] needs to be saved in the directory "data/uncleaned_data/" and the following files need to be executed in this specific order: preprocessing.ipynb, autoencoder.ipynb and data_selection_splitting.ipynb. The resulting sentence embeddings can then be fed into the classifier. If the classifier without prior active learning is desired to be executed, the file classifier_without_al.ipynb is run. For executing the classifier with active learning, the file classifier_with_al.ipynb is run.


**Autoencoder for general and domain-specific sentence embeddings**  

Next, we will generate the general and sentence embeddings by using BiLSTM-based autoencoder with Self-Attention. We will run the script: autoencoder_gen_spec_embeddings.py, with the embedding type GENERAL to calculate the general embeddings:

python autoencoder_gen_spec_embeddings.py --embedding-type GENERAL

Then the same script with SPECIFIC as embedding type to calculate the domain-specific embeddings of each of the 16 domains:  

python autoencoder_gen_spec_embeddings.py --embedding-type SPECIFIC

**Classification with AL**  
Then we will use the general and specific embeddings that where created previous as inputs of the classifier. The domain specific embeddings are augmented by multiplying them three times and then we select the general sentence embeddings from four most similar domains by using the Jensen-Shanon distance. the function jensen_shannon_with_AL from jensen_shannon_augmentation_AL to calculate and return the general embeddings the four most similar domains. We use the proposed Active learning algorithm with Uncertainty Sampling and Elliptic Envelope with our proposed classifier to query the most informative instances. After trials we have concluded that by using only 38% of the initial labeled data we achieve highly-accurate results. Then, We run the script: 

python al-module.py SPEC_INDEX_VALUE PAR0_VALUE

where SPEC_INDEX_VALUE is for the domain number (it is between 0 and 15) and PARO_VALUE is the number of labeled instances (1600). So we will run 

python python al-module.py 3 1600   

for domain 3 and 1600 labeled instances for the active learning. However, if we want to get the results for all domain we do not run python python al-module.py 3 1600 but we only run:

Test_Case.py

to get the accuracy results for the 16 domains.


**Classification without AL**  

In a similar way as above, we train our classifier withhout the presence of Active Learning. Again, we find the general embeddings of the four most similar domains to the target domain by calling the function jensen_shannon from jensen_shannon_augmentation when we run the script:

python classifier_without_al.py 

Then we run 

