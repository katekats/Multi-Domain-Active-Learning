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

The order we should run our scripts is the following:
* data_loader.ipynb
* preprocessing-withoutTFIDF.ipynb
* python autoencoder_gen_spec_embeddings.py --embedding-type GENERAL
* python autoencoder_gen_spec_embeddings.py --embedding-type SPECIFIC
* python data_selection_splitting.py
* python jensen jensen_shannon_augmentation.py SPEC_INDEX_VALUE
* python hyperparameter_tuning.py
* python classifier_without_al.py SPEC_INDEX_VALUE (if we want the results for a single domain) __OR__ python Test_Case_without_AL.py (if we want to get the results for all domains)
* python al-module.py SPEC_INDEX_VALUE PAR0_VALUE (if we want the results for a single domain) __OR__ python Test_Case.py (if we want to get the results for all domains)
**Loading the data**
We use the __data_loader.ipynb__ to load and clean the data.
**Preprocessing the Data**
In the script __preprocessing-withoutTFIDF.ipynb__ we proceed with preprocessing the data by giving a specific number to each domain:
{'MR': 0,
 'apparel': 1,
 'baby': 2,
 'books': 3,
 'camera_photo': 4,
 'dvd': 5,
 'electronics': 6,
 'health_personal_care': 7,
 'imdb': 8,
 'kitchen_housewares': 9,
 'magazines': 10,
 'music': 11,
 'software': 12,
 'sports_outdoors': 13,
 'toys_games': 14,
 'video': 15} 

 Based on the histogram, we decide on the sequence length and vocabularry size and we use pre-trained FastText embeddings for vectorizing the reviews to be suitable inputs for the autoencoder.

**Autoencoder for general and domain-specific sentence embeddings**  

Next, we will generate the general and sentence embeddings by using BiLSTM-based autoencoder with Self-Attention. We will run the script: autoencoder_gen_spec_embeddings.py, with the embedding type GENERAL to calculate the general embeddings:

__python autoencoder_gen_spec_embeddings.py --embedding-type GENERAL__

Then the same script with SPECIFIC as embedding type to calculate the domain-specific embeddings of each of the 16 domains:  

__python autoencoder_gen_spec_embeddings.py --embedding-type SPECIFIC__

**Data Splitting**

We proceed with augmenting the domain specific embedding data by multiplying them three times and sort them with the general embeddings based on the same sentiment (positive or negative). Then, we split the data into 70-10-20 for train-validation-test sets (as in the state-of-the-art).

**Classification without AL**  
Then we will use the general and specific embeddings that where created previous as inputs of the classifier. The domain specific embeddings are augmented by multiplying them three times and then we select the general sentence embeddings from four most similar domains by using the Jensen-Shanon distance. We run the script:
__python jensen jensen_shannon_augmentation.py SPEC_INDEX_VALUE__

where __SPEC_INDEX_VALUE__ is the domain number (between 0 and 15). Then since we have the two input for the classifier we proceed with hyperparameter tuning:
__python hyperparameter_tuning.py__

Then we proceed with the classifier without the presence of active learning. We call the function __jensen_shannon__ from __jensen_shannon_augmentation__ to calculate and return the general embeddings the four most similar domains. We run the script: 

__python classifier_without_al.py SPEC_INDEX_VALUE__

where SPEC_INDEX_VALUE is for the domain number (it is between 0 and 15). However, if we want to get the results for all domain we do not run python python al-module.py 3 1600 but we only run:

__python Test_Case_without_AL.py__


**Classification with AL**  
Then we will use the general and specific embeddings that where created previous as inputs of the classifier. The domain specific embeddings are augmented by multiplying them three times and then we select the general sentence embeddings from four most similar domains by using the Jensen-Shanon distance. the function __jensen_shannon_with_AL__ from __jensen_shannon_augmentation_AL__ to calculate and return the general embeddings the four most similar domains. Firstly  We use the proposed Active learning algorithm with Uncertainty Sampling and Elliptic Envelope with our proposed classifier to query the most informative instances. After trials we have concluded that by using only 38% of the initial labeled data we achieve highly-accurate results. Then, We run the script: 

__python al-module.py SPEC_INDEX_VALUE PAR0_VALUE__

where SPEC_INDEX_VALUE is for the domain number (it is between 0 and 15) and PARO_VALUE is the number of labeled instances (1600). So we will run 

__python al-module.py 3 1600__   

for domain 3 and 1600 labeled instances for the active learning. However, if we want to get the results for all domain we do not run python python al-module.py 3 1600 but we only run:

__python Test_Case.py__

to get the accuracy results for the 16 domains.





