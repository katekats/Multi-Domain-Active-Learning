# Multi-Domain-Active-Learning
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
Then we will use the general and specific embeddings that where created previous as inputs of the classifier. The domain specific embeddings are augmented by multiplying them three times and then we select the general sentence embeddings from four most similar domains by using the Jensen-Shanon disrance. the function jensen_shannon_with_AL from jensen_shannon_augmentation_AL to calculate and return the general embeddings the four most similar domains. Then, We run the script: 

python al-module.py

that uses the Active learning alforithm with Uncertainty Sampling and Isolation Forests with our proposed classifier to query the most informative instances. After trials we have concluded that by using only 38% of the initial labeled data we achieve highly-accurate results. Then we run the script:

Test_Case.py

to get the accuracy results for the 16 domains.


**Classification without AL**  
python classifier_without_al.py --spec-index 5 



