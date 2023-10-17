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
