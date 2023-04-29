## Gene-Disease link prediction using Pytorch Geometric

Reproduced from [Disease-Gene Interactions with Graph Neural Networks and Graph Autoencoders](https://biolactosil.medium.com/how-to-load-and-visualize-time-series-data-5d72cbf47901) 

Paper: [Towards Probabilistic Generative Models Harnessing Graph Neural Networks
for Disease-Gene Prediction](https://arxiv.org/pdf/1907.05628.pdf)


### Create conda env
conda env create -f environment.yml
conda activate gene_disease


### Run VGAE and make predictions
python src\train_vgae.py

### Further reading
1. https://medium.com/@sunil7545/variational-autoencoders-ce7fe921cce7
2. https://medium.com/@sunil7545/kl-divergence-js-divergence-and-wasserstein-metric-in-deep-learning-995560752a53