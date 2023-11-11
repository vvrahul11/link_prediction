## Disease-Gene link prediction using Pytorch Geometric

Reproduced from [Disease-Gene Interactions with Graph Neural Networks and Graph Autoencoders](https://medium.com/stanford-cs224w/disease-gene-interactions-with-graph-neural-networks-and-graph-autoencoders-bfae2c28f5ec) 

Paper: [Towards Probabilistic Generative Models Harnessing Graph Neural Networks
for Disease-Gene Prediction](https://arxiv.org/pdf/1907.05628.pdf)


### Create conda env
```bash
conda env create -f environment.yml
conda activate gene_disease
```

### Run VGAE and make predictions
```bash
python src\train_vgae.py
```

### Further reading
1. https://medium.com/@sunil7545/variational-autoencoders-ce7fe921cce7
2. https://medium.com/@sunil7545/kl-divergence-js-divergence-and-wasserstein-metric-in-deep-learning-995560752a53
3. [geneDRAGNN: Gene Disease Prioritization using Graph Neural Networks](https://ieeexplore.ieee.org/document/9863043)
4. linear algebra (https://datahacker.rs/dot-product-inner-product), (https://datahacker.rs/category/linear-algebra/page/3/)
