visualize_tsne_embeddings(gae_model,
                          train_dataset,
                          'Untrained GAE: train set embeddings t-SNE',
                          labeled=True,
                          labels=[40, 190, 230, 1830, 260, 110, 280, 1967]
)

visualize_pca_embeddings(gae_model,
                         train_dataset,
                         'Untrained GAE: train set embeddings PCA',
                         labeled=True,
                         labels=[40, 190, 230, 1830, 260, 110, 280, 1967]
)

visualize_tsne_embeddings(gae_model,
                          train_dataset,
                          title='Trained GAE: train set embeddings',
                          perplexity=5,
                          labeled=True,
                          labels=[40, 190, 230, 1830, 260, 110, 280, 1967]
)

visualize_pca_embeddings(gae_model,
                         train_dataset,
                         title='Trained GAE: train set embeddings',
                         labeled=True,
                         labels=[40, 190, 230, 1830, 260, 110, 280, 1967]
)