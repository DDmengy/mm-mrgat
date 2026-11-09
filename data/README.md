# Benchmark data and model inputs

## Data provenance

The archived benchmark entity and relation workbooks, adjacency matrices numbered 01–13 and 16–17, and the no-similarity index map match files distributed in the [circ2DGNN repository at commit 664c6be](https://github.com/CEN-KL/circ2DGNN/tree/664c6be6392c948f861769bcb0081ed0dbd75c3a/data). Cite the [circ2DGNN article](https://doi.org/10.1109/TCBB.2024.3488281) when using those materials.

The manuscript identifies the following underlying database resources:

| Resource | Official access and citation |
| --- | --- |
| RNADisease v4.0 | [Download page and usage conditions](https://rnadisease.org/download), [database article](https://doi.org/10.1093/nar/gkac814) |
| RNAInter v4.0 | [Database article and official access links](https://doi.org/10.1093/nar/gkab997) |
| circBase | [Database](https://circbase.org/), [database article](https://doi.org/10.1261/rna.043687.113) |

This repository provides source links for upstream materials. Use the applicable terms of each provider when obtaining the files.

## Local input layout

| Path | Format and role |
| --- | --- |
| `data/heterogeneous_graphs/heterographs.bin` | DGL binary containing four graphs in this order: total, training, test, and negative test. The target canonical edge type is `('circ', 'circ_e_disease', 'disease')`. |
| `data/circ_seq_embedding.npy` | A two-dimensional NumPy array with one row per circRNA in graph node-ID order |
| `data/dis_sem_embedding.npy` | A two-dimensional NumPy array with one row per disease in graph node-ID order |

The local archive contains a preprocessed graph and sparse similarity matrices, while the two node-feature arrays expected by the training script are absent from the inspected copy. The similarity matrices represent pairwise similarities. The optional feature arguments expect node-level arrays aligned to the chosen graph. The authors confirmed that the archived `heterographs.bin` is the graph used for the manuscript's main experiments. The use and provenance of the two node-feature arrays in the reported runs remain to be verified.

## Processed graph availability

The [MM-MRGAT data repository](https://huggingface.co/datasets/liudded/mm-mrgat-data) is a private staging repository. The main-experiment graph remains in the authors' local archive while the terms for sharing its underlying data are verified.

| Archived file | Bytes | SHA-256 |
| --- | ---: | --- |
| `heterographs.bin` | 1,697,071,150 | `07bae91dcbe94d2ba2ff382ba0ace17493e826c58ba79581f242d6db9f04bbe3` |
