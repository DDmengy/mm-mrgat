# MM-MRGAT

MM-MRGAT uses relation-specific graph attention and masked edge reconstruction to score candidate circRNA–disease associations. The archived implementation accepts sequence features for circRNA nodes and semantic features for disease nodes.

## Files

| Path | Contents |
| --- | --- |
| `src/model.py` | Heterogeneous graph encoder, attribute fusion, and pair-scoring decoder |
| `src/train.py` | Masked-edge training and evaluation on a supplied DGL graph |
| `data/README.md` | Benchmark provenance, source links, and expected inputs |
| `requirements.txt` | Candidate CPU environment for this source snapshot |

## Running the archived implementation

Use Python 3.8 with compatible PyTorch and DGL builds. The package versions in `requirements.txt` are a CPU starting point based on the archived environment notes. For GPU execution, select mutually compatible PyTorch and DGL builds for the system's CUDA version before installing the remaining dependencies. The exact environment that generated the manuscript tables has yet to be established.

With the environment and required graph in place, run from the repository root:

```sh
python src/train.py --help
python src/train.py --fivefold
```

The default graph path is `data/heterogeneous_graphs/heterographs.bin`. A different compatible graph can be selected with `--graph_path`. The default feature inputs are `data/circ_seq_embedding.npy` and `data/dis_sem_embedding.npy`. Pass `--circ_seq_path` and `--dis_sem_path` to select different files. The script reports whether each input is loaded and leaves the corresponding feature branch unused when its file is absent. Evaluation arrays and model checkpoints from a run are written to the local `results/` directory.

This source snapshot preserves the training and scoring procedure in the local archive. The changes for distribution are limited to project-relative paths, input messages, and a local seeding routine. Reproducing the manuscript's full multimodal results requires the exact graph, aligned feature arrays, and the original experiment settings. The full cold-start, baseline, and paired statistical comparisons require additional scripts or records.

The benchmark files distributed with circ2DGNN are credited in [the data notes](data/README.md).
