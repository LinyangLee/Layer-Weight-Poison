# Layer-Weight-Poison

Code for *[Backdoor Attacks on Pre-trained Models by Layerwise Weight Poisoning](https://arxiv.org/abs/2108.13888)*

We were using an old version of transformers (2.9.0), therefore, for faster re-implementation, we only provide the key components for faster transfer to recent-versions of huggingface transformers.



Poisoned Data Generation:

We are using the generate_triggers.py to generate triggered data for each datasets.

The main experiment is to use the combined triggers.
For a certian task, there should be a clean trainingset, a poisoned trainingset, a clean valid set and a poisoned valid set in the data directory.

For ablation studies, generating single-token trigger dataset is also available.

Training and Testing: 

We provide a sample script of running the sst-2 dataset experiement, there should be pre-generated poisoned dataset in the data directory.

We control different experiment setting via a hyper param weight_poison including normal fine-tuning, badnet, restricted inner-product method, and our proposed laywerwise weight-poison method.

