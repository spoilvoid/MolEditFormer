# MolEditFormer 
This respository contains code about multi-modal Molecular Model used for Molecular Editing named MolEditFormer. The experiment is not done yet and is prepared to submit to NMI. ***Checkpoints, datasets and guidance about how to reproduce results will be released after acceptance. Coming soon……***

## Framework

MolEditFormer is pretrained in Contrastive Learning way to align embedding space between natural language and molecular description. In this procedure, decoders of each modality will be trained simultaneously. Pretrain Procedure will empower model to conduct modality translation.

<p align="center">
<img src="./figures/pretrain.png" alt="" align=center />
</p>

After getting decoder of joint embedding space, the branch of natural language is finetuned in molecular pairs with specific tasks. Finetune Procedure will empower model to conduct molecular editing based on input molecule and task description.

<p align="center">
<img src="./figures/finetune.png" alt="" align=center />
</p>

## Normal Editing Benchmark
Because ChatDrug is multi-round editing with failure information, it's actually different with other models. We'll change ChatDrug to other baseline or empower our model to conduct multi-round editing.

<p align="center">
<img src="./figures/single_prop_edit.png" alt="" align=center />

MolEditFormer actually owns advantages in multi-property molecular editing, even compared with ChatDrug.

<p align="center">
<img src="./figures/double_prop_edit.png" alt="" align=center />

## Docking Editing Benchmark
Waiting for implementation, we'll update result after finding proper ground truth and benchmark, as editing benchmark in Docking is rare, we need to find proper evaluation method on any molecules as our ground truth.

*We find [Boltz-2](https://github.com/jwohlwend/boltz) as our ground truth of receptor affinity values. Fine-tuning and evaluation are in progress.*