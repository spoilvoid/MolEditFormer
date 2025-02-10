for task_id in $(seq 101 108)
do
    python3 -m MolEditFormer.smiles_molecular_edit \
        --model_mode edit \
        --molecule_type SMILES \
        --text_model_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_text_model.pth \
        --mol_model_path ckpt/MolEditFormer/pretrain/PubChemEdit-ZINC250K-SMILE-Decoder-Jan-21-2025/best_molecule_model.pth \
        --fuser_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_modality_fuser.pth \
        --data_dir data/MolPair/mol_pair \
        --task_id $task_id \
        --store_dir ckpt/MolEditFormer/inference/edit \
        --sampling_alg greedy \
        --batch_size 8 \
        --seed 42 \
        --gpu 0
done

for task_id in $(seq 201 206)
do
    python3 -m MolEditFormer.smiles_molecular_edit \
        --model_mode edit \
        --molecule_type SMILES \
        --text_model_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_text_model.pth \
        --mol_model_path ckpt/MolEditFormer/pretrain/PubChemEdit-ZINC250K-SMILE-Decoder-Jan-21-2025/best_molecule_model.pth \
        --fuser_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_modality_fuser.pth \
        --data_dir data/MolPair/mol_pair \
        --task_id $task_id \
        --store_dir ckpt/MolEditFormer/inference/edit \
        --sampling_alg greedy \
        --batch_size 8 \
        --seed 42 \
        --gpu 0
done