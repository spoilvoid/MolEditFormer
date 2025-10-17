for task_id in $(seq 101 108)
do
    python3 -m MolEditFormer.smiles_molecular_edit \
        --model_mode edit \
        --text_model_path ckpt/MolEditFormer/finetune/Mixed-MolPair-Tagged-V3-SMILE-Decoder-April-02-2025/best_text_model.pth \
        --mol_model_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_molecule_model.pth \
        --fuser_path ckpt/MolEditFormer/finetune/Mixed-MolPair-Tagged-V3-SMILE-Decoder-April-02-2025/best_modality_fuser.pth \
        --data_dir data/EditBenchmark/zero_shot/v2 \
        --value_type discrete \
        --property_type name \
        --task_id $task_id \
        --dataset_mode iterative \
        --store_dir test \
        --dir_name tagged-v3-iterative \
        --sampling_alg greedy \
        --batch_size 48 \
        --seed 42 \
        --gpu 0
done

# for task_id in $(seq 201 206)
# do
#     python3 -m MolEditFormer.smiles_molecular_edit \
#         --model_mode edit \
#         --text_model_path ckpt/MolEditFormer/finetune/Mixed-MolPair-Tagged-V3-SMILE-Decoder-April-02-2025/best_text_model.pth \
#         --mol_model_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_molecule_model.pth \
#         --fuser_path ckpt/MolEditFormer/finetune/Mixed-MolPair-Tagged-V3-SMILE-Decoder-April-02-2025/best_modality_fuser.pth \
#         --data_dir data/EditBenchmark/zero_shot \
#         --version v3 \
#         --task_id $task_id \
#         --dataset_mode iterative \
#         --store_dir ckpt/MolEditFormer/inference/Mixed-MolPair-Tagged-V3-SMILE-Decoder-April-02-2025 \
#         --dir_name tagged-v3-iterative \
#         --sampling_alg greedy \
#         --batch_size 48 \
#         --seed 42 \
#         --gpu 0
# done