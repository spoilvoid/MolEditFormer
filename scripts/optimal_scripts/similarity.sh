# python3 -m MolEditFormer.smiles_reconstruct \
#     --model_mode reconstruct \
#     --text_tokenizer_dir ckpt/SciBERT \
#     --mol_model_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_molecule_model.pth \
#     --text_model_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_text_model.pth \
#     --dataset_mode main \
#     --data_dir data/PubChemEdit_ZINC250k/v1 \
#     --value_type continuous \
#     --property_type name \
#     --batch_size 16 \
#     --mixed \
#     --non2can_ratio 0.2 \
#     --can2can_ratio 0.8 \
#     --scaffold_hint \
#     --store_dir ckpt/test \
#     --dir_name Mixed-PubChemEdit-ZINC250K-Continuous-Name-SMILE-Decoder-Oct-16-2025 \
#     --validation_ratio 0.05 \
#     --epoch_num 10 \
#     --alpha 1.0 \
#     --text_lr 2e-5 \
#     --graph_lr 2e-5 \
#     --seed 42 \
#     --gpu 0

export CUDA_VISIBLE_DEVICES=1

python3 -m MolEditFormer.smiles_pretrain_similarity \
    --model_mode reconstruct \
    --text_tokenizer_dir ckpt/SciBERT \
    --mol_model_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_molecule_model.pth \
    --text_model_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_text_model.pth \
    --text_projector_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_text2latent.pth \
    --mol_projector_path ckpt/MolEditFormer/pretrain/Mixed-PubChemEdit-ZINC250K-Tagged-V3-SMILE-Decoder-April-02-2025/best_mol2latent.pth \
    --dataset_mode main \
    --data_dir data/PubChemEdit_ZINC250k/v1 \
    --value_type continuous \
    --property_type name \
    --batch_size 32 \
    --mixed \
    --scaffold_hint \
    --store_dir ckpt/test_original \
    --dir_name Mixed-PubChemEdit-ZINC250K-Continuous-Name-SMILE-Decoder-Oct-16-2025 \
    --validation_ratio 0.05 \
    --epoch_num 10 \
    --alpha 1.0 \
    --text_lr 2e-5 \
    --graph_lr 2e-5 \
    --seed 42 \
    --gpu 0