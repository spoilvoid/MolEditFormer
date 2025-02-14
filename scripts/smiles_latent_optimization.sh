python3 -m MolEditFormer.smiles_latent_optimization \
    --model_mode edit \
    --text_model_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_text_model.pth \
    --mol_model_path ckpt/MolEditFormer/pretrain/PubChemEdit-ZINC250K-SMILE-Decoder-Jan-21-2025/best_molecule_model.pth \
    --fuser_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_modality_fuser.pth \
    --data_dir data/EditBenchmark/QED_constrained_optimization \
    --sim_threshold 0.4 \
    --task_name QED_constrained_optimization \
    --dataset_mode iterative \
    --store_dir ckpt/MolEditFormer/inference/QED_constrained_optimization \
    --dir_name MolPair-SMILE-Encoder-Fuser-Feb-08-2025/iterative \
    --sampling_alg greedy \
    --batch_size 32 \
    --seed 42 \
    --gpu 0

python3 -m MolEditFormer.smiles_latent_optimization \
    --model_mode edit \
    --text_model_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_text_model.pth \
    --mol_model_path ckpt/MolEditFormer/pretrain/PubChemEdit-ZINC250K-SMILE-Decoder-Jan-21-2025/best_molecule_model.pth \
    --fuser_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_modality_fuser.pth \
    --data_dir data/EditBenchmark/QED_constrained_optimization \
    --sim_threshold 0.6 \
    --task_name QED_constrained_optimization \
    --dataset_mode iterative \
    --store_dir ckpt/MolEditFormer/inference/QED_constrained_optimization \
    --dir_name MolPair-SMILE-Encoder-Fuser-Feb-08-2025/iterative \
    --sampling_alg greedy \
    --batch_size 32 \
    --seed 42 \
    --gpu 0




python3 -m MolEditFormer.smiles_latent_optimization \
    --model_mode edit \
    --text_model_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_text_model.pth \
    --mol_model_path ckpt/MolEditFormer/pretrain/PubChemEdit-ZINC250K-SMILE-Decoder-Jan-21-2025/best_molecule_model.pth \
    --fuser_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_modality_fuser.pth \
    --data_dir data/EditBenchmark/PlogP_constrained_optimization \
    --sim_threshold 0.4 \
    --task_name PlogP_constrained_optimization \
    --dataset_mode iterative \
    --store_dir ckpt/MolEditFormer/inference/PlogP_constrained_optimization \
    --dir_name MolPair-SMILE-Encoder-Fuser-Feb-08-2025/iterative \
    --sampling_alg greedy \
    --batch_size 32 \
    --seed 42 \
    --gpu 0

python3 -m MolEditFormer.smiles_latent_optimization \
    --model_mode edit \
    --text_model_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_text_model.pth \
    --mol_model_path ckpt/MolEditFormer/pretrain/PubChemEdit-ZINC250K-SMILE-Decoder-Jan-21-2025/best_molecule_model.pth \
    --fuser_path ckpt/MolEditFormer/finetune/MolPair-SMILE-Encoder-Fuser-Feb-08-2025/best_modality_fuser.pth \
    --data_dir data/EditBenchmark/PlogP_constrained_optimization \
    --sim_threshold 0.6 \
    --task_name PlogP_constrained_optimization \
    --dataset_mode iterative \
    --store_dir ckpt/MolEditFormer/inference/PlogP_constrained_optimization \
    --dir_name MolPair-SMILE-Encoder-Fuser-Feb-08-2025/iterative \
    --sampling_alg greedy \
    --batch_size 32 \
    --seed 42 \
    --gpu 0