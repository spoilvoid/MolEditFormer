lr_list="0.1 0.01 0.001"
for lr in $lr_list
do  
    store_dir="ckpt/MolAlign/edit_2nd_step/lr${lr}_100epoch"
    for edit_task_id in $(seq 101 111)
    do
        python3 molecule_edit.py \
            --edit_task_id $edit_task_id \
            --resume \
            --text_model_path ckpt/MolAlign/pretrain/2DGraph_gin-Sep-25-2024_17-50-40/best_text_model.pth \
            --text_projector_path ckpt/MolAlign/pretrain/2DGraph_gin-Sep-25-2024_17-50-40/best_text2latent.pth \
            --gen2joint_projector_path ckpt/MolAlign/edit_1st_step/2DGraph_gin_lr0.01-Sep-26-2024_22-48-11/best_gen2joint_projector.pth \
            --joint2gen_projector_path ckpt/MolAlign/edit_1st_step/2DGraph_gin_lr0.01-Sep-26-2024_22-48-11/best_joint2gen_projector.pth \
            --store_dir $store_dir \
            --lr $lr \
            --epoch_num 100 \
            --seed 42 \
            --gpu 1
    done

    for edit_task_id in $(seq 201 206)
    do
        python3 molecule_edit.py \
            --edit_task_id $edit_task_id \
            --resume \
            --text_model_path ckpt/MolAlign/pretrain/2DGraph_gin-Sep-25-2024_17-50-40/best_text_model.pth \
            --text_projector_path ckpt/MolAlign/pretrain/2DGraph_gin-Sep-25-2024_17-50-40/best_text2latent.pth \
            --gen2joint_projector_path ckpt/MolAlign/edit_1st_step/2DGraph_gin_lr0.01-Sep-26-2024_22-48-11/best_gen2joint_projector.pth \
            --joint2gen_projector_path ckpt/MolAlign/edit_1st_step/2DGraph_gin_lr0.01-Sep-26-2024_22-48-11/best_joint2gen_projector.pth \
            --store_dir $store_dir \
            --lr $lr \
            --epoch_num 100 \
            --seed 42 \
            --gpu 1
    done
done
