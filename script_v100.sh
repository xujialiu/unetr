CUDA_VISIBLE_DEVICES=3 python train.py \
    --result_root_path ./test \
    --result_name test_1 \
    --csv_path /raid0/xujialiu/DDR_seg/ddr_seg_cls.csv \
    --data_path /raid0/xujialiu/DDR_seg/preprocess \
    --finetune /raid0/xujialiu/checkpoints/my_VFM_Fundus_weights.pth \
    --nb_classes_cls 4 \
    --nb_classes_seg 4 \
    --batch_size 32 \
    --blr 1e-3