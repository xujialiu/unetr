CUDA_VISIBLE_DEVICES=3 python /data_A/xujialiu/projects/0_exploration/UNETR/train.py \
    --result_root_path ./test \
    --result_name test_1 \
    --csv_path /data_A/xujialiu/datasets/DDR_seg/250527_1_get_label/ddr_seg_cls.csv \
    --data_path /home/xujia/ddr_seg \
    --finetune /home/xujia/my_VFM_Fundus_weights.pth \
    --nb_classes_cls 4 \
    --nb_classes_seg 4 \
    --batch_size 32 \
    --blr 1e-3