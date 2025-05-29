import sys
import runpy
import os

os.chdir("/data_A/xujialiu/projects/0_exploration/UNETR")
os.environ["CUDA_VISIBLE_DEVICES"] = 3
# cmd = 'python /data_A/xujialiu/projects/0_exploration/UNETR/visionfm_model.py'
# cmd = 'python /data_A/xujialiu/projects/0_exploration/UNETR/unetr.py'
# cmd = 'python /data_A/xujialiu/projects/0_exploration/UNETR/visionfm_unetr.py'
cmd = r"""
python /data_A/xujialiu/projects/0_exploration/UNETR/train.py
    --result_root_path ./test
    --result_name test_1
    --csv_path /data_A/xujialiu/datasets/DDR_seg/250527_1_get_label/ddr_seg_cls.csv
    --data_path /data_A/xujialiu/datasets/DDR_seg/preprocess
    --finetune /data_A/xujialiu/checkpoints/foundation_model_weights/my_VFM_Fundus_weights.pth
    --nb_classes_cls 4
    --nb_classes_seg 4
    --batch_size 1
    --blr 1e-3
"""

cmd = cmd.split()
if cmd[0] == "python":
    """pop up the first in the args"""
    cmd.pop(0)

if cmd[0] == "-m":
    """pop up the first in the args"""
    cmd.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(cmd[1:])


fun(cmd[0], run_name="__main__")