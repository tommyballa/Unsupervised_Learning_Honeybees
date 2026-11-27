cd "$(dirname "$0")/../.."   # segmentation/
venv/bin/python train_yolo.py device=2 imgsz=1440 freeze=10 name=bee_model_finetuned_yolo26s_batch3+4_img1440_freeze10 > logs/train_batch3+4_img1440_freeze10.log 2>&1
venv/bin/python train_yolo.py device=2 imgsz=1440 seed=1 name=bee_model_finetuned_yolo26s_batch3+4_img1440_seed1 > logs/train_batch3+4_img1440_seed1.log 2>&1
venv/bin/python logs/ablation/abl_eval.py 2 logs/ablation/abl_freeze_gpu2.json bee_model_finetuned_yolo26s_batch3+4_img1440_freeze10:1440 bee_model_finetuned_yolo26s_batch3+4_img1440_freeze10:2048 bee_model_finetuned_yolo26s_batch3+4_img1440_seed1:1440 bee_model_finetuned_yolo26s_batch3+4_img1440_seed1:2048 > logs/ablation/abl_freeze_gpu2.log 2>&1
