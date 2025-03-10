#sleep 1h
#--master_port 65531 
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run  --nproc_per_node=1 \
--master_port 25006 deit_retrain.py \
--data-path Path-to-ImageNet \
--finetune ./pretrained/deit_small_patch16_224-cd65a155.pth \
--pruned_model_path Path-to-searched-checkpoint \
--model deit_small_patch16_224 \
--epochs 300 \
--batch-size 1024 \
--lr 1e-4 \
--warmup-epochs 0 \
--output_dir output/deit_small/retrain_0 \
--resume False \



