CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 \
--master_port 65532 --use_env deit_search.py  \
--data-path Path-to-ImageNet \
--finetune ./pretrained/deit_small_patch16_224-cd65a155.pth \
--model deit_small_patch16_224 \
--epochs-search 60 \
--epochs 300 \
--batch-size 256 \
--lr-search 1e-3 \
--lr 1e-3 \
--warmup-epochs 0 \
--p 0.5 \
--interval 10 \
--output_dir output/deit_small/spp_search \

