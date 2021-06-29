
data_dir=./data/fairseq/bin
save_dir=./model
tensorboard_dir=./tensorboard
log_path=./log/train.log

fairseq-train $data_dir \
	--arch transformer --num-workers 4 --optimizer adam  --adam-betas '(0.9, 0.98)' --fp16 \
	--lr 5e-4 --lr-scheduler inverse_sqrt  --warmup-updates 4000 --clip-norm 0.1 --dropout 0.3 \
	--weight-decay 0.0001 --max-tokens 4096  --keep-last-epochs 5 --no-progress-bar --max-epoch 5000 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--save-dir $save_dir \
	--tensorboard-logdir $tensorboard_dir \
	> $log_path

