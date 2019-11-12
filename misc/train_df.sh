game=./go/game model=df_policy model_file=./go/df_model \
    python3 misc/train.py --batchsize 128 \
                          --freq_update 1 \
                          --num_games 512 \
                          --T 1 \
                          --tqdm \
                          --list_file /home/yuandong/local/go/go_gogod/train.lst \
                          --trainer_stats rewards "$@"
