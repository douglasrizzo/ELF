#!/bin/sh

game=./rts/game_MC/game model=actor_critic model_file=./rts/game_MC/model \
    python3 run.py --batchsize 128 \
                   --players "type=AI_NN,fs=50,args=backup/AI_SIMPLE|start/500|decay/0.99;type=AI_SIMPLE,fs=20" \
                   --trainer_stats winrate \
                   --tqdm
