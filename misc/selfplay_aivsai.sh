game=./rts/game_MC/game model=actor_critic model_file=./rts/game_MC/model \
    python3 misc/eval_selfplay_aivsai.py --num_games 1024 \
                                         --batchsize 128 \
                                         --tqdm \
                                         --players "fs=50,type=AI_NN;fs=50,type=AI_NN" \
                                         --eval0_stats winrate \
                                         --eval1_stats winrate \
                                         --additional_labels id,last_terminal,seq \
                                         --load0 /home/yuandong/private_models/model-winrate-80.0-357800.bin \
                                         --load1 /home/yuandong/private_models/model-minirts-selfplay-0.5old-0.5simpleai.bin \
                                         --gpu 2 \
                                         --T 1 \
                                         --shuffle_player "$@"
