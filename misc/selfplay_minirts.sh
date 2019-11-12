MODEL=$1

# this script accepts the --gpu argument too
game=./rts/game_MC/game model_file=./rts/game_MC/model model=actor_critic \
    python3 misc/selfplay.py --num_games 1024 \
                             --batchsize 128 \
                             --tqdm \
                             --players "fs=50,type=AI_NN;fs=50,type=AI_NN" \
                             --trainer_stats winrate \
                             --additional_labels id,last_terminal,seq \
                             --load ${MODEL} \
                             --T 20 \
                             --shuffle_player
