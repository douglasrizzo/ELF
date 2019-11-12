game=./go/game model=df_policy model_file=./go/df_model \
    python3 misc/df_console.py --online \
                               --use_mcts "$@"
