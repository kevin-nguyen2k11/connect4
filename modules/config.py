class Config:
    #Game specific settings
    width=7
    height=6
    win_length=4
    max_moves=width*height
    dirs=[(1,0),(1,1),(0,1),(-1,1)]
    
    #Self play
    max_saved_games=160000
    num_simultaneous_games=8
    # num_high_temp_moves=20
    # num_simulations=400
    # num_threads=10
    buffer_directory='./data'

    #Root prior exploration noise
    # alpha=0.1
    # epsilon=0.25

    #UCB formula
    # c_base=500
    # c_init=1.25
    
    #Evaluation
    num_eval_games=100 #caution, change in cpp code as well
    win_margin=0.50

    #Training
    train_multiprocess=False
    num_workers=10
    batch_size=4096
    initial_epoch=0
    epochs_per_update=10
    epochs_per_checkpoint=2000
    batches_per_epoch=5

    #Model settings
    model_directory='./models'
    num_blocks=5
    num_filters=128
    momentum=(0.85,0.95)
    weight_decay=0.0001
    # alpha=0.3