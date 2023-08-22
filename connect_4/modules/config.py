class Config:
    #Game specific settings
    width=7
    height=6
    win_length=4
    max_moves=width*height
    dirs=[(1,0),(1,1),(0,1),(-1,1)]
    
    #Self play
    max_saved_games=30000
    num_simultaneous_games=8 #was 8
    num_high_temp_moves=5
    num_simulations=100
    num_threads=10
    buffer_directory='/Users/kevinnguyen/Documents/Python files/connect_4/data'

    #Root prior exploration noise
    alpha=0.1
    epsilon=0.25

    #UCB formula
    c_base=500
    c_init=1.05
    
    #Evaluation
    num_eval_games=100
    win_margin=0.5

    #Training
    train_multiprocess=False
    num_workers=1
    batch_size=128
    initial_epoch=0
    epochs_per_update=10
    epochs_per_checkpoint=150
    batches_per_epoch=50

    #Model settings
    model_directory='/Users/kevinnguyen/Documents/Python files/connect_4/models'
    num_blocks=1
    momentum=0.9
    weight_decay=0.0001
    learning_rate_schedule=[
        (0,1e-3),
        (1000,1e-4),
        (3000,1e-5),
        (5000,1e-6),
        (7000,1e-7),
        (9000,1e-8)
    ]
