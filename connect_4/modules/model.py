from modules.config import Config as config
from modules.game import GameStorage
import os
import time
from datetime import datetime
import logging
# import signal
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
# from modules.mcts import MonteCarloTree

class LoggingCallback(keras.callbacks.Callback):

    def __init__(self,print_fcn=print):
        keras.callbacks.Callback.__init__(self)
        self.print_fcn=print_fcn

    def on_epoch_end(self,epoch,logs={}):
        msg="{Epoch: %i} %s"%(epoch,",".join("%s:%f"%(k,v) for k,v in logs.items()))
        self.print_fcn(msg)

class DelayFilter(logging.Filter):

    def __init__(self,delay):
        self.delay=delay
        self.last_msg_time=None

    def filter(self,record):
        temp=datetime.now()
        try:
            interval=(temp-self.last_msg_time).seconds
        except TypeError:
            interval=self.delay
        if interval>=self.delay:
            self.last_msg_time=temp
            return True
        return False

@keras.saving.register_keras_serializable()
class Custom_loss(keras.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        loss=(tf.keras.losses.mean_squared_error(tf.stop_gradient(y_true[1]),y_pred[1])
                +tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true[0]),logits=y_pred[0]))
        return loss
    
def scheduler(epoch,lr):
    new_lr=lr
    for item in config.learning_rate_schedule:
        if epoch>=item[0]:
            new_lr=item[1]
        else:
            break
    return new_lr
        
def save_model(model,version):
    directory=os.path.join(config.model_directory,version)
    # for file in os.listdir(directory):
    #     os.remove(os.path.join(directory,file))
    # checkpoint.save(os.path.join(directory,'model'))
    # model.save_weights(directory+'/'+version+'/model')
    keras.models.save_model(model,directory,save_format='tf')

def get_model(version):
    directory=os.path.join(config.model_directory,version)
    return keras.models.load_model(directory)

def bootstrap_model():
    non_hidden_files=[f for f in os.listdir(os.path.join(config.model_directory,'best')) if not f.startswith('.')]
    if not non_hidden_files:
        model=initialize_model()
        save_model(model,'best')

def initialize_model():
        weight_decay=config.weight_decay

        inputs=keras.Input(shape=(config.height,config.width,2),name='board')
        x=layers.Conv2D(256,3,kernel_regularizer=regularizers.L2(l2=weight_decay),
                        kernel_initializer='Orthogonal')(inputs)
        x=layers.BatchNormalization()(x)
        conv_block=layers.ReLU()(x)

        for _ in range(config.num_blocks):
            #res block
            x=layers.Conv2D(256,3,padding='same',kernel_regularizer=regularizers.L2(l2=weight_decay),
                            kernel_initializer='Orthogonal')(conv_block)
            x=layers.BatchNormalization()(x)
            x=layers.ReLU()(x)
            x=layers.Conv2D(256,3,padding='same',kernel_regularizer=regularizers.L2(l2=weight_decay),
                            kernel_initializer='Orthogonal')(x)
            x=layers.BatchNormalization()(x)
            add=layers.add([conv_block,x])
            conv_block=layers.ReLU()(add)

        #policy head
        policy=layers.Conv2D(2,1,kernel_regularizer=regularizers.L2(l2=weight_decay),
                             kernel_initializer='Orthogonal')(conv_block)
        policy=layers.BatchNormalization()(policy)
        policy=layers.ReLU()(policy)
        policy=layers.Flatten()(policy)
        policy_output=layers.Dense(config.width,name='policy',kernel_regularizer=regularizers.L2(l2=weight_decay),
                                   kernel_initializer='Orthogonal')(policy)

        #value head
        value=layers.Conv2D(1,1,kernel_regularizer=regularizers.L2(l2=weight_decay),
                            kernel_initializer='Orthogonal')(conv_block)
        value=layers.BatchNormalization()(value)
        value=layers.ReLU()(value)
        value=layers.Flatten()(value)
        value=layers.Dense(256,kernel_regularizer=regularizers.L2(l2=weight_decay),
                           kernel_initializer='Orthogonal')(value)
        value=layers.ReLU()(value)
        value_output=layers.Dense(1,activation='tanh',name='value',kernel_regularizer=regularizers.L2(l2=weight_decay),
                                  kernel_initializer='Orthogonal')(value)

        custom_loss=Custom_loss()
        avg_loss=tf.keras.metrics.Mean('loss',dtype=tf.float32)
        class custom_fit(tf.keras.Model):
            def train_step(self,data):
                states,labels=data
                with tf.GradientTape() as tape:
                    outputs=self(states,training=True) # forward pass 
                    reg_loss=tf.reduce_sum(self.losses)
                    pred_loss=custom_loss.call(labels,outputs)
                    total_loss=tf.reduce_sum(pred_loss)+reg_loss
                gradients=tape.gradient(total_loss,self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
                avg_loss.update_state(total_loss)
                return {'loss':avg_loss.result()}

            @property
            def metrics(self):
                return [avg_loss]

        model=custom_fit(inputs=[inputs],outputs=[policy_output,value_output],name='f_theta')
        opt=tf.keras.optimizers.legacy.SGD(momentum=config.momentum)
        model.compile(optimizer=opt,run_eagerly=False,loss=custom_loss)
        return model

def train_model(recieve_pipe_list,new_latest):
    print('Started train process')
    start_time=datetime.now()
    logging.basicConfig(filename='logs/train.log',filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    logger_callback=LoggingCallback(logging.info)
    lr_callback=tf.keras.callbacks.LearningRateScheduler(scheduler)
    train_logger=logging.getLogger('train_logger')
    train_logger.addFilter(DelayFilter(300))
    logging.info('Starting')
    model=get_model('best')
    game_storage=GameStorage()
    game_storage.load_buffer()
    epochs=config.initial_epoch
    try:
        while True:
            games=[]
            for recieve_pipe in recieve_pipe_list:
                while recieve_pipe.poll():
                    games.extend(recieve_pipe.recv())
            if games:
                game_storage.save_games(games)
            if game_storage.num_saved_games<config.max_saved_games/2:
                train_logger.info('Waiting for sufficient games in buffer (currently %d)',game_storage.num_saved_games)
                time.sleep(5)
                continue
            model.fit(
                x=game_storage,
                epochs=config.epochs_per_update+epochs,
                callbacks=[logger_callback,lr_callback],
                initial_epoch=epochs,
                max_queue_size=100,
                workers=config.num_workers,
                use_multiprocessing=config.train_multiprocess,
                verbose=0
            )
            epochs+=config.epochs_per_update
            if not new_latest.is_set() and epochs%config.epochs_per_checkpoint==0:
                save_model(model,'latest')
                new_latest.set()
            if epochs%1000==0:
                game_storage.save_buffer()
    except KeyboardInterrupt:
        end_time=datetime.now()
        logging.info('Ending')
        duration=end_time-start_time
        hours,remainder=divmod(duration.seconds,3600)
        minutes,seconds=divmod(remainder,60)
        logging.info('Duration: %d hours, %d minutes, %d seconds',hours,minutes,seconds)
        logging.info('Total epochs: %d',epochs)
        game_storage.save_buffer()
        return

# def evaluate_model(new_latest,new_best,terminate_event):
#     print('Started evaluate process')
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
#     logging.basicConfig(filename='logs/evaluate.log',filemode='a',
#                         format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
#     eval_logger=logging.getLogger('eval_logger')
#     eval_logger.addFilter(DelayFilter(600))
#     while not terminate_event.is_set():
#         if not new_latest.is_set():
#             eval_logger.info('Waiting for new model to evaluate')
#             time.sleep(120)
#             continue
#         logging.info('Starting')
#         latest_model=get_model('latest')
#         new_latest.clear()
#         best_model=get_model('best')
#         latest_wins=0
#         draws=0
#         for i in range(config.num_eval_games):
#             if((i+1)%10==0):
#                 logging.info('Total games: %d',(i+1))
#             game=g.GameController()
#             trees=[MonteCarloTree(best_model,noise=False),MonteCarloTree(latest_model,noise=False)]
#             current_player=i%2
#             while game.win_state is None:
#                 move,pi=trees[current_player].choose_move(game,temperature=0)
#                 current_player^=1
#                 trees[current_player].opponent_move(move)
#                 game.make_move(move)
#             if game.win_state==0:
#                 draws+=1
#             elif not current_player:
#                 latest_wins+=1
#         if draws==config.num_eval_games:
#             logging.info('Ending with all draws')
#             continue
#         if (latest_wins/(config.num_eval_games))>=config.win_margin or (latest_wins>=45 and draws>=50):
#             save_model(latest_model,'best')
#             for event in new_best:
#                 event.set()
#             logging.info('Updated best model')
#         logging.info('Finished evaluation with %f win rate',(latest_wins/(config.num_eval_games)))
#         logging.info('Wins: %d Losses: %d Draws: %d',latest_wins,config.num_eval_games-latest_wins-draws,draws)
#     logging.info('Ending')