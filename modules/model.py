from modules.config import Config as config
import os
import shutil
import time
from datetime import datetime
import logging
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers

def save_model(model,version):
    directory=os.path.join(config.model_directory,version)
    # for file in os.listdir(directory):
    #     os.remove(os.path.join(directory,file))
    # checkpoint.save(os.path.join(directory,'model'))
    # model.save_weights(directory+'/'+version+'/model')
    keras.models.save_model(model,directory,save_format='tf')

def get_model(version):
    directory=os.path.join(config.model_directory,version)
    # model=keras.models.load_model(directory)
    # print(model.get_weights())
    try:
        return keras.models.load_model(directory)
    except ValueError:
        return keras.models.load_model(directory,custom_objects={'Custom>Custom_loss':PolicyLoss})

def bootstrap_model():
    non_hidden_files=[f for f in os.listdir(os.path.join(config.model_directory,'best')) if not f.startswith('.')]
    if not non_hidden_files:
        save_model(initialize_model(),'best')

def initialize_model(w=None,n=None,m=None,f=None):
        weight_decay=w if w else config.weight_decay
        num_filters=f if f else config.num_filters

        inputs=keras.Input(shape=(config.height,config.width,2),name='board')
        x=layers.Conv2D(num_filters,3,kernel_regularizer=regularizers.L2(l2=weight_decay),
                        kernel_initializer='Orthogonal')(inputs)
        x=layers.BatchNormalization()(x)
        conv_block=layers.ReLU()(x)

        #res block
        for _ in range(n if n else config.num_blocks):
            x=layers.Conv2D(num_filters,3,padding='same',kernel_regularizer=regularizers.L2(l2=weight_decay),
                            kernel_initializer='Orthogonal')(conv_block)
            x=layers.BatchNormalization()(x)
            x=layers.ReLU()(x)
            x=layers.Conv2D(num_filters,3,padding='same',kernel_regularizer=regularizers.L2(l2=weight_decay),
                            kernel_initializer='Orthogonal')(x)
            x=layers.BatchNormalization()(x)
            add=layers.add([conv_block,x])
            conv_block=layers.ReLU()(add)

        #policy head
        policy=layers.Conv2D(32,1,kernel_regularizer=regularizers.L2(l2=weight_decay),
                             kernel_initializer='Orthogonal')(conv_block)
        policy=layers.BatchNormalization()(policy)
        policy=layers.ReLU()(policy)
        policy=layers.Flatten()(policy)
        policy_output=layers.Dense(config.width,name='policy',kernel_regularizer=regularizers.L2(l2=weight_decay),
                                   kernel_initializer='Orthogonal')(policy)

        #value head
        value=layers.Conv2D(32,1,kernel_regularizer=regularizers.L2(l2=weight_decay),
                            kernel_initializer='Orthogonal')(conv_block)
        value=layers.BatchNormalization()(value)
        value=layers.ReLU()(value)
        value=layers.Flatten()(value)
        value=layers.Dense(256,kernel_regularizer=regularizers.L2(l2=weight_decay),
                           kernel_initializer='Orthogonal')(value)
        value=layers.ReLU()(value)
        value_output=layers.Dense(1,activation='tanh',name='value',kernel_regularizer=regularizers.L2(l2=weight_decay),
                                  kernel_initializer='Orthogonal')(value)

        model=keras.Model(inputs=[inputs],outputs=[policy_output,value_output],name='f_theta')
        momentum=tf.Variable(m if m else 0.9)
        opt=tf.keras.optimizers.legacy.SGD(momentum=momentum)
        model.compile(optimizer=opt,loss=[PolicyLoss(),tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)])
        return model

@keras.saving.register_keras_serializable(name='Policy_loss')
class PolicyLoss(keras.losses.Loss):

    def __init__(self,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,name=None):
        super().__init__(reduction,name)

    def call(self,y_true,y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(tf.stop_gradient(y_true),y_pred)

class LoggingCallback(keras.callbacks.Callback):

    def __init__(self,print_fcn=print):
        super().__init__()
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

class EarlyStoppingByLossVal(keras.callbacks.Callback):

    def __init__(self,monitor='value_loss',value=0.15,verbose=1):
        super().__init__()
        self.monitor=monitor
        self.value=value
        self.verbose=verbose

    def on_epoch_end(self,epoch,logs={}):
        current=logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!'%self.monitor,RuntimeWarning)
        if current<=self.value:
            if self.verbose>0:
                print('Epoch %05d: loss threshold reached, stopping early'%epoch)
            self.model.stop_training=True
            raise KeyboardInterrupt

class LRFinder(keras.callbacks.Callback):

    def __init__(self,epochs=500,min=1e-9,max=1e-1,smoothing=0.005):
        super().__init__()
        self.epochs=epochs
        self.iters=epochs*config.batches_per_epoch
        self.smoothing=smoothing
        self.lr_list=min*np.power(max/min,np.linspace(0,1,self.iters))
        # self.lr_list=np.power(10,np.linspace(min,max,self.iters))

    def on_train_batch_begin(self,batch,logs=None):
        self.model.optimizer.learning_rate=next(self.lr_gen)
        
    def on_train_batch_end(self,batch,logs=None):
        loss=logs['loss']
        if self.loss_list:
            loss=self.smoothing*loss+(1-self.smoothing)*self.loss_list[-1]
        self.loss_list.append(loss)
        if loss>1.2*self.loss_list[0]:
            self.model.stop_training=True

    def get_max_lr(self,game_storage):
        test_model=get_model('latest')
        self.loss_list=[]
        self.lr_gen=(lr for lr in self.lr_list)
        test_model.fit(
            x=game_storage,
            epochs=self.epochs,
            callbacks=[self],
            max_queue_size=10,
            workers=config.num_workers,
            use_multiprocessing=config.train_multiprocess,
            verbose=0
        )
        import pickle
        thing=(self.lr_list,self.loss_list)
        with open('/Users/kevinnguyen/Documents/Python files/connect_4/lrfinder.dat','wb') as f:
            pickle.dump((thing),f)
        return self.valley()

    def valley(self):
        length=len(self.loss_list)
        max_start,max_end=0,0
        lds=[1]*length
        for i in range(1,length):
            for j in range(0,i):
                if (self.loss_list[i]<self.loss_list[j]) and (lds[i]<lds[j]+1):
                    lds[i]=lds[j]+1
                if lds[max_end]<lds[i]:
                    max_end=i
                    max_start=max_end-lds[max_end]
        sections=(max_end-max_start)/3
        idx=max_start+2*int(sections)
        return self.lr_list[idx]
    
class Annealer(keras.callbacks.Callback):

    def __init__(self,div=25,div_final=100000,pct_start=0.25):
        super().__init__()
        self.div=div
        self.div_final=div_final
        self.pct_start=pct_start
        self.iters=config.epochs_per_checkpoint
        self.lengths=(int(self.iters*pct_start),self.iters-int(self.iters*pct_start))
        self.lr_list=None
        self.mom_list=np.concatenate((
            self.cos_anneal(config.momentum[0],config.momentum[1],self.lengths[0],1),
            self.cos_anneal(config.momentum[0],config.momentum[1],self.lengths[1],-1)))
        self.counter=None

    def on_epoch_begin(self,epoch,logs=None):
        self.model.optimizer.learning_rate=self.lr_list[self.counter]
        self.model.optimizer.momentum=self.mom_list[self.counter]
        self.counter+=1

    def set_max_lr(self,max_lr,epochs):
        self.lr_list=np.concatenate((
            self.cos_anneal(max_lr/self.div,max_lr,self.lengths[0],-1),
            self.cos_anneal(max_lr/self.div_final,max_lr,self.lengths[1],1)))
        self.counter=epochs

    def cos_anneal(self,min,max,length,sign):
        return min+(1/2)*(max-min)*(1+sign*np.cos((np.pi/length)*np.arange(length)))

def train_model(recieve_pipe_list,new_latest,game_storage):
    print('Started train process')
    start_time=datetime.now()
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(filename='logs/train.log',filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    train_logger=logging.getLogger('train_logger')
    train_logger.addFilter(DelayFilter(300))
    logger_callback=LoggingCallback(logging.info)
    lr_finder=LRFinder()
    annealer=Annealer()
    non_hidden_files=[f for f in os.listdir(os.path.join(config.model_directory,'latest')) if not f.startswith('.')]
    if not non_hidden_files:
        shutil.copytree(config.model_directory+'/best',config.model_directory+'/latest',dirs_exist_ok=True)
    model=get_model('latest')
    max_lr=model.optimizer.learning_rate.numpy()
    epochs=config.initial_epoch
    logging.info('Starting')
    try:
        while True:
            games=[]
            for pipe in recieve_pipe_list:
                while pipe.poll():
                    games.extend(pipe.recv())
            if games:
                game_storage.save_games(games)
            if game_storage.num_saved_games<config.max_saved_games/2:
                train_logger.info('Waiting for sufficient games in buffer (currently %d)',game_storage.num_saved_games)
                time.sleep(60)
                continue
            if not np.any(annealer.lr_list):
                if np.isnan(max_lr):
                    max_lr=lr_finder.get_max_lr(game_storage)
                logging.info('Setting max lr to %f',max_lr)
                annealer.set_max_lr(max_lr,epochs%config.epochs_per_checkpoint)
            model.fit(
                x=game_storage,
                epochs=config.epochs_per_update+epochs,
                callbacks=[logger_callback,annealer],
                initial_epoch=epochs,
                max_queue_size=10,
                workers=config.num_workers,
                use_multiprocessing=config.train_multiprocess,
                verbose=0
            )
            epochs+=config.epochs_per_update
            if epochs%config.epochs_per_checkpoint==0:
                model.optimizer.learning_rate=np.nan
                save_model(model,'latest')
                game_storage.save_buffer()
                annealer.lr_list=None
                max_lr=np.nan
                if not new_latest.is_set():
                    new_latest.set()
    except KeyboardInterrupt:
        end_time=datetime.now()
        logging.info('Ending')
        duration=end_time-start_time
        days,remainder=divmod(duration.total_seconds(),86400)
        hours,remainder=divmod(remainder,3600)
        minutes,seconds=divmod(remainder,60)
        logging.info('Duration: %d days, %d hours, %d minutes, %d seconds',days,hours,minutes,seconds)
        logging.info('Total epochs: %d',epochs)
        model.optimizer.learning_rate=max_lr
        save_model(model,'latest')
        return
    
def train_opt(train,max_epochs,args):
    w,n,m,f,l,b=args
    filename='{}-{}-{}-{}-{}-{}'.format(*args)
    logger=logging.getLogger('train_logger')
    ch=logging.FileHandler('opt_logs/'+filename+'.log')
    formatter=logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger_callback=LoggingCallback(logger.info)
    lr_callback=tf.keras.callbacks.LearningRateScheduler(lambda _:l)
    logger.info('Starting')
    model=initialize_model(w,n,m,f)
    try:
        model.fit(
            x=train,
            epochs=max_epochs,
            callbacks=[logger_callback,lr_callback],
            max_queue_size=10000,
            workers=10,
            use_multiprocessing=False,
            verbose=0
        )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info('Ending')
        logger.removeHandler(ch)
        os.mkdir('./models/'+filename)
        save_model(model,filename)
        return model


# def evaluate_model(new_latest,new_best,terminate_event):
#     print('Started evaluate process')
#     signal.signal(signal.SIGINT,signal.SIG_IGN)
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