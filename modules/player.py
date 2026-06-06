from modules.config import Config as config
from modules.build.src import cppconnect
from datetime import datetime
import select
import signal
import logging
import time
import shutil
import os

def pipe_full(conn):
    r,w,x=select.select([],[conn],[],0.0)
    return 0==len(w)

def self_play(send_game,new_best,terminate_event,id):
    import modules.model as m
    import modules.game as g
    import numpy as np
    print('Started self play process:',id)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    logging.basicConfig(filename='logs/self_play_{}.log'.format(id),filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    start_time=datetime.now()
    self_player=cppconnect.Self_player(config.model_directory+'/best')
    game_batch=[]
    total_games=0
    draws=0
    logging.info('Starting')
    while not terminate_event.is_set():
        if new_best[id].is_set():
            self_player.load_model(config.model_directory+'/best')
            logging.info('Loaded new best model')
            new_best[id].clear()
        # game_batch.append(self_player.play_game())
        game=self_player.play_game()
        if game[2]==0:
            # logging.info('Draw')
            if draws>=2:
                continue
            draws+=1
        # elif len(game[0])%2!=1:
        #     logging.info('Loss with %d moves',len(game[0]))
        #     model=m.get_model('best')
        #     thing1=g.GameStorage()
        #     temp1=game[1]
        #     for i in range(len(game[0])):
        #         state=thing1.state_from_hist(game[0],i)
        #         logging.info(np.flip(state[:,:,0]+state[:,:,1]*2,0))    
        #         row,col=game[0][i]
        #         factor=1 if i%2==len(game[0])%2 else -1
        #         value=factor*game[2]
        #         logging.info("MOVE: %d %d",row,col)
        #         logging.info("POLICY: %s",np.array2string(temp1[i]))
        #         logging.info("VALUE: %f",value)
        #         predictions=model(np.expand_dims(state,0),training=False)
        #         policies=predictions[0].numpy()
        #         logging.info("PREDICTED POLICY: %s",np.array2string(policies))
        #         logging.info("PREDICTED VALUE: %f",np.asarray(predictions[1]))
        #     continue
        # else:
        #     logging.info('Total: %d',total_games+1)
        game_batch.append(game)
        total_games+=1
        if(total_games%10==0):
            logging.info('Total games: %d',total_games)
            while pipe_full(send_game):
                logging.info('FULL')
                time.sleep(120)
            send_game.send(game_batch)
            game_batch=[]
            draws=0
    if game_batch:
        send_game.send(game_batch)
    send_game.close()
    end_time=datetime.now()
    logging.info('Ending')
    duration=end_time-start_time
    days,remainder=divmod(duration.total_seconds(),86400)
    hours,remainder=divmod(remainder,3600)
    minutes,seconds=divmod(remainder,60)
    logging.info('Duration: %d days, %d hours, %d minutes, %d seconds',days,hours,minutes,seconds)

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

def evaluate_model(new_latest,new_best,terminate_event):
    print('Started evaluate process')
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    logging.basicConfig(filename='logs/evaluate.log',filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    eval_logger=logging.getLogger('eval_logger')
    eval_logger.addFilter(DelayFilter(1200))
    while not terminate_event.is_set():
        if not new_latest.is_set():
            eval_logger.info('Waiting for new model to evaluate')
            time.sleep(120)
            continue
        logging.info('Starting')
        shutil.copytree(config.model_directory+'/latest',config.model_directory+'/test',dirs_exist_ok=True)
        latest_wins,draws,best_length,latest_length=cppconnect.evaluate(config.model_directory+'/best',config.model_directory+'/test')
        losses=config.num_eval_games-latest_wins-draws
        a=2*best_length/config.num_eval_games
        b=2*latest_length/config.num_eval_games
        if latest_wins/(config.num_eval_games)>=config.win_margin or \
        (latest_wins/(config.num_eval_games)>=0.4 and latest_wins>losses*2):
            if latest_wins!=losses or a>b:
                shutil.copytree(config.model_directory+'/test',config.model_directory+'/best',dirs_exist_ok=True)
                time.sleep(5)
                for event in new_best:
                    event.set()
                logging.info('Updated best model')
        new_latest.clear()
        logging.info('Finished evaluation with %f win rate',(latest_wins/(config.num_eval_games)))
        logging.info('Wins: %d Losses: %d Draws: %d',latest_wins,config.num_eval_games-latest_wins-draws,draws)
        logging.info('Average game length when first: %f second: %f',b,a)
        logging.info('\n')
    logging.info('Ending')