from modules.config import Config as config
from modules.build.src import cppconnect
from datetime import datetime
import select
import signal
import logging
import time
import shutil

def pipe_full(conn):
    r,w,x=select.select([],[conn],[],0.0)
    return 0==len(w)

def self_play(send_game,new_best,terminate_event,id):
    print('Started self play process:',id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(filename='logs/self_play_{}.log'.format(id),filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    start_time=datetime.now()
    logging.info('Starting')
    self_player=cppconnect.Self_player(config.model_directory+'/best')
    game_batch=[]
    total_games=0
    while not terminate_event.is_set():
        if new_best[id].is_set():
            self_player.load_model(config.model_directory+'/best')
            logging.info('Loaded new best model')
            new_best[id].clear()
        game_batch.append(self_player.play_game())
        total_games+=1
        if(total_games%10==0):
            logging.info('Total games: %d',total_games)
            while pipe_full(send_game):
                logging.info('FULL')
                time.sleep(5)
            send_game.send(game_batch)
            game_batch=[]
    end_time=datetime.now()
    logging.info('Ending')
    duration=end_time-start_time
    hours,remainder=divmod(duration.seconds,3600)
    minutes,seconds=divmod(remainder,60)
    logging.info('Duration: %d hours, %d minutes, %d seconds',hours,minutes,seconds)

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
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(filename='logs/evaluate.log',filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    eval_logger=logging.getLogger('eval_logger')
    eval_logger.addFilter(DelayFilter(600))
    while not terminate_event.is_set():
        if not new_latest.is_set():
            eval_logger.info('Waiting for new model to evaluate')
            time.sleep(120)
            continue
        logging.info('Starting')
        shutil.copytree(config.model_directory+'/latest',config.model_directory+'/test')
        new_latest.clear()
        latest_wins,draws=cppconnect.evaluate(config.model_directory+'/test',config.model_directory+'/best')
        if draws==config.num_eval_games:
            logging.info('Ending with all draws')
            continue
        if (latest_wins/(config.num_eval_games))>=config.win_margin or (latest_wins>=45 and draws>=50):
            shutil.rmtree(config.model_directory+'/best')
            shutil.move(config.model_directory+'test',config.model_directory+'/best')
            for event in new_best:
                event.set()
            logging.info('Updated best model')
        logging.info('Finished evaluation with %f win rate',(latest_wins/(config.num_eval_games)))
        logging.info('Wins: %d Losses: %d Draws: %d',latest_wins,config.num_eval_games-latest_wins-draws,draws)
    logging.info('Ending')