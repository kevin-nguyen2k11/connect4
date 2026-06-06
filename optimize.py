from modules.config import Config as config
import modules.model as m
import modules.game as g
import numpy as np
import logging
from datetime import datetime
import hyperopt as h


logging.basicConfig(filename='logs/train.log',filemode='a',
                    format='%(asctime)s %(levelname)s:%(message)s',level=logging.INFO)

def objective(args):
    print('starting evaluation with parameters',args)
    start_time=datetime.now()
    b=args[-1]
    train=g.GameStorage(True)
    test=g.GameStorage(True)
    train.batch_size=b
    test.batch_size=2048
    train.load_buffer()
    i=0
    while not train.buffer[i]:
        i+=1
    train.buffer=train.buffer[i:]
    train.moves_per_game=train.moves_per_game[i:]
    split=int(0.8*len(train.buffer))
    test.buffer=train.buffer[split:]
    test.moves_per_game=train.moves_per_game[split:]
    train.buffer=train.buffer[:split]
    train.moves_per_game=train.moves_per_game[:split]
    train.max_saved_games=len(train.buffer)
    train.get_pointers()
    test.get_pointers()
    model=m.train_opt(train,10,args)
    result=model.evaluate(test)
    end_time=datetime.now()
    duration=end_time-start_time
    hours,remainder=divmod(duration.seconds,3600)
    minutes,seconds=divmod(remainder,60)
    print('Duration: {} hours, {} minutes, {} seconds'.format(hours,minutes,seconds))
    print('results for parameters',args)
    print(dict(zip(model.metrics_names,result)))
    return result[0]

if __name__=='__main__':
    space=(
        10**-h.hp.choice('w',np.arange(1,5,1,dtype=float)),
        h.hp.choice('n',np.arange(1,4,1,dtype=int)),
        h.hp.loguniform('m',-0.5,0),
        2**h.hp.choice('f',np.arange(4,9,1)),
        10**h.hp.uniform('l',-4,-1),
        2**h.hp.choice('b',np.arange(6,12,1))
    )
    best=h.fmin(objective,
                space=space,
                algo=h.tpe.suggest,
                max_evals=50)
    print(best)