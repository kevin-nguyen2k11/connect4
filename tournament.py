from modules.config import Config as config
from modules.build.src import cppconnect
import logging
import multiprocessing
from multiprocessing import Manager
import time
import itertools as iter
from pathlib import Path

def evaluate(pipe,ready_list,id):
    logging.basicConfig(filename='tournament_logs/evaluate_{}.log'.format(id),filemode='a',
                        format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    try:
        while pipe.poll(None):
            try:
                players=pipe.recv()
            except EOFError:
                break
            logging.info('Starting matches between %d %d',players[0],players[1])
            wins=0
            draws=0
            formatted_results=[]
            results=cppconnect.evaluate_tournament(config.model_directory+'/tournament/{}'.format(players[0]),
                                                config.model_directory+'/tournament/{}'.format(players[1]))
            for i,result in enumerate(results):
                first=i%2
                temp='addresult {} {} {}'.format(players[first],players[1-first],result)
                formatted_results.append(temp)
                logging.info(temp)
                if result==1:
                    draws+=1
                elif (result==2 and first==0) or (result==0 and first==1):
                    wins+=1
            logging.info('Wins: %d Losses: %d Draws: %d',wins,config.num_eval_games-wins-draws,draws)
            logging.info('\n')
            pipe.send(formatted_results)
            ready_list[id].clear()
        logging.info('Ending')
    except OSError:
        pass

if __name__=='__main__':
    models=range(8)
    matches=list(iter.combinations(models,2))
    ready_list=[multiprocessing.Event() for _ in range(config.num_simultaneous_games)]
    results=[]
    processes=[]
    pipe_list=[]
    for id in range(config.num_simultaneous_games):
        a,b=multiprocessing.Pipe(duplex=True)
        pipe_list.append(a)
        processes.append(multiprocessing.Process(target=evaluate,args=(b,ready_list,id)))
        processes[-1].start()
        time.sleep(1)
    match_num=0
    while len(results)<len(matches)*config.num_eval_games:
        for id,status in enumerate(ready_list):
            try:
                if pipe_list[id].poll():
                    results.extend(pipe_list[id].recv())
            except OSError:
                continue
            if not status.is_set():
                status.set()
                if match_num==len(matches)-1:
                    pipe_list[id].close()
                else:
                    pipe_list[id].send(matches[match_num])
                    match_num+=1
        time.sleep(30)
    print(results)

    for process in processes:
        process.join()   