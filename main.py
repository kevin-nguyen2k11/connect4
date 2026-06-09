from modules.config import Config as config
import modules.player as p
import modules.model as m
from modules.game import GameStorage
import multiprocessing
import time
from pathlib import Path

if __name__=='__main__':
    Path(config.buffer_directory).mkdir(exist_ok=True)
    Path(config.model_directory+'/best').mkdir(exist_ok=True)
    Path(config.model_directory+'/latest').mkdir(exist_ok=True)
    Path(config.model_directory+'/test').mkdir(exist_ok=True)
    Path('./logs').mkdir(exist_ok=True)
    m.bootstrap_model()
    new_latest=multiprocessing.Event()
    new_best=[multiprocessing.Event() for _ in range(config.num_simultaneous_games)]
    terminate_event=multiprocessing.Event()
    processes=[]
    recieve_pipe_list=[]
    for id in range(config.num_simultaneous_games):
        recieve,send=multiprocessing.Pipe(duplex=False)
        recieve_pipe_list.append(recieve)
        processes.append(multiprocessing.Process(target=p.self_play,args=(send,new_best,terminate_event,id)))
        processes[-1].start()
        time.sleep(1)
    processes.append(multiprocessing.Process(target=p.evaluate_model,args=(new_latest,new_best,terminate_event)))
    processes[-1].start()
    game_storage=GameStorage()
    game_storage.load_buffer()
    m.train_model(recieve_pipe_list,new_latest,game_storage)

    terminate_event.set()
    for pipe in recieve_pipe_list:
        try:
            if pipe.poll(None):
                try:
                    game_storage.save_games(pipe.recv())
                except EOFError:
                    continue
        except OSError:
            continue
    game_storage.save_buffer()
    for process in processes:
        process.join()