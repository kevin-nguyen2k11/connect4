from modules.config import Config as config
import modules.player as p
import modules.model as m
import multiprocessing
from multiprocessing import Manager
import signal
import time

# if __name__=='__main__':
#     m.bootstrap_model()
#     new_latest=multiprocessing.Event()
#     new_best=[multiprocessing.Event() for _ in range(config.num_simultaneous_games)]
#     terminate_event=multiprocessing.Event()
#     processes=[]
#     recieve_pipe_list=[]
#     for id in range(config.num_simultaneous_games):
#         recieve,send=multiprocessing.Pipe(duplex=False)
#         recieve_pipe_list.append(recieve)
#         processes.append(multiprocessing.Process(target=g.self_play,args=(send,new_best,terminate_event,id)))
#         processes[-1].start()
#         time.sleep(1)
#     # processes.append(multiprocessing.Process(target=m.evaluate_model,args=(new_latest,new_best,terminate_event)))
#     # processes[-1].start()
#     m.train_model(recieve_pipe_list,new_latest)

#     terminate_event.set()
#     for process in processes:
#         process.join()   
    
# def self_play():
#     from modules.game import GameController
#     from modules.mcts import MonteCarloTree
#     import logging

#     logging.basicConfig(filename='logs/test.log',filemode='a',
#                         format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
#     model=m.get_model('best')
#     game_batch=[]
#     total_games=0
#     for i in range(100):
#         game=GameController()
#         mc_tree=MonteCarloTree(model)
#         while game.win_state is None:
#             temperature=1 if game.num_moves<=config.num_high_temp_moves else 0
#             try:
#                 move,pi=mc_tree.choose_move(game,temperature)
#             except:
#                 print('Exception occured in self play {}, retrying'.format(id))
#                 mc_tree.exception_occurred.clear()
#                 mc_tree.exception_dict={}
#                 for i in range(config.num_threads):
#                     mc_tree.queue_events[i].clear()
#                 mc_tree.results_event.clear()
#                 continue
#             game.make_move(move)
#             game.policy_history.append(pi)
#         game_batch.append(game)
#         total_games+=1
#         if(total_games%10==0):
#             logging.info('Total games: %d',total_games)

def error_callback(error):
    print(f'Error: {error}')

if __name__=='__main__':
    # # self_play()
    # import logging
    # from modules.build.src import player

    # logging.basicConfig(filename='logs/test.log',filemode='a',
    #                     format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    # self_player=player.Self_player(config.model_directory+'/best')
    # total_games=0
    # game_batch=[]
    # for i in range(100):
    #     game_batch.append(self_player.play_game(True))
    #     total_games+=1
    #     if(total_games%10==0):
    #         logging.info('Total games: %d',total_games)

    # game_storage.save_games(game_batch)
    # game_storage.print_board(0)
    # print(game_batch[0][0])
    # for i in range(10):
    #     game_storage.print_board(i)
    # print(game_batch)

    
    m.bootstrap_model()
    with (multiprocessing.Pool(multiprocessing.cpu_count(),initializer=signal.signal,initargs=(signal.SIGINT,signal.SIG_IGN)) as pool,
          Manager() as manager):
        new_latest=manager.Event()
        new_best=[manager.Event() for _ in range(config.num_simultaneous_games)]
        terminate_event=manager.Event()
        processes=[]
        pipe_list=[]
        for id in range(config.num_simultaneous_games):
            pipe_list.append(multiprocessing.Pipe(duplex=False))
            pool.apply_async(func=p.self_play,args=(pipe_list[id][1],new_best,terminate_event,id),error_callback=error_callback)
            time.sleep(1)
        recieve_pipe_list=[pipe[0] for pipe in pipe_list]
        processes.append(pool.apply_async(func=p.evaluate_model,args=(new_latest,new_best,terminate_event),error_callback=error_callback))
        m.train_model(recieve_pipe_list,new_latest)
    
        terminate_event.set()
        pool.close()
        for process in processes:
            process.get(300)
        pool.join()