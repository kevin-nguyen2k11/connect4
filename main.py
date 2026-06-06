from modules.config import Config as config
import modules.player as p
import modules.model as m
from modules.game import GameStorage
import multiprocessing
from multiprocessing import Manager
import signal
import time

if __name__=='__main__':
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

    # import modules.game as g
    # import numpy as np
    # thing=g.GameStorage(True)
    # thing.load_buffer()
    # games=0
    # draws=0
    # for game in thing.buffer:
    #     if game[2]==0:
    #         draws+=1
    # print(len(thing.buffer),draws)
    # # for game in thing.buffer:
    # game=thing.buffer[-4]
    # # if not game or game[2]==0:
    # #     continue
    # print(len(game[0]))
    # for i in range(len(game[0])):
    #     state=thing.state_from_hist(game[0],i)
    #     print(np.flip(state[:,:,0]+state[:,:,1]*2,0))    
    #     row,col=game[0][i]
    #     factor=1 if i%2==len(game[0])%2 else -1
    #     value=factor*game[2]
    #     print("MOVE:",row,col)
    #     print("VALUE:",value)
    # break
    # print(games,draws)
    # print(thing.num_saved_games)


    # import modules.game as g
    # import numpy as np
    # thing1=g.GameStorage()
    # thing1.load_buffer()
    # # exclude=[0,1,4]
    # # # for num,game in enumerate(thing1.buffer):
    # game=thing1.buffer[-10]
    # factor=1 if 0%2==len(game[0])%2 else -1
    # value=factor*game[2]
    # # if value!=-1 or num in exclude:
    # #     continue
    # print("game length:",len(game[0]))
    # print('value:',value)
    # model=m.get_model('best')
    # temp1=game[1]
    # for i in range(len(game[0])):
    #     state=thing1.state_from_hist(game[0],i)
    #     print(np.flip(state[:,:,0]+state[:,:,1]*2,0))    
    #     row,col=game[0][i]
    #     factor=1 if i%2==len(game[0])%2 else -1
    #     value=factor*game[2]
    #     print("MOVE:",row,col)
    #     print("POLICY:",temp1[i])
    #     print("VALUE:",value)
    #     predictions=model(np.expand_dims(state,0),training=False)
    #     policies=predictions[0]
    #     print("PREDICTED POLICY:",policies)
    #     print("PREDICTED VALUE:",np.asarray(predictions[1]))
    #     # print('num:',num)
    #     # break
# def self_play():
#     from modules.game import GameController
#     from modules.mcts import MonteCarloTree
#     import logging

#     logging.basicConfig(filename='logs/test.log',filemode='a',
#                         format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
    # import numpy as np
    # model=m.get_model('latest')
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

# def error_callback(error):
#     print(f'Error: {error}')

# if __name__=='__main__':
    # self_play()
    # import logging
    # from modules.build.src import cppconnect

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

    
    # with (multiprocessing.Pool(multiprocessing.cpu_count(),initializer=signal.signal,initargs=(signal.SIGINT,signal.SIG_IGN)) as pool,
    #       Manager() as manager):
    #     m.bootstrap_model()
    #     new_latest=manager.Event()
    #     new_best=[manager.Event() for _ in range(config.num_simultaneous_games)]
    #     terminate_event=manager.Event()
    #     processes=[]
    #     pipe_list=[]
    #     for id in range(config.num_simultaneous_games):
    #         pipe_list.append(multiprocessing.Pipe(duplex=False))
    #         # pool.apply_async(func=p.self_play,args=(pipe_list[id][1],new_best,terminate_event,id),error_callback=error_callback)
    #         # time.sleep(1)
    #     recieve_pipe_list=[pipe[0] for pipe in pipe_list]
    #     processes.append(pool.apply_async(func=p.evaluate_model,args=(new_latest,new_best,terminate_event),error_callback=error_callback))
    #     m.train_model(recieve_pipe_list,new_latest)
    
    #     terminate_event.set()
    #     pool.close()
    #     for process in processes:
    #         process.get(300)
    #     pool.join()

    # from modules.game import GameStorage
    # import numpy as np
    # game_storage=GameStorage()
    # game_storage.load_buffer()
    # game=game_storage.buffer[-1]
    # for i in range(len(game[0])):
    #     board=game_storage.state_from_hist(game[0],i)
    #     if i%2:
    #         board=np.flip(board[:,:,0]+board[:,:,1]*2,0)
    #     else:
    #         board=np.flip(board[:,:,1]+board[:,:,0]*2,0)
    #     print(board,'\n')

    # from modules.build.src import cppconnect
    # self_player=cppconnect.Self_player(config.model_directory+'/best')
    # game_batch=[]
    # total_games=0
    # while True:
    #     game_batch.append(self_player.play_game())
    #     total_games+=1
    #     if(total_games%10==0):
    #         print('Total games: {}'.format(total_games))
    #         game_batch=[]
    #         self_player.load_model(config.model_directory+'/best')
    #         print('Loaded new best model')