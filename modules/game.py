from modules.config import Config as config
import numpy as np
import pickle
import tensorflow as tf

class GameController:

    width=config.width
    height=config.height
    win_length=config.win_length
    max_moves=config.max_moves
    dirs=np.asarray(config.dirs)

    def __init__(self,num_moves=0,ref_board=None,board=None):
        self.num_moves=num_moves
        self.move_history=np.empty((2,self.max_moves),dtype=int)
        self.policy_history=[]
        self.win_state=None
        self.current_player=(self.num_moves+1)%2
        self.ref_board=np.zeros((self.height,self.width,2),dtype=int) if ref_board is None else ref_board
        self.board=np.zeros((4,self.height,self.width,2),dtype=int) if board is None else board

    def __str__(self):
        return np.array2string(np.flip(self.ref_board[:,:,0]+self.ref_board[:,:,1]*2,0))    
    
    def clone(self):
        return GameController(self.num_moves,self.ref_board.copy(),self.board.copy())
    
    def get_state(self):
        if self.current_player:
            return np.flip(self.ref_board,axis=-1)
        return self.ref_board

    def get_legal_moves(self):
        if self.win_state!=None:
            return np.full(self.width,False)
        top_row=np.sum(self.ref_board[-1,:,:].T,axis=0)
        return top_row==0

    def make_move(self,x):
        if self.win_state!=None:
            print('Game already done')
            return None
        col=np.sum(self.ref_board[:,x,:].T,axis=0)
        y=np.argmax(col==0)
        if y==0 and col[0]==1:
            raise ValueError('Illegal move, chosen column is full')
        if x<0 or x>=self.width:
            raise ValueError('Illegal move, outside boundaries')
        self.current_player^=1
        self.ref_board[y,x,self.current_player]=1
        self.board[:,y,x,self.current_player]=1
        self.move_history[:,self.num_moves]=[x,y]
        self.num_moves+=1
        if self.is_win_condition(x,y):
            pass

    def is_win_condition(self,x,y):
        current_board=self.board[:,:,:,self.current_player]
        for i,dir in enumerate(self.dirs):
            board_slice=current_board[i]
            dir_tuple=(dir,-1*dir)
            temp=[0]*2
            for j in range(2):
                coord=np.asarray([y,x])+dir_tuple[j]
                try:
                    temp[j]=board_slice[tuple(coord)] if coord[0]>=0 and coord[1]>=0 else 0
                except IndexError:
                    continue
            board_slice[y,x]+=temp[0]+temp[1]
            if board_slice[y,x]>=self.win_length:
                self.win_state=1
                return True
            for j in range(2):
                for k in range(temp[j]):
                    coord=np.asarray([y,x])+(1+k)*dir_tuple[j]
                    board_slice[tuple(coord)]=board_slice[y,x]
        if self.num_moves==self.max_moves:
            self.win_state=0
            return True
        return False
    
    def state_from_hist(self,move_ind):
        if move_ind>=self.num_moves:
            raise ValueError('Move number out of range of game length')
        board=np.zeros((self.height,self.width,2),dtype=int)
        player=(move_ind+1)%2
        for i in range(2):
            ind=self.move_history[:,player:move_ind:2]
            board[:,:,i][ind[1],ind[0]]=1
            player^=1
        return board
    
    def label_from_hist(self,move_ind):
        factor=1 if (move_ind+1)%2==self.current_player else -1
        return (self.policy_history[move_ind],factor*self.win_state)

class GameStorage(tf.keras.utils.Sequence):

    width=config.width
    height=config.height
    max_saved_games=config.max_saved_games
    batch_size=config.batch_size
    batches_per_epoch=config.batches_per_epoch
    directory=config.buffer_directory
    random=np.random.default_rng()

    def __init__(self,evaluate=False):
        self.buffer=[None]*self.max_saved_games
        self.moves_per_game=np.zeros(self.max_saved_games,dtype=int)
        self.pointer=0
        self.num_saved_games=0
        self.evaluate=evaluate
        if evaluate:
            self.game_pointers=[]

    def get_pointers(self):
        moves=0
        for i,game in enumerate(self.buffer):
            for j in range(len(game[0])):
                if moves%self.batch_size==0:
                    self.game_pointers.append((i,j))
                moves+=1

    def save_games(self,games):
        if len(games)>self.max_saved_games:
            games=games[len(games)-self.max_saved_games:]
        for game in games:
            self.buffer[self.pointer]=game
            self.moves_per_game[self.pointer]=len(game[0])
            self.pointer+=1
            if self.pointer==self.max_saved_games:
                self.pointer=0
        self.num_saved_games+=len(games)
        if self.num_saved_games>self.max_saved_games:
            self.num_saved_games=self.max_saved_games

    def __getitem__(self,idx):
        if self.evaluate:
            size=self.batch_size
            if (idx+1)*self.batch_size>(end:=np.sum(self.moves_per_game)):
                size=end-idx*self.batch_size
            states=np.empty((size,self.height,self.width,2),dtype=int)
            policies=np.empty((size,self.width))
            values=np.empty((size,1),dtype=int)
            game_pointer,move_pointer=self.game_pointers[idx]
            for i in range(size):
                game=self.buffer[game_pointer]
                states[i,:,:,:]=self.state_from_hist(game[0],move_pointer)
                policies[i,:]=game[1][move_pointer]
                factor=1 if move_pointer%2==self.moves_per_game[game_pointer]%2 else -1
                values[i,:]=game[2]*factor
                move_pointer+=1
                if move_pointer==self.moves_per_game[game_pointer]:
                    move_pointer=0
                    game_pointer+=1
        else:
            normalized_moves_per_game=self.moves_per_game/np.sum(self.moves_per_game)
            chosen_games=self.random.choice(self.max_saved_games,size=self.batch_size,p=normalized_moves_per_game)
            move_ind=self.random.integers(self.moves_per_game[chosen_games])
            states=np.empty((self.batch_size,self.height,self.width,2),dtype=int)
            policies=np.empty((self.batch_size,self.width))
            values=np.empty((self.batch_size,1),dtype=int)
            for i in range(self.batch_size):
                game=self.buffer[chosen_games[i]]
                states[i,:,:,:]=self.state_from_hist(game[0],move_ind[i])
                policies[i,:]=game[1][move_ind[i]]
                factor=1 if move_ind[i]%2==self.moves_per_game[chosen_games[i]]%2 else -1
                values[i,:]=game[2]*factor
            to_flip=self.random.integers(1,size=self.batch_size,endpoint=True).astype(bool)
            states[to_flip,:,:,:]=np.flip(states[to_flip,:,:,:],2)
            policies[to_flip,:]=np.flip(policies[to_flip,:],1)
        return (states,(policies,values))

    def save_buffer(self):
        rotated=self.buffer[self.pointer:]+self.buffer[:self.pointer]
        with open(self.directory+'/buffer.dat','wb') as f:
            pickle.dump(rotated,f)

    def load_buffer(self):
        try:
            with open(self.directory+'/buffer.dat','rb') as f:
                buffer=pickle.load(f)
                if (num_games:=len(buffer))!=self.max_saved_games:
                    print('Warning: num_max_saved_games: {} differs from number of games in loaded buffer: {}'.format(self.max_saved_games,num_games))
                    if num_games<self.max_saved_games:
                        self.buffer[self.max_saved_games-num_games:]=buffer
                    else:
                        self.buffer=buffer[-self.max_saved_games:]
                else:
                    self.buffer=buffer
                print('Loaded',self.directory+'/buffer.dat')
        except OSError:
            print('Unable to load or could not find',self.directory+'/buffer.dat')
            return
        for i,game in enumerate(self.buffer):
            if not game:
                continue
            self.moves_per_game[i]=len(game[0])
            self.num_saved_games+=1
        
    def state_from_hist(self,history,move_ind):
        board=np.zeros((self.height,self.width,2),dtype=int)
        player=(move_ind+1)%2
        for i in range(2):
            ind=history[player:move_ind:2,:]
            board[:,:,i][ind[:,1],ind[:,0]]=1
            player^=1
        return board
    
    def __len__(self):
        if self.evaluate:
            return (np.sum(self.moves_per_game)+self.batch_size-1)//self.batch_size
        return self.batches_per_epoch