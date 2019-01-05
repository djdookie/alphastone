from utils import Bar, AverageMeter
import numpy as np
from types import *
import time
from concurrent.futures import ProcessPoolExecutor
import tqdm
import logging


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, x, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        # Suppress logging from fireplace
        logger = logging.getLogger("fireplace")
        logger.setLevel(logging.WARNING)
        players = [self.player2, None, self.player1]
        # curPlayer = 1
        current_game = self.game.getInitGame()
        curPlayer = 1 if current_game.current_player.name == 'Player1' else -1
        #print(id(self.game))
        #print('\r\nStarting player: ' + current_game.current_player.name + ' ' + str(current_game.current_player.hero))
        it = 0
        while not current_game.ended or current_game.turn > 180:
            it+=1
            # if verbose:
            #     assert(self.display)
            #     print("Turn ", str(it), "Player ", str(curPlayer))
            #     self.display(current_game)
            if type(players[curPlayer+1]) is MethodType:
                action = players[curPlayer + 1](current_game)
                next_state, curPlayer = self.game.getNextState(curPlayer, (action), current_game)
            else:
                pi = players[curPlayer+1](self.game.getState(current_game))     # call partial function MCTS.getActionProb(currentState) for current active player
                pi_reshape = np.reshape(pi, (21, 18))
                action = np.where(pi_reshape==np.max(pi_reshape))
                next_state, curPlayer = self.game.getNextState(curPlayer, (action[0][0], action[1][0]), current_game)
        # if verbose:
        #     assert(self.display)
        #     print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        #     self.display(board)
        if verbose:
            print('\r\n' + str(current_game.players[0].hero) + " vs. " + str(current_game.players[1].hero))
            print(" Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(current_game, 1)))
        return self.game.getGameEnded(current_game, 1)     # returns 0 if game has not ended, 1 if player 1 won, -1 if player 1 lost

    def playGames(self, num, numThreads, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.    #TODO: Not necessary to switch sides because fireplace decides randomly who starts by tossing a coin

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        # logger = logging.getLogger("fireplace")
        # logger.setLevel(logging.WARNING)
        # eps_time = AverageMeter()
        # bar = Bar('Arena.playGames', max=num)
        # end = time.time()
        # eps = 0
        # maxeps = int(num)

        #num = int(num/2)
        halfNum = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        #for _ in range(num):
        with ProcessPoolExecutor(numThreads) as executor:
            results = list(tqdm.tqdm(executor.map(self.playGame, range(halfNum)), total=halfNum, desc='1st half'))

        #gameResult = self.playGame(verbose=verbose)
        for gameResult in results:
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            # eps += 1
            # eps_time.update(time.time() - end)
            # end = time.time()
            # bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
            #                                                                                            total=bar.elapsed_td, eta=bar.eta_td)
            # bar.next()

        # show intermediate result
        print('P1/P2 WINS : %d / %d ; DRAWS : %d' % (oneWon, twoWon, draws))
        self.player1, self.player2 = self.player2, self.player1     # switching sides not really needed since fireplace is randomly assigning sides to players
        
        #for _ in range(num):
        with ProcessPoolExecutor(numThreads) as executor:
            results = list(tqdm.tqdm(executor.map(self.playGame, range(halfNum)), total=halfNum, desc='2nd half'))

        #gameResult = self.playGame(verbose=verbose)
        for gameResult in results:
            if gameResult==1:
                oneWon+=1                
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
        #     eps += 1
        #     eps_time.update(time.time() - end)
        #     end = time.time()
        #     bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
        #                                                                                                total=bar.elapsed_td, eta=bar.eta_td)
        #     bar.next()
            
        # bar.finish()

        return oneWon, twoWon, draws
