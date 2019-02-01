from utils import Bar, AverageMeter
import numpy as np
from types import *
import time
# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import tqdm
import logging
from tensorboardX import SummaryWriter
import sqlite3

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

    def playGame(self, game_number, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        # Suppress logging from fireplace
        fireplace_logger = logging.getLogger("fireplace")
        fireplace_logger.setLevel(logging.WARNING)

        if verbose:
            fireplace_logger.handlers = []
            # action_logger = logging.getLogger("action")
            # game_logger = logging.getLogger("game")
            # result_logger = logging.getLogger("result")
            fireplace_logger.setLevel(logging.WARNING)
            # action_logger.setLevel(logging.INFO)
            # game_logger.setLevel(logging.INFO)
            # result_logger.setLevel(logging.INFO)
            # create a file handler
            fireplace_handler = logging.FileHandler(f'logs/fireplace-{game_number}.log')
            # action_handler = logging.FileHandler(f'logs/action-{game_number}.log')
            # game_handler = logging.FileHandler(f'logs/game-{game_number}.csv')
            # result_handler = logging.FileHandler(f'logs/results.csv')
            # handler.setLevel(logging.INFO)
            # create a logging format
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fireplace_handler.setFormatter(formatter)
            # action_handler.setFormatter(formatter)
            # game_handler.setFormatter(logging.Formatter())
            # result_handler.setFormatter(logging.Formatter())
            # add the handlers to the logger
            fireplace_logger.addHandler(fireplace_handler)
            # action_logger.addHandler(action_handler)
            # game_logger.addHandler(game_handler)
            # result_logger.addHandler(result_handler)
            records = []
            
        players = [self.player2, None, self.player1]
        # curPlayer = 1
        current_game = self.game.getInitGame()
        curPlayer = 1 if current_game.current_player.name == 'Player1' else -1
        # print(id(self.game))
        # print('\r\nStarting player: ' + current_game.current_player.name + ' ' + str(current_game.current_player.hero))
        # if verbose:
            # game_logger.info(';'.join(["Game", "Turn", "Current player", "P1 Mana", "P2 Mana", "P1 Health", "P2 Health", "P1 Handsize", "P2 Handsize", "P1 Fieldsize", "P2 Fieldsize", "P1 Decksize", "P2 Decksize", "Action", "Target", "Activity"]))
            #result_logger.info(';'.join(["Game", "Hero 1", "Hero 2", "Result"]))
            # arena_writer = SummaryWriter('runs/arena-1')
        it = 0
        while not current_game.ended or current_game.turn > 180:
            it+=1
            # if verbose:
            #     assert(self.display)
            #     print("Turn ", str(it), "Player ", str(curPlayer))
            #     self.display(current_game)
            if verbose:
                # print("########## Turn ", str(it), current_game.current_player.name, "Mana", str(current_game.current_player.mana), "Health", str(current_game.players[0].hero.health), " - ", str(current_game.players[1].hero.health))
                name = current_game.current_player.name
                p1mana = current_game.players[0].mana
                p2mana = current_game.players[1].mana
                p1health = current_game.players[0].hero.health
                p2health = current_game.players[1].hero.health
                p1handsize = len(current_game.players[0].hand)
                p2handsize = len(current_game.players[1].hand)
                p1fieldsize = len(current_game.players[0].field)
                p2fieldsize = len(current_game.players[1].field)
                p1decksize = len(current_game.players[0].deck)
                p2decksize = len(current_game.players[1].deck)
            if type(players[curPlayer+1]) is MethodType:
                action = players[curPlayer + 1](current_game)
                if verbose:
                    act = [action[0], action[1]]
                    activity = self.game.getActionInfo((action[0][0], action[1][0]), current_game)
                next_state, curPlayer = self.game.getNextState(curPlayer, (action), current_game)
            else:
                fireplace_logger.setLevel(logging.WARNING)                      # disable logging in MCTS
                pi = players[curPlayer+1](self.game.getState(current_game))     # call partial function MCTS.getActionProb(currentState) for current active player
                if verbose: fireplace_logger.setLevel(logging.DEBUG)            # reenable logging, if logging is activated

                pi_reshape = np.reshape(pi, (21, 18))
                #action = np.where(pi_reshape==np.max(pi_reshape))
                #x = np.random.choice(len(action[0]))                           # pick random action for multiple available max-pi actions
                x = np.random.choice(len(pi), p=pi)                             # pick action using the probability vector pi
                action = np.unravel_index(np.ravel(x, np.asarray(pi).shape), pi_reshape.shape)
                if verbose:
                    act = [action[0][0], action[1][0]]
                    activity = self.game.getActionInfo((action[0][0], action[1][0]), current_game)
                next_state, curPlayer = self.game.getNextState(curPlayer, (action[0][0], action[1][0]), current_game)   # choose random action for real randomness, otherwise Player 1 has disadvantage because he is often picked as target=0
            if verbose:
                # print("########## Action ", str(action[0][0]), str(action[1][0]))
                # action_logger.info(name + ": " + str(activity))
                # game_logger.info(';'.join([str(it), name, str(p1mana), str(p2mana), str(p1health), str(p2health), str(p1handsize), str(p2handsize), str(p1fieldsize), str(p2fieldsize), str(p1decksize), str(p2decksize), str(act[0]), str(act[1]), str(activity)]))
                # Insert a row of data
                #c.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (str(game_number), str(it), name, str(p1mana), str(p2mana), str(p1health), str(p2health), str(p1handsize), str(p2handsize), str(p1fieldsize), str(p2fieldsize), str(p1decksize), str(p2decksize), str(act[0]), str(act[1]), str(activity)))
                records.append((str(game_number), str(it), name, str(p1mana), str(p2mana), str(p1health), str(p2health), str(p1handsize), str(p2handsize), str(p1fieldsize), str(p2fieldsize), str(p1decksize), str(p2decksize), str(act[0]), str(act[1]), str(activity)))
        # if verbose:
        #     assert(self.display)
        #     print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        #     self.display(board)
        result = self.game.getGameEnded(current_game, 1)
        if verbose:
            # print('\r\n' + str(current_game.players[0].hero) + " vs. " + str(current_game.players[1].hero))
            ##print(" Game over: Turn ", str(it), "Result ", str(result))
            # result_logger.info(';'.join([str(game_number), str(current_game.players[0].hero), str(current_game.players[1].hero), str(result)]))
            try:
                conn = sqlite3.connect('logs/alphastone.db')
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS games (Game INT, Turn INT, Current_player TEXT, P1_Mana INT, P2_Mana INT, P1_Health INT, P2_Health INT, P1_Handsize INT, P2_Handsize INT, P1_Fieldsize INT, P2_Fieldsize INT, P1_Decksize INT, P2_Decksize INT, Action INT, Target INT, Activity TEXT)")
                c.execute("CREATE TABLE IF NOT EXISTS results (Game INT, Hero_1 TEXT, Hero_2 TEXT, Result INT)")
                c.execute("INSERT INTO results VALUES (?,?,?,?)", (str(game_number), str(current_game.players[0].hero), str(current_game.players[1].hero), str(result)))
                c.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
                conn.commit()
                conn.close()
            except:
                # TODO: logging here
                raise
            # arena_writer.add_scalar('result', result, game_number)
            # arena_writer.add_scalar('turns', it, game_number)
            # arena_writer.add_scalar('player1_health', p1health, game_number)
            # arena_writer.add_scalar('player2_health', p2health, game_number)
        return result     # returns 0 if game has not ended, 1 if player 1 won, -1 if player 1 lost

    def playGames(self, num, numThreads, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.    #TODO: Not necessary to switch sides because fireplace decides randomly who starts by tossing a coin

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        logger = logging.getLogger("fireplace")
        logger.setLevel(logging.WARNING)
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
        # with ProcessPoolExecutor(numThreads) as executor:
        #     results = list(tqdm.tqdm(executor.map(self.playGame, range(halfNum)), total=halfNum, desc='1st half'))
        with Pool(numThreads) as pool:
            results = list(tqdm.tqdm(pool.imap(self.playGame, range(1, halfNum+1)), total=halfNum, desc='1st half'))

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

        # show intermediate results (P1 is agent 1, P2 is agent 2)
        print('A1/A2 WINS : %d / %d ; DRAWS : %d' % (oneWon, twoWon, draws))
        self.player1, self.player2 = self.player2, self.player1     # agents switching sides, not game players or heroes
        
        #for _ in range(num):
        # with ProcessPoolExecutor(numThreads) as executor:
        #     results = list(tqdm.tqdm(executor.map(self.playGame, range(halfNum)), total=halfNum, desc='2nd half'))
        with Pool(numThreads) as pool:
            results = list(tqdm.tqdm(pool.imap(self.playGame, range(halfNum+1, num+1)), total=halfNum, desc='2nd half'))

        #gameResult = self.playGame(verbose=verbose)
        for gameResult in results:
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
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
