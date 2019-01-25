import Arena
from MCTS import MCTS
from Game import YEET as Game
from NNet import NNetWrapper as NNet
from dotted_dict import DottedDict as dotdict
import numpy as np
import logging, psutil, os, random, functools, sys
#from multiprocessing import freeze_support
import multiprocessing as mp

args = dotdict({
    'numGames': 16,                 # 48
    'numThreads': mp.cpu_count()    # 8
})

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, game_instance):
        # agent = game_instance.current_player
        choices = np.argwhere(self.game.getValidMoves(game_instance) == 1)
        return random.choice(choices)


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, game_instance):
        # display(board)
        idxid = 0
        you = game_instance.current_player

        print(f'YOUR HERO: {you.hero} | HEALTH: {you.hero.health}')
        print(f'OPPONENT\'S HEALTH: {you.opponent.hero.health}')
        print(f'MANA: {you.mana}')
        print('\n----------Hand----------')
        for idx, card in enumerate(you.hand):
            print(f'Name: {card}, Index: {idx}, Cost: {card.cost}, Is Playable? {card.is_playable()}')
            if card.type == 4:
                print(f'Attack: {card.atk}, Health: {card.health}')

        print('\n----------Your Field----------')
        for idx, card in enumerate(you.field):
            print(f' Name: {card}, Index: {idx+10}, Can Attack? {card.can_attack()}')
        print('\n----------Enemy Field----------')
        for idx, card in enumerate(you.opponent.field):
            print(f'Enemy: {card}')

        print('\n----------Other Actions----------')
        if you.hero.power.is_usable():
            print('Hero Power Available: Index: 17')
        if you.hero.can_attack():
            print('Attack with Weapon, Index: 18')
        print('End Turn, Index: 19 \n')

        actionid = int(input('Enter action index: '))

        if 0 <= actionid <= 9:
            if you.hand[actionid].requires_target():
                print('Choose a target:')
                for idx, target in enumerate(you.hand[actionid].targets):
                    print(f'Name: {target}, Index: {idx}')
                idxid = int(input('Enter idx:'))

        elif 10 <= actionid <= 16:
            print('Choose a target:')
            for idx, target in enumerate(you.field[actionid - 10].attack_targets):
                print(f'Name: {target}, Index: {idx}')
            idxid = int(input('Enter idx:'))

        elif actionid == 17:
            if you.hero.power.requires_target():
                print('Choose a target:')
                for idx, target in enumerate(you.hero.power.targets):
                    print(f'Name: {target}, Index: {idx}')
                idxid = int(input('Enter idx:'))

        elif actionid == 18:
            print('Choose a target:')
            for idx, target in enumerate(you.hero.power.attack_targets):
                print(f'Name: {target}, Index: {idx}')
            idxid = int(input('Enter idx:'))

        elif actionid == -1:
            you.hero.to_be_destroyed = True
            return 19, 0

        return actionid, idxid

if __name__ == '__main__':
    #freeze_support()
    # Start processes with lower priority to prevent system overload/hangs/freezes. Also set multiprocessing start method to spawn for Linux, since forking makes trouble
    p = psutil.Process(os.getpid())
    if sys.platform.startswith('win32'):
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif sys.platform.startswith('linux'):
        p.nice(5)
        mp.set_start_method('spawn')
    
    # Set number of threads for OpenMP
    os.environ["OMP_NUM_THREADS"] = "1"

    g = Game(is_basic=True)
    # Suppress logging from fireplace
    logger = logging.getLogger("fireplace")
    logger.setLevel(logging.WARNING)

    # all players
    hp = HumanPlayer(g).play
    rp = RandomPlayer(g).play

    # nnet players
    n1 = NNet()
    #n1.nnet.cuda()
    n1.load_checkpoint('./temp/', 'temp.pth.tar')
    # n1.load_checkpoint('./temp/', 'best18-287k-75i.pth.tar')           # newest network
    # n1.load_checkpoint('../remote/models/', 'test.pth.tar')
    argsNN = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, argsNN)
    #a1p = lambda x: mcts1.getActionProb(x, temp=0)
    a1p = functools.partial(mcts1.getActionProb, temp=0)

    n2 = NNet()
    n2.load_checkpoint('./temp/', 'temp.pth.tar')
    # n2.load_checkpoint('./temp/', 'temp18.pth.tar')
    # n2.load_checkpoint('../remote/models/', 'temp.pth.tar')
    argsNN = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, argsNN)
    #a2p = lambda x: mcts2.getActionProb(x, temp=0)
    a2p = functools.partial(mcts2.getActionProb, temp=0)

    # define agent 1 and agent 2. Are switched after half of the games. TODO: MCTS tree reset needed between games?
    arena = Arena.Arena(a1p, a2p, g)

    # show final results (P1 is agent 1, P2 is agent 2)
    p1_won, p2_won, draws = arena.playGames(args.numGames, args.numThreads, verbose=False)
    print(f'\nResults: P1 {p1_won}, P2 {p2_won}, Draws {draws}')

'''
ai 21, random 29

./temp, temp.pth.tar
Results: P1 41, P2 9, Draws 0

./tem, best.pth.tar
Results: P1 10, P2 40, Draws 0

./tem/, temp.pth.tar
Results: P1 10, P2 40, Draws 0
'''