import math
import numpy as np
import random
from fireplace.exceptions import GameOver, InvalidAction
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        the state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            # print('\r\n'), print(i)
            #self.search(state, create_copy=True)
            game_copy = self.game.cloneAndRandomize(self.game.game)
            self.search(state, game_copy, 1)                            # TODO: Calculate everything relative to own perspective (own = 1, opponent = -1)

        s = self.game.stringRepresentation(state)

        counts = [self.Nsa[(s,(a,b))] if (s,(a,b)) in self.Nsa else 0 for a in range(21) for b in range(18)]
        if temp==0:     # return only the most visited action (first max if multiple with same value exist) if temperature is 0
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs
        counts = [x**(1./temp) for x in counts]         # if temp is binary  (only 0 or 1), we can say counts = x if temp is 1 and possibly speedup this calculation here
        probs = [x/float(sum(counts)) for x in counts]  # fraction x of counts, so that sum of all action probs is 1
        return probs

    def search(self, state, game_copy, curPlayer):
        """
        NEEDS TO RUN ON DEEPCOPY!!!

        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current state between -1 and 1 (1 means 100% chance for the current player to win).
                Given by neural network prediction or finding a terminal state (1 if current player wins, 0 if he lost)
        """
        # if create_copy:
        #     game_copy = self.cloneAndRandomize(self.game.game)
        # print(id(game_copy))

        s = self.game.stringRepresentation(state)                                # TODO: Accelerate by using one-hot-encoded bit representation?
        #curPlayer = 1 if game_copy.current_player.name == 'Player1' else -1     # TODO: always start with curPlayer to only reflect self and other player and compute MCTS tree for own perspective
        #curPlayer = 1                                                           # TODO: Calculate everything relative to own perspective (own = 1, opponent = -1)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(game_copy, 1)                   # TODO: 1 should be correct, not current player
        if game_copy.ended or game_copy.turn > 180:
            # terminal node
            # return -self.Es[s]
            return self.Es[s]       # return game result for current player, 1 if he won, -1 if he lost. 0 if not ended.

        if s not in self.Ps:
            # leaf node first visit
            self.Ps[s], v = self.nnet.predict(state)        # let neural network predict action vector P and state value v
            valids = self.game.getValidMoves(game_copy)     # get valid moves for game_copy.current_player
            self.Ps[s] = self.Ps[s]*valids                  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s                      # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # return -v
            return v

        # no leaf node first visit, no terminal node 
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(21):
            for b in range(18):
                if valids[a,b]:
                    if (s,(a,b)) in self.Qsa:
                        u = self.Qsa[(s,(a,b))] + self.args.cpuct*self.Ps[s][a,b]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,(a,b))])
                    else:
                        u = self.args.cpuct*self.Ps[s][a,b]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = (a,b)

        a = best_act

        next_s, curPlayer = self.game.getNextState(curPlayer, a, game_copy)
        # next_s = self.game.getState(game_copy)                                # was redundant, happend implicitly in getNextState
        # if not game_copy.ended:
            # v = self.search(next_s, create_copy=False)                        #call recursively
        v = self.search(next_s, game_copy, curPlayer)                           #call recursively
        # else:
            # v = -self.Es[s]
            # v = self.Es[s]

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        # return -v
        return v