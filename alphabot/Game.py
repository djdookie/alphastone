import random
import numpy as np
import pickle
import copy

from fireplace import cards
from fireplace.exceptions import GameOver, InvalidAction
from fireplace.game import Game
from fireplace.player import Player
from fireplace.utils import random_draft
from hearthstone.enums import CardClass, CardSet
from utils import UnhandledAction
from fireplace.exceptions import GameOver
from fireplace.deck import Deck
from sklearn.preprocessing import label_binarize

class YEET:
    """
    Use 1 for player1 and -1 for player2.
    21 possible actions per move, and 8 possible targets per action + 1 if no targets
    is_basic = True initializes game between priest and rogue only
    """

    def __init__(self, is_basic=True):
        self.game = None        # Fireplace game instance
        #self.is_basic = True
        self.players = ['player1', 'player2']
        self.is_basic = is_basic
        self.isolate = False

    def isolateSet(self, filename='notbasicset', set='CardSet.CORE'):
        # isolates the specified card set for exclusion in drafting
        cards.db.initialize()
        extraset = []
        for index, card in cards.db.items():
            if str(card.card_set) != set:
                    extraset.append(card.id)
        with open(f'{filename}.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(extraset, filehandle)

    def getInitGame(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        if self.isolate:
            self.isolateSet()

        cards.db.initialize()
        if self.is_basic: #create quick simple game
            # with open('notbasic.data', 'rb') as f:
            #     extra_set = pickle.load(f)
            
            extra_set = cards.filter(
                card_set = [CardSet.EXPERT1, CardSet.HOF, CardSet.NAXX, CardSet.GVG, CardSet.BRM, CardSet.TGT, CardSet.LOE, CardSet.OG, CardSet.KARA, CardSet.GANGS,
                            CardSet.UNGORO, CardSet.ICECROWN, CardSet.LOOTAPALOOZA, CardSet.GILNEAS, CardSet.BOOMSDAY, CardSet.TROLL]
            )
            # LOOTAPALOOZA = Kobolds and Catacombs # GILNEAS = Witchwood # TROLL = Rasthakan's Rumble

            # p1 = 6 #priest
            p1 = 7 #rogue
            p2 = 7 #rogue
            # p1 = 4 # mage
            # p2 = 4 # mage
            # deck1 = random_draft(CardClass(p1), exclude=extra_set)
            # deck2 = random_draft(CardClass(p2), exclude=extra_set)
            deck1 = self.roguebasic_draft() # use same shuffled rogue AI basic decks for now
            deck2 = self.roguebasic_draft()
        else:
            p1 = random.randint(1, 9)
            p2 = random.randint(1, 9)
            deck1 = random_draft(CardClass(p1))
            deck2 = random_draft(CardClass(p2))
        self.players[0] = Player("Player1", deck1, CardClass(p1).default_hero)
        self.players[1] = Player("Player2", deck2, CardClass(p2).default_hero)
        game = Game(players=self.players)
        game.start()

        # Skip mulligan for now (only mulligan expensive cards)
        for player in game.players:
            # if player.name == 'Player1':
            # cards_to_mulligan = [c for c in player.choice.cards if c.cost > 3]
            # else:
            cards_to_mulligan = random.sample(player.choice.cards, 0)
            player.choice.choose(*cards_to_mulligan)

        # track played card list
        self.players[0].playedcards = []
        self.players[1].playedcards = []

        #game.player_to_start = game.current_player      # obsolete?
        self.game = game
        return game

    def getNextState(self, player, action, game_instance):
        """
        Input:
            player: current player (1 or -1)
            action: action taken by current player
            game_instance: the game object (actual game or deepcopy for MCTS)

        Returns:
            next_state: state after applying action
            next_player: player who plays in the next turn (should be -player if the action is end turn)
        """
        # if game_instance == None:
        #     game_instance = self.game

        try:
            self.performAction(action, game_instance)
        except GameOver:
            raise GameOver

        next_state = self.getState(game_instance)

        if action[0] != 19:
            return next_state, player
        else:
            return next_state, -player

    def getValidMoves(self, game_instance):
        """
        Input:
            game_instance: the game object (actual game or deepcopy for MCTS)

        Returns:
            validMoves: a 21x18 binary matrix, 1 for
                        moves that are valid from the current game instance and player,
                        0 for invalid moves
        """
        # if game_instance == None:
        #     game_instance = self.game

        actions = np.zeros((21,18))
        player = game_instance.current_player
        #If the player is being given a choice, return only valid choices
        if player.choice:
            for index, card in enumerate(player.choice.cards):
                actions[20, index] = 1

        else:
            # add cards in hand
            for index, card in enumerate(player.hand):
                if card.is_playable():
                    if card.requires_target():
                        for target, card in enumerate(card.targets):
                            actions[index, target] = 1
                    elif card.must_choose_one:
                        for choice, card in enumerate(card.choose_cards):
                            actions[index, choice] = 1
                    else:
                        actions[index] = 1
            # add targets available to minions that can attack
            for position, minion in enumerate(player.field):
                if minion.can_attack():
                    for target, card in enumerate(minion.attack_targets):
                        actions[position+10, target] = 1
            # add hero power and targets if applicable
            if player.hero.power.is_usable():
                if player.hero.power.requires_target():
                    for target, card in enumerate(player.hero.power.targets):
                        actions[17, target] = 1
                else:
                    actions[17] = 1
            # add hero attacking if applicable
            if player.hero.can_attack():
                for target, card in enumerate(player.hero.attack_targets):
                    actions[18, target] = 1
            # add end turn
            actions[19,1] = 1
        return actions

    # def performAction(self, a, game_instance):
    #     """
    #     utility to perform an action tuple

    #     Input:
    #         a, a tuple representing index of action
    #         game_instance: the game object (actual game or deepcopy for MCTS)

    #     """
    #     player = game_instance.current_player
    #     if not game_instance.ended:
    #         try:
    #             if 0 <= a[0] <= 9:
    #                 if player.hand[a[0]].requires_target():
    #                     player.hand[a[0]].play(player.hand[a[0]].targets[a[1]])
    #                 elif player.hand[a[0]].must_choose_one:
    #                     player.hand[a[0]].play(choose=player.hand[a[0]].choose_targets[a[1]])
    #                 else:
    #                     player.hand[a[0]].play()
    #             elif 10 <= a[0] <= 16:
    #                 player.field[a[0]-10].attack(player.field[a[0]-10].attack_targets[a[1]])
    #             elif a[0] == 17:
    #                 if player.hero.power.requires_target():
    #                     player.hero.power.use(player.hero.power.play_targets[a[1]])
    #                 else:
    #                     player.hero.power.use()
    #             elif a[0] == 18:
    #                 player.hero.attack(player.hero.attack_targets[a[1]])
    #             elif a[0] == 19:
    #                 player.game.end_turn()
    #             elif a[0] == 20 and not player.choice:
    #                 player.game.end_turn()
    #             elif player.choice:
    #                 player.choice.choose(player.choice.cards[a[1]])
    #             else:
    #                 raise UnhandledAction
    #         except UnhandledAction as e:
    #             # print("\r\nAttempted to take an inappropriate action!")
    #             # print(a)
    #             print(str(e))
    #             raise
    #         except InvalidAction as e:
    #             # print("\r\nAttempted to do something I can't!")
    #             # print(a)
    #             # print(str(e))
    #             player.game.end_turn()      # TODO: Find out why we often land here!!!
    #         except IndexError:
    #             try:
    #                 player.game.end_turn()
    #             except GameOver:
    #                 pass
    #         except GameOver:
    #             pass

    def performAction(self, a, game_instance):
        """
        utility to perform an action tuple

        Input:
            a, a tuple representing index of action
            game_instance: the game object (actual game or deepcopy for MCTS)

        """
        player = game_instance.current_player
        if not game_instance.ended:
            try:
                if 0 <= a[0] <= 9:      # play card 1-10
                    if player.hand[a[0]].requires_target():
                        player.hand[a[0]].play(player.hand[a[0]].targets[a[1]])
                    elif player.hand[a[0]].must_choose_one:
                        player.hand[a[0]].play(choose=player.hand[a[0]].choose_targets[a[1]])
                    else:
                        player.hand[a[0]].play()
                    # save to played cards list for current player
                    player.playedcards.append(player.hand[a[0]].id)
                elif 10 <= a[0] <= 16:  # attack with minion 1-7
                    player.field[a[0]-10].attack(player.field[a[0]-10].attack_targets[a[1]])
                elif a[0] == 17:        # use hero power
                    if player.hero.power.requires_target():
                        player.hero.power.use(player.hero.power.play_targets[a[1]])
                    else:
                        player.hero.power.use()
                elif a[0] == 18:        # hero attack
                    player.hero.attack(player.hero.attack_targets[a[1]])
                elif a[0] == 19:        # end turn
                    player.game.end_turn()
                elif a[0] == 20 and not player.choice:
                    player.game.end_turn()
                elif player.choice:
                    player.choice.choose(player.choice.cards[a[1]])
                else:
                    raise UnhandledAction
            except UnhandledAction as e:
                # print("\r\nAttempted to take an inappropriate action!")
                # print(a)
                print(str(e))
                raise
            except InvalidAction as e:
                # print("\r\nAttempted to do something I can't!")
                # print(a)
                # print(str(e))
                player.game.end_turn()      # TODO: Find out why we often land here!!!
            except IndexError:
                try:
                    player.game.end_turn()
                except GameOver:
                    pass
            except GameOver:
                pass

    def getActionInfo(self, a, game_instance):
        """
        helper method to get info about the action and target for logging reasons

        Input:
            a, a tuple representing index of action
            game_instance: the game object (actual game or deepcopy for MCTS)

        """
        player = game_instance.current_player
        if not game_instance.ended:
            try:
                if 0 <= a[0] <= 9:
                    if player.hand[a[0]].requires_target():
                        return str(player.hand[a[0]]) + " targets " + str(player.hand[a[0]].targets[a[1]]) + " (" + str(player.hand[a[0]].targets[a[1]].controller.name) + ")"
                    elif player.hand[a[0]].must_choose_one:
                        return str(player.hand[a[0]]) + " chooses " + str(player.hand[a[0]].choose_targets[a[1]]) + " (" + str(player.hand[a[0]].choose_targets[a[1]].controller.name) + ")"
                    else:
                        return "plays " + str(player.hand[a[0]])
                elif 10 <= a[0] <= 16:
                    return str(player.field[a[0]-10]) + " attacks " + str(player.field[a[0]-10].attack_targets[a[1]]) + " (" + str(player.field[a[0]-10].attack_targets[a[1]].controller.name) + ")"
                elif a[0] == 17:
                    if player.hero.power.requires_target():
                        return "uses hero power on target " + str(player.hero.power.play_targets[a[1]]) + " (" + str(player.hero.power.play_targets[a[1]].controller.name) + ")"
                    else:
                        return "uses hero power"
                elif a[0] == 18:
                    return "hero attacks target " + str(player.hero.attack_targets[a[1]]) + " (" + str(player.hero.attack_targets[a[1]].controller.name) + ")"
                elif a[0] == 19:
                    return "ends turn"
                elif a[0] == 20 and not player.choice:
                    return "ends turn"
                elif player.choice:
                    return "chooses " + str(player.choice.cards[a[1]])
            except IndexError:
                try:
                    return "ends turn, because index was out of range"
                except GameOver:
                    pass
            except GameOver:
                pass

    # def getGameEnded(self, game_instance):
    #     """
    #     Input:
    #         game_instance: the game object (actual game or deepcopy for MCTS)

    #     Returns:
    #         r: 0 if game has not ended. 1 if starting player won, -1 if player lost,
    #            small non-zero value for draw.
    #     """
    #     # if game_instance == None:
    #     #     game_instance = self.game

    #     p1 = game_instance.player_to_start

    #     if p1.playstate == 4:
    #         return 1
    #     elif p1.playstate == 5:
    #         return -1
    #     elif p1.playstate == 6:
    #         return 0.0001
    #     elif game_instance.turn > 180:
    #         game_instance.ended = True
    #         return 0.0001
    #     return 0

    def getGameEnded(self, game_instance, player):
        """
        Input:
            game_instance: the game object (actual game or deepcopy for MCTS)
            player: the player to check for

        Returns:
            r: 0 if game has not ended. 1 if given player won, -1 if given player lost,
               small non-zero value for draw.
        """
        # if game_instance == None:
        #     game_instance = self.game

        if player == 1:
            # curPlayer is player 1
            p = game_instance.players[0]
        elif player == -1:
            # curPlayer is player 2
            p = game_instance.players[1]

        if p.playstate == 1:
            # still playing, early return
            return 0
        if p.playstate == 4:
            # given player won
            return 1
        elif p.playstate == 5:
            # given player lost
            return -1
        elif p.playstate == 6:
            # draw
            return 0.0001
        elif game_instance.turn > 180:
            game_instance.ended = True
            return 0.0001
        return 0

    # def getState(self, game_instance):
    #     """
    #     Args:
    #         game_instance: the game object (actual game or deepcopy for MCTS)
    #     return:
    #         a 273 length numpy array of features extracted from the
    #         supplied game.
    #     """
    #     # if game_instance == None:
    #     #     game_instance = self.game

    #     s = np.zeros(273, dtype=np.int32)

    #     p1 = game_instance.current_player
    #     p2 = p1.opponent

    #     #0-9 player1 class, we subtract 1 here because the classes are from 1 to 10
    #     s[p1.hero.card_class-1] = 1
    #     #10-19 player2 class
    #     s[10 + p2.hero.card_class-1] = 1
    #     i = 20
    #     # 20-21: current health of current player, then opponent
    #     s[i] = p1.hero.health
    #     s[i + 1] = p2.hero.health

    #     # 22: hero power usable y/n
    #     s[i + 2] = p1.hero.power.is_usable()*1
    #     # 23-24: # of mana crystals for you opponent
    #     s[i + 3] = p1.max_mana
    #     s[i + 4] = p2.max_mana
    #     # 25: # of crystals still available
    #     s[i + 5] = p1.mana
    #     #26-31: weapon equipped y/n, pow., dur. for you, then opponent
    #     s[i + 6] = 0 if p1.weapon is None else 1
    #     s[i + 7] = 0 if p1.weapon is None else p1.weapon.damage
    #     s[i + 8] = 0 if p1.weapon is None else p1.weapon.durability

    #     s[i + 9] = 0 if p2.weapon is None else 1
    #     s[i + 10] = 0 if p2.weapon is None else p2.weapon.damage
    #     s[i + 11] = 0 if p2.weapon is None else p2.weapon.durability

    #     # 32: number of cards in opponents hand
    #     s[i + 12] = len(p2.hand)
    #     #in play minions

    #     i = 33
    #     #33-102, your monsters on the field
    #     p1_minions = len(p1.field)
    #     for j in range(0, 7):
    #         if j < p1_minions:
    #             # filled y/n, pow, tough, current health, can attack
    #             s[i] = 1
    #             s[i + 1] = p1.field[j].atk
    #             s[i + 2] = p1.field[j].max_health
    #             s[i + 3] = p1.field[j].health
    #             s[i + 4] = p1.field[j].can_attack()*1
    #             # deathrattle, div shield, taunt, stealth y/n
    #             s[i + 5] = p1.field[j].has_deathrattle*1
    #             s[i + 6] = p1.field[j].divine_shield*1
    #             s[i + 7] = p1.field[j].taunt*1
    #             s[i + 8] = p1.field[j].stealthed*1
    #             s[i + 9] = p1.field[j].silenced*1
    #         i += 10

    #     #103-172, enemy monsters on the field
    #     p2_minions = len(p2.field)
    #     for j in range(0, 7):
    #         if j < p2_minions:
    #             # filled y/n, pow, tough, current health, can attack
    #             s[i] = 1
    #             s[i + 1] = p2.field[j].atk
    #             s[i + 2] = p2.field[j].max_health
    #             s[i + 3] = p2.field[j].health
    #             s[i + 4] = p2.field[j].can_attack()*1
    #             # deathrattle, div shield, taunt, stealth y/n
    #             s[i + 5] = p2.field[j].has_deathrattle*1
    #             s[i + 6] = p2.field[j].divine_shield*1
    #             s[i + 7] = p2.field[j].taunt*1
    #             s[i + 8] = p2.field[j].stealthed*1
    #             s[i + 9] = p2.field[j].silenced*1
    #         i += 10

    #     #in hand

    #     #173-272, your cards in hand
    #     p1_hand = len(p1.hand)
    #     for j in range(0, 10):
    #         if j < p1_hand:
    #             #card y/n
    #             s[i] = 1
    #             # minion y/n, attk, hp, battlecry, div shield, deathrattle, taunt
    #             s[i + 1] = 1 if p1.hand[j].type == 4 else 0
    #             s[i + 2] = p1.hand[j].atk if s[i + 1] == 1 else 0
    #             s[i + 3] = p1.hand[j].health if s[i + 1] == 1 else 0
    #             s[i + 4] = p1.hand[j].divine_shield*1 if s[i + 1] == 1 else 0
    #             s[i + 5] = p1.hand[j].has_deathrattle*1 if s[i + 1] == 1 else 0
    #             s[i + 6] = p1.hand[j].taunt*1 if s[i + 1] == 1 else 0
    #             # weapon y/n, spell y/n, cost
    #             s[i + 7] = 1 if p1.hand[j].type == 7 else 0
    #             s[i + 8] = 1 if p1.hand[j].type == 5 else 0
    #             s[i + 9] = p1.hand[j].cost
    #         i += 10
    #     return s

    def getState(self, game_instance):
        """
        Args:
            game_instance: the game object (actual game or deepcopy for MCTS)
        return:
            a 278 length numpy array of features extracted from the
            supplied game.
        """
        # if game_instance == None:
        #     game_instance = self.game

        # s = np.zeros(273, dtype=np.int32)
        # s = np.zeros(282, dtype=np.float32)
        # s = np.zeros(2618, dtype=np.float32) # 282 + 160 + 112 + 112 + 480 + 480 + 496 + 496
        s = np.zeros(2624, dtype=np.float32) # 282 + 160 + 112 + 112 + 480 + 480 + 496 + 496 + 6 for netsize

        p1 = game_instance.current_player
        p2 = p1.opponent

        #0-9 player1 class, we subtract 1 here because the classes are from 1 to 10
        s[p1.hero.card_class-1] = 1
        #10-19 player2 class
        s[10 + p2.hero.card_class-1] = 1
        i = 20
        # 20-21: current health of current player, then opponent
        s[i] = p1.hero.health / 30
        s[i + 1] = p2.hero.health / 30

        # 22: hero power usable y/n
        s[i + 2] = p1.hero.power.is_usable()*1
        # 23-24: # of mana crystals for you opponent
        s[i + 3] = p1.max_mana / 10
        s[i + 4] = p2.max_mana / 10
        # 25: # of crystals still available
        s[i + 5] = p1.mana / 10
        #26-31: weapon equipped y/n, pow., dur. for you, then opponent
        s[i + 6] = 0 if p1.weapon is None else 1                        # TODO weapon id?
        s[i + 7] = 0 if p1.weapon is None else p1.weapon.damage
        s[i + 8] = 0 if p1.weapon is None else p1.weapon.durability

        s[i + 9] = 0 if p2.weapon is None else 1                        # TODO: weapon id?
        s[i + 10] = 0 if p2.weapon is None else p2.weapon.damage
        s[i + 11] = 0 if p2.weapon is None else p2.weapon.durability

        # 32: number of cards in opponents hand
        s[i + 12] = len(p2.hand) / 10
        #in play minions

        i = 33
        #33-102, your monsters on the field
        p1_minions = len(p1.field)
        for j in range(0, 7):
            if j < p1_minions:
                # filled y/n, pow, tough, current health, can attack
                s[i] = 1
                # s[i] = p1.field[j].data.dbf_id                        # TODO: card id normalized or one-hot-encoded, max_attack?
                s[i + 1] = p1.field[j].atk / 20
                s[i + 2] = p1.field[j].max_health / 20
                s[i + 3] = p1.field[j].health / 20
                s[i + 4] = p1.field[j].can_attack()*1
                # deathrattle, div shield, taunt, stealth y/n
                s[i + 5] = p1.field[j].has_deathrattle*1
                s[i + 6] = p1.field[j].divine_shield*1
                s[i + 7] = p1.field[j].taunt*1
                s[i + 8] = p1.field[j].stealthed*1
                s[i + 9] = p1.field[j].silenced*1
            i += 10

        #103-172, enemy monsters on the field
        p2_minions = len(p2.field)
        for j in range(0, 7):
            if j < p2_minions:
                # filled y/n, pow, tough, current health, can attack
                s[i] = 1
                # s[i] = p2.field[j].data.dbf_id                        # TODO: card id normalized or one-hot-encoded
                s[i + 1] = p2.field[j].atk / 20
                s[i + 2] = p2.field[j].max_health / 20
                s[i + 3] = p2.field[j].health / 20
                s[i + 4] = p2.field[j].can_attack()*1
                # deathrattle, div shield, taunt, stealth y/n
                s[i + 5] = p2.field[j].has_deathrattle*1
                s[i + 6] = p2.field[j].divine_shield*1
                s[i + 7] = p2.field[j].taunt*1
                s[i + 8] = p2.field[j].stealthed*1
                s[i + 9] = p2.field[j].silenced*1
            i += 10

        #in hand

        #173-272, your cards in hand
        p1_hand = len(p1.hand)
        for j in range(0, 10):
            if j < p1_hand:
                #card y/n
                s[i] = 1
                # s[i] = p1.hand[j].data.dbf_id                        # TODO: card id normalized or one-hot-encoded
                # minion y/n, attk, hp, battlecry, div shield, deathrattle, taunt
                s[i + 1] = 1 if p1.hand[j].type == 4 else 0
                s[i + 2] = p1.hand[j].atk / 20 if s[i + 1] == 1 else 0
                s[i + 3] = p1.hand[j].health / 20 if s[i + 1] == 1 else 0
                s[i + 4] = p1.hand[j].divine_shield*1 if s[i + 1] == 1 else 0
                s[i + 5] = p1.hand[j].has_deathrattle*1 if s[i + 1] == 1 else 0
                s[i + 6] = p1.hand[j].taunt*1 if s[i + 1] == 1 else 0
                # weapon y/n, spell y/n, cost
                s[i + 7] = 1 if p1.hand[j].type == 7 else 0             # TODO: weapon stats
                s[i + 8] = 1 if p1.hand[j].type == 5 else 0             # TODO: spell stats?
                s[i + 9] = p1.hand[j].cost / 25
            i += 10
        
        # number of crystals still available opponent
        s[273] = p2.mana / 10
        # number of player minions
        s[274] = len(p1.field) / 7
        # number of opponent minions
        s[275] = len(p2.field) / 7
        # number of player cards in hand
        s[276] = len(p1.hand) / 10
        # number of cards left in player deck
        s[277] = len(p1.deck) / 30
        # number of cards left in opponent deck
        s[278] = len(p2.deck) / 30
        # card advantage
        s[279] = 1 if len(p1.hand) > len(p2.hand) else 0
        # board advantage
        s[280] = 1 if len(p1.field) > len(p2.field) else 0
        # turn
        s[281] = game_instance.turn / 180     # normalize turn by maximal possible turn

        i = 282
        # hand cards player (10*16 = 160 bits; len(cardrepo) = 16)
        if len(p1.hand) > 0:
            p1hand_onehot = np.reshape(label_binarize([c.id for c in p1.hand], classes=self.cardrepo), -1)
            for j in range(0, len(p1hand_onehot)):
                s[i + j] = p1hand_onehot[j]          

        i = 442
        # board cards player (7*16 = 112 bits; len(cardrepo) = 16)  # TODO: could remove spells and weapons for minionrepo
        if len(p1.field) > 0:
            p1field_onehot = np.reshape(label_binarize([c.id for c in p1.field], classes=self.cardrepo), -1)
            for j in range(0, len(p1field_onehot)):
                s[i + j] = p1field_onehot[j]

        i = 554
        # board cards opponent (7*16 = 112 bits; len(cardrepo) = 16)    # TODO: could remove spells and weapons for minionrepo
        if len(p2.field) > 0:
            p2field_onehot = np.reshape(label_binarize([c.id for c in p2.field], classes=self.cardrepo), -1)
            for j in range(0, len(p2field_onehot)):
                s[i + j] = p2field_onehot[j]

        i = 666
        # starting deck cards player, sorted (30*16 = 480 bits; len(cardrepo) = 16)      # TODO: could remove the coin for deckrepo
        p1startingdeck = np.sort([c for c in p1.starting_deck])
        if len(p1startingdeck) > 0:
            p1startingdeck_onehot = np.reshape(label_binarize([c for c in p1startingdeck], classes=self.cardrepo), -1)
            for j in range(0, len(p1startingdeck_onehot)):
                s[i + j] = p1startingdeck_onehot[j]

        i = 1146
        # cards left in player deck, sorted (30*16 = 480 bits; len(cardrepo) = 16)
        p1deck = np.sort([c.id for c in p1.deck])
        if len(p1deck) > 0:
            p1deck_onehot = np.reshape(label_binarize([c for c in p1deck], classes=self.cardrepo), -1)
            for j in range(0, len(p1deck_onehot)):
                s[i + j] = p1deck_onehot[j]

        i = 1626
        # played cards player (31*16 = 496 bits; len(cardrepo) = 16)    # max playable cards are deck + coin
        if len(p1.playedcards) > 0:
            p1playedcards_onehot = np.reshape(label_binarize([c for c in p1.playedcards], classes=self.cardrepo), -1)
            for j in range(0, len(p1playedcards_onehot)):
                s[i + j] = p1playedcards_onehot[j]

        i = 2122
        # played cards opponent (31*16 = 496 bits; len(cardrepo) = 16)    # max playable cards are deck + coin
        if len(p2.playedcards) > 0:
            p2playedcards_onehot = np.reshape(label_binarize([c for c in p2.playedcards], classes=self.cardrepo), -1)
            for j in range(0, len(p2playedcards_onehot)):
                s[i + j] = p2playedcards_onehot[j]
        
        i = 2618

        return s

    # def getSymmetries(self, state, pi):
    #     """
    #     Input:
    #         state: current state
    #         pi: policy vector of size self.getActionSize()

    #     Returns:
    #         symmForms: a list of [(actions,pi)] where each tuple is a symmetrical
    #                    form of the actions and the corresponding pi vector. This
    #                    is used when training the neural network from examples.
    #     """
    #     assert(len(pi) == 168)
    #     pi_reshape = np.reshape(pi, (21, 9))
    #     l = []

    #     for i in range(1, 5):
    #         for j in [True, False]:
    #             newS = np.rot90(state, i)
    #             newPi = np.rot90(pi_reshape, i)
    #             if j:
    #                 newS = np.fliplr(newS)
    #                 newPi = np.fliplr(newPi)
    #             l += [(newS, list(newPi.ravel()) + [pi[-1]])]
    #     return l

    def stringRepresentation(self, state):
        """
        Input:
            state: np array of state

        Returns:
            stateString: a quick conversion of state to a string format.
                         Required by MCTS for hashing.
        """
        return state.tostring()

    def cloneAndRandomize(self, game):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player.
        """
        game_copy = copy.deepcopy(game)
        enemy = game_copy.current_player.opponent
        random.shuffle(enemy.hand)                          # Why shuffle? Could be more performant without
        random.shuffle(enemy.deck)                          # Why shuffle? Could be more performant without
        # for idx, card in enumerate(enemy.hand):
        #     if card.id == 'GAME_005':
        #         coin = enemy.hand.pop(idx)
        #
        # combined = enemy.hand + enemy.deck
        # random.shuffle(combined)
        # enemy.hand, enemy.deck = combined[:len(enemy.hand)], combined[len(enemy.hand):]
        # enemy.hand.append(coin)
        return game_copy

    def roguebasic_draft(self):
        """
        Return the basic AI rogue deck (from practice mode)
        """
        # TODO: fireplace.cards.filter(name="Wisp") could help

        deck = []

        # Assassinate
        deck.append('CS2_076')
        deck.append('CS2_076')
        # Backstab
        deck.append('CS2_072')
        deck.append('CS2_072')
        # Bloodfen Raptor
        deck.append('CS2_172')
        deck.append('CS2_172')
        # Deadly Poison
        deck.append('CS2_074')
        deck.append('CS2_074')
        # Dragonling Mechanic
        deck.append('EX1_025')
        deck.append('EX1_025')
        # Elven Archer
        deck.append('CS2_189')
        deck.append('CS2_189')
        # Gnomish Inventor
        deck.append('CS2_147')
        deck.append('CS2_147')
        # Goldshire Footman
        deck.append('CS1_042')
        deck.append('CS1_042')
        # Ironforge Rifleman 	
        deck.append('CS2_141')
        deck.append('CS2_141')
        # Nightblade
        deck.append('EX1_593')
        deck.append('EX1_593')
        # Novice Engineer
        deck.append('EX1_015')
        deck.append('EX1_015')
        # Sap
        deck.append('EX1_581')
        deck.append('EX1_581')
        # Sinister Strike
        deck.append('CS2_075')
        deck.append('CS2_075')
        # Stormpike Commando
        deck.append('CS2_150')
        deck.append('CS2_150')
        # Stormwind Knight
        deck.append('CS2_131')
        deck.append('CS2_131')

        random.shuffle(deck)
        return deck

    cardrepo = ['GAME_005', 'CS2_076', 'CS2_072', 'CS2_172', 'CS2_074', 'EX1_025', 'CS2_189', 'CS2_147', 'CS1_042', 'CS2_141', 'EX1_593', 'EX1_015', 'EX1_581', 'CS2_075', 'CS2_150', 'CS2_131']