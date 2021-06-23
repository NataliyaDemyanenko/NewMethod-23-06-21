#!/usr/bin/env python3
import math
from math import *
import matplotlib.pyplot as plt
import itertools
from itertools import permutations, combinations
from math import factorial as fac


def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom

def powerset(s):
     x = len(s)
     masks = [1 << i for i in range(x)]
     for i in range(1 << x):
         yield [ss for mask, ss in zip(masks, s) if i & mask]



class CooperativeGame():
     def __init__(self,characteristic_function): 
         if type(characteristic_function) is not dict:
             raise TypeError("characteristic function must be a dictionary")

         self.ch_f = characteristic_function
         for key in self.ch_f:
             if len(str(key)) == 1 and type(key) is not tuple:
                 self.ch_f[(key,)] = self.ch_f.pop(key)
             elif type(key) is not tuple:
                 raise TypeError("key must be a tuple")
         for key in self.ch_f:
             sortedkey = tuple(sorted(list(key)))
             self.ch_f[sortedkey] = self.ch_f.pop(key)

         self.player_list = max(characteristic_function.keys(), key=lambda key: len(key))
 
         self.number_players = len(self.player_list)

     def shapley_value(self):
        payoff_vector = {}
        n = int(len(self.player_list))
        for player in self.player_list:
            weighted_contribution = 0
            for coalition in powerset(self.player_list):
                if coalition:  # If non-empty
                    k = int(len(coalition))
                    weight = 1 / (binomial(n,k) * k)
                    t = tuple(p for p in coalition if p != player)
                    weighted_contribution += weight * (self.ch_f[tuple(coalition)]
                                                       - self.ch_f[t])
            payoff_vector[player] = weighted_contribution

        return payoff_vector

def is_monotone(self):
        return not any([set(p1) <= set(p2) and self.ch_f[p1] > self.ch_f[p2]
                        for p1, p2 in permutations(self.ch_f.keys(), 2)])

def is_superadditive(self):
        sets = self.ch_f.keys()
        for p1, p2 in combinations(sets, 2):
            if not (set(p1) & set(p2)):
                union = tuple(sorted(set(p1) | set(p2)))
                if self.ch_f[union] < self.ch_f[p1] + self.ch_f[p2]:
                    return False
        return True

def _repr_(self):        
    return "A {} player co-operative game".format(self.number_players)

def _latex_(self):
     cf = self.ch_f
     output = "v(c) = \\begin{cases}\n"
     for key in sorted(cf.keys(), key=lambda key: len(key)):
         if not key:  # == ()
            coalition = "\\emptyset"
         else:
            coalition = "\\{" + ", ".join(str(player) for player in key) + "\\}"
         output += "{}, &amp; \\text{{if }} c = {} \\\\\n".format(cf[key], coalition)
     output += "\\end{cases}"
     return output

def is_efficient(self, payoff_vector):
         pl = tuple(sorted(list(self.player_list)))
         return sum(payoff_vector.values()) == self.ch_f[pl]

def nullplayer(self, payoff_vector):
         for player in self.player_list:
             results = []
             for coalit in self.ch_f:
                 if player in coalit:
                     t = tuple(sorted(set(coalit) - {player}))
                     results.append(self.ch_f[coalit] == self.ch_f[t])
             if all(results) and payoff_vector[player] != 0:
                 return False
         return True

def is_symmetric(self, payoff_vector):
         sets = self.ch_f.keys()
         element = [i for i in sets if len(i) == 1]
         for c1, c2 in combinations(element, 2):
             results = []
             for m in sets:
                 junion = tuple(sorted(set(c1) | set(m)))
                 kunion = tuple(sorted(set(c2) | set(m)))
                 results.append(self.ch_f[junion] == self.ch_f[kunion])
             if all(results) and payoff_vector[c1[0]] != payoff_vector[c2[0]]:
                 return False
         return True
