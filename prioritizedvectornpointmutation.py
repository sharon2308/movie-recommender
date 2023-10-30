from random import random

from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation

class PrioritizedVectorNPointMutation(VectorNPointMutation):
    """
    N Point Bit-Flip Mutation
    """
    def __init__(self, probability=1.0, arity=1, events=None, probability_for_each=0.2, n=1, moviesScores=[], lowerBound=1.5):
        self.probability_for_each = probability_for_each
        self.movieScores = moviesScores
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, 
                         index: individual.bit_flip(index) if 
                         random() <= self.probability_for_each and (individual.cell_value(index) == 1 and (individual.cell_value(index)*moviesScores[index] < lowerBound)) 
                         or (individual.cell_value(index) == 0 and (individual.cell_value(index) * moviesScores[index] >= lowerBound))
                         else individual.cell_value(index),
                         events=events,
                         n=n)
