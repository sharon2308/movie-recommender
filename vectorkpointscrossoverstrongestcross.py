from random import sample

from eckity.genetic_operators.genetic_operator import GeneticOperator


class VectorKPointsCrossoverStrongestCross(GeneticOperator):
    def __init__(self, probability=1, arity=2, events=None, moviesScores=[], lowerBound=1.5):
        """
            Vector N Point Mutation.

            Randomly chooses N vector cells and performs a small change in their values.

            Parameters
            ----------
            probability : float
                The probability of the mutation operator to be applied

            arity : int
                The number of individuals this mutation is applied on

            k : int
                Number of points to cut the vector for the crossover.

            events: list of strings
                Events to publish before/after the mutation operator
        """
        self.individuals = None
        self.applied_individuals = None
        self.k = 1
        self.points = None
        self.moviesScores = moviesScores
        self.lowerBound = lowerBound
        super().__init__(probability=probability, arity=arity, events=events)

    def individual_fitness_sum(self, start, end, sum_init, individuals, individuals_index):
        for i in range(start, end):
            movieScore = self.moviesScores[i]
            if movieScore >= self.lowerBound:
                sum_init += individuals[individuals_index].get_vector()[i] * movieScore
            else:
                sum_init += individuals[individuals_index].get_vector()[i] * -1 * (2 - movieScore)
        return sum_init

    def apply(self, individuals):
        """
        Attempt to perform the mutation operator

        Parameters
        ----------
        individuals : list of individuals
            individuals to perform crossover on

        Returns
        ----------
        list of individuals
            individuals after the crossover
        """
        self.individuals = individuals
        # self.points = [int(len(self.individuals[0].get_vector())/2)]
        self.points = sorted(sample(range(0, individuals[0].size()), self.k))


        start_index = 0
        for end_point in self.points:

            original_sum1 = self.individual_fitness_sum(0, len(individuals[0].get_vector()), 0, individuals, 0)
            original_sum2 = self.individual_fitness_sum(0, len(individuals[1].get_vector()), 0, individuals, 1)
            max_original = max(original_sum2, original_sum1)

            sum1 = self.individual_fitness_sum(start_index, end_point, 0, individuals, 0)
            sum1 = self.individual_fitness_sum(end_point, start_index, sum1, individuals, 1)
            sum2 = self.individual_fitness_sum(start_index, end_point, 0, individuals, 1)
            sum2 = self.individual_fitness_sum(end_point, start_index, sum2, individuals, 0)

            if max_original > sum1 and max_original > sum2:
                if original_sum1 > original_sum2:
                    replaced_part = individuals[0].get_vector()
                    individuals[1].replace_vector_part(replaced_part, 0)
                else:
                    replaced_part = individuals[1].get_vector()
                    individuals[0].replace_vector_part(replaced_part, 0)

            elif sum1 > sum2:
                replaced_part = individuals[0].get_vector_part(start_index, end_point)
                replaced_part = individuals[1].replace_vector_part(replaced_part, start_index)
                individuals[0].replace_vector_part(replaced_part, start_index)

            else:
                replaced_part = individuals[1].get_vector_part(start_index, end_point)
                replaced_part = individuals[0].replace_vector_part(replaced_part, start_index)
                individuals[1].replace_vector_part(replaced_part, start_index)
            start_index = end_point

        self.applied_individuals = individuals
        return individuals
