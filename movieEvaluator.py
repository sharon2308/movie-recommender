from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

languagesList = {}
genresList = {}


class movieEvaluator(SimpleIndividualEvaluator):
    

    def __init__(self, moviesScores, lowerBound):
        super().__init__()
        #list of all the movies objects generated from the api
        self.moviesScores = moviesScores
        self.lowerBound = lowerBound


    def _evaluate_individual(self, individual):
        individualVector = individual.vector
        sum = 0

        for i in range(len(individualVector)):
            movieScore = self.moviesScores[i]
            if movieScore >= self.lowerBound:
                sum += individualVector[i] * movieScore
            else:
                sum += individualVector[i] * -1 *  (2 - movieScore)
    
        return sum

   