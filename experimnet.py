import json

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorNFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker

from Movie import Movie
from api import MoviesApi
from prioritizedvectornpointmutation import PrioritizedVectorNPointMutation
from movieEvaluator import movieEvaluator
from vectorkpointscrossoverstrongestcross import VectorKPointsCrossoverStrongestCross


MAX_GENERATION = 300


def main():
    movies = MoviesApi.loadMovies()
    num_of_movies = len(movies)
    lower_bound_grade = 1.5
    finish_all = False

    while not finish_all:
        movies_scores = get_user_req_and_generate_movie_scores(movies)
        max_fitness = 0
        matched_movies = 0
        for movie_score in movies_scores:
            if movie_score >= lower_bound_grade:
                max_fitness += movie_score
                matched_movies += 1
        print("\nmax_fitness:" + str(max_fitness))
        #print("matched_movies:" + str(matched_movies))
        threshold = 0.3 * max_fitness

        algo = SimpleEvolution(
            Subpopulation(creators=GABitStringVectorCreator(length=num_of_movies),
                          population_size=300,
                          # user-defined fitness evaluation method with the lower bound of matching criteria for each movie
                          evaluator=movieEvaluator(movies_scores, lower_bound_grade),
                          # higher fitness = better recomandtion
                          higher_is_better=True,
                          elitism_rate=5/300,
                          # genetic operators sequence to be applied in each generation
                          operators_sequence=[
                              # VectorKPointsCrossover(probability=0.5, k=1),
                              VectorKPointsCrossoverStrongestCross(probability=0.5, arity=2, events=None, moviesScores=movies_scores,
                                                                   lowerBound=lower_bound_grade),
                              PrioritizedVectorNPointMutation(probability=0.2, probability_for_each=0.02, n=num_of_movies, moviesScores=movies_scores, lowerBound=lower_bound_grade)
                          ],
                          selection_methods=[
                              # tuple-> (selection method, selection probability)
                              (TournamentSelection(tournament_size=2, higher_is_better=True), 1)
                          ]
                          ),
            breeder=SimpleBreeder(),
            max_workers=4,
            max_generation=MAX_GENERATION,
            termination_checker=ThresholdFromTargetTerminationChecker(optimal=max_fitness, threshold=threshold),
            statistics=BestAverageWorstStatistics())

        algo.evolve()
        result = algo.execute()
        if result.count(1) < 1:
            print("Sadly, there are not enough recommendations for you...")
            print("Please consider modifying some of your choices. Thank you.")
            continue
        else:
            finish_all = True

    print("\nRecommendations for you:")
    rec = []
    counter=1
    for i in range(len(movies)):
        if result[i]:
            print(str(counter) + ". " + movies[i].title)
            counter += 1
            rec.append(movies[i].json())

    print("Number of movies in database: " + str(len(movies)))
    print("Number of recommended movies in database: " + str(len(rec)))
    json_res = json.dumps({"Recommendations": rec}, indent=4)
    with open("Recommendations.json", "w") as outfile:
        outfile.write(json_res)


def get_user_req_and_generate_movie_scores(movies):
    criterionsSize = {"genres": float, "language": float, "year": float, "timeInMinutes": float}
    print(
        "Please rate the importance of each criterion to you or choose the default rating option.\n"
        "To do so, you should assign a value from 0 to 1 for each criterion, ensuring that the total sum of all criteria is 1.")
    finish = "y" == str(input("Do you want to use the default rate? y/n\n"))

    if finish:
        criterionsSize["genres"] = 0.6
        criterionsSize["language"] = 0.2
        criterionsSize["year"] = 0.1
        criterionsSize["timeInMinutes"] = 0.1

    while not finish:
        genres_rate = float(input("Please indicate the level of importance you assign to the genres criterion:"))
        language_rate = float(input("Please indicate the level of importance you assign to the language criterion:"))
        year_rate = float(input("Please indicate the level of importance you assign to the minimum publish year criterion:"))
        time_in_minutes_rate = float(input("Please indicate the level of importance you assign to the max length of a movie criterion:"))
        total_rate = genres_rate + language_rate + year_rate + time_in_minutes_rate

        if total_rate == 1.0:
            finish = True
            criterionsSize["genres"] = genres_rate
            criterionsSize["language"] = language_rate
            criterionsSize["year"] = year_rate
            criterionsSize["timeInMinutes"] = time_in_minutes_rate
        else:
            print("The sum of all criteria together is not 1! Please try again")
            continue

    user_request = {"genres": set(), "languages": set(), "year": None, "timeInMinutes": None}
    print("We are ready to go!!!")
    print("Please enter the key corresponding to the genre from the following list:\n")
    for g in list(Movie.GENRES):
        print(str(g) + ". " + Movie.GENRES_MAP[str(g)])

    finish = False
    while not finish:
        genres_choice = input("To finish, enter 'finish'.\n")
        finish = genres_choice == "finish"
        if finish and not user_request["genres"]:
            print("No genres have been entered. Please try again.\n")
            finish = False
            continue
        if not finish:
            user_request["genres"].add(int(genres_choice))

    print("\nPlease select your preferred languages from the following list: \n")
    for l in Movie.LANGUAGES:
        print(l)
    finish = False
    while not finish:
        languages_choice = input("To finish, enter 'finish'.\n")
        finish = languages_choice == "finish"
        if finish and not user_request["languages"]:
            print("No languages have been entered. Please try again.")
            finish = False
            continue
        if not finish:
            user_request["languages"].add(languages_choice)

    user_request['year'] = int(input("Please specify the minimum publication year for the movie:\n"))
    user_request['timeInMinutes'] = int(input("Please select your preferred maximum movie length (in minutes):\n"))

    return grading_movies(movies, user_request, criterionsSize)


def grading_movies(movies, user_request, criterions_size):
    genre_weight = criterions_size["genres"] / len(user_request["genres"])
    movies_scores = []
    for movie in movies:
        score = 0
        # Sum all the matching genres
        for genre in user_request["genres"]:
            if genre in movie.genres:
                score += genre_weight

        # If any of the requested languages match, add the language score.
        for language in user_request["languages"]:
            if language in movie.language:
                score += criterions_size["language"]
                break

        if user_request["year"] <= movie.year:
            score += criterions_size["year"]

        if user_request["timeInMinutes"] >= movie.timeInMinutes:
            score += criterions_size["timeInMinutes"]

        movies_scores.append(score + (movie.imdb_rank / 100))

    return movies_scores


if __name__ == '__main__':
    main()
