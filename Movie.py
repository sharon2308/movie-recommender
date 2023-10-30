class Movie:
    GENRES_MAP = dict()
    GENRES = set()
    LANGUAGES = set()

    def __init__(self, genres, language, timeInMinutes, year, title, imdb_rank):
        self.genres = genres
        self.language = language
        Movie.LANGUAGES.add(self.language)
        Movie.GENRES.update(self.genres)
        self.timeInMinutes = timeInMinutes
        self.year = year
        self.title = title
        self.imdb_rank = imdb_rank

    def __str__(self):
        return "Movie title: " + self.title + \
               "\nGenres: " + str(self.genres) + "\nLanguage: " + self.language + "\nTime in minutes: " + str(self.timeInMinutes) + "\nYear: " + str(self.year) + "\nIMDB ranking: " + str(self.imdb_rank)

    def json(self):
        return {"title": self.title, "genres": self.genres, "language": self.language, "timeInMinutes": self.timeInMinutes, "year": self.year}

    def setGenres(genres):
        Movie.GENRES_MAP = genres

