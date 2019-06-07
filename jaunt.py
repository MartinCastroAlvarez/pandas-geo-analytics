"""
This library is the solution for the OpenJaunt challenge.

Scope:

The goal of the program is to parse a list of such restaurants
and print out to the command line (stdout) answers to the
following questions. A list of sample restaurants is sent as a
separate file, so you can test your code with them. Assume each
question has exactly one valid answer.

1. What is the number of unique restaurants present in the file?
2. Which two restaurants are furthest apart?  which two are closest?
   What are the distances?
3. Which restaurants mention menu items in the `tips` section costing
   more than $10?
4. Classify each restaurant into one of the following two categories
   using any technique you prefer:
    Category 1: Restaurants known for drinks
    Category 2: Restaurants known for food

References:
- https://www.tutorialspoint.com/Tokenize-text-using-NLTK-in-python
- https://gist.github.com/jovianlin/805189d4b19332f8b8a79bbd07e3f598
- https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html
- https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/
- https://stackoverflow.com/questions/19626737
- https://stackoverflow.com/questions/43909954
- https://stackoverflow.com/questions/19412462
- https://stackoverflow.com/questions/40452759
- https://stackoverflow.com/questions/20303323
"""

import re
import begin
import logging
import os
import typing

import pandas as pd
import numpy as np

from slugify import slugify
from textblob import TextBlob

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  # Required by NLTK tokenized.
nltk.download('wordnet')  # Required by food extractor.

logger = logging.getLogger(__name__)

# The following functions should be placed inside a utils.py file.

EARTH_RADIUS = 6371


def sigmoid(x: float) -> float:
    """
    Apply the math sigmoid activation function.

    @raises: TypeError.
    """
    if not isinstance(x, (int, float)):
        raise TypeError("Expecting number, got:", type(x))
    return 1 / (1 + np.exp(-x))


def get_polarity(text: str) -> float:
    """
    Detects the polarity of one text.

    @raises: TypeError.
    """
    if not isinstance(text, str):
        raise TypeError("Invalid text:", type(text))
    return TextBlob(text).sentiment.polarity


def get_distance(x_lat: float,
                 x_long: float,
                 y_lat: float,
                 y_long: float) -> float:
    """
    A helper function to compute distance of two points.

    @raises: TypeError.
    """
    logger.debug("Calculating distance between (%s, %s) and (%s, %s)",
                 x_lat, x_long, y_lat, y_long)
    if not isinstance(x_lat, (int, float)):
        raise TypeError("Invalid $x_lat:", x_lat)
    if not isinstance(y_lat, (int, float)):
        raise TypeError("Invalid $y_lat:", y_lat)
    if not isinstance(x_long, (int, float)):
        raise TypeError("Invalid $x_long:", x_long)
    if not isinstance(y_long, (int, float)):
        raise TypeError("Invalid $y_long:", y_long)
    x_lat = np.radians(x_lat)
    y_lat = np.radians(y_lat)
    x_long = np.radians(x_long)
    y_long = np.radians(y_long)
    delta_lat = np.sin((y_lat - x_lat) / 2) ** 2
    delta_long = np.sin((y_long - x_long) / 2) ** 2
    tmp = delta_lat + np.cos(x_lat) * np.cos(y_lat) * delta_long
    return EARTH_RADIUS * 2 * np.arcsin(np.sqrt(tmp))


def get_text_score(text: str,
                   corpora: set,
                   lemmatizer: WordNetLemmatizer) -> float:
    """
    This function is to be applied to a Pandas cell.
    It computes the score based on a dict of words.

    @raises: TypeError.
    """
    if not isinstance(text, str):
        raise TypeError("Invalid text:", type(text))
    if not isinstance(corpora, set):
        raise TypeError("Invalid corpora:", type(corpora))
    if not isinstance(lemmatizer, WordNetLemmatizer):
        raise TypeError("Invalid lemmatizer:", type(lemmatizer))
    tokens = {
        lemmatizer.lemmatize(token.lower())
        for token in word_tokenize(text)
    }
    match = corpora & tokens
    if match:
        return 100 * len(match) / len(tokens)
    return 0


def find_expensive_items(text: str) -> float:
    """
    This function is to be applied to a Pandas cell.
    It extracts expensive items from a text.

    @raises: TypeError.
    """
    if not isinstance(text, str):
        raise TypeError("Invalid text:", type(text))
    match = re.search("^.*\$(\d+\.?\d*).*$", text)
    if match:
        return float(match.group(1))
    return 0


# Main class:


class Dataset(object):
    """
    Restaurants Dataset entity.

    NOTE: Given the amount of data, all restaurants
    will be loaded and transformed in memory.
    """

    class Attribute(object):
        """
        Dataset attribute entity.
        """
        PLACE = "Place"
        ADDRESS = "Address"
        LATITUDE = "Latitude"
        LONGITUDE = "Longitude"
        TIPS = "Tips"

    class MetaAttribute(object):
        """
        Dataset meta-attributes derived from Attributes.
        """
        PLACE_TITLE = "Place_Title"
        PLACE_SLUG = "Place_Slugified"
        TIPS_SIGMOD_POLARITY = "Tips_SigmoidPolarity"
        TIPS_EXPENSIVE_SCORE = "Tips_ExpensiveScore"
        TIPS_FOOD_SCORE = "Tips_FoodScore"
        TIPS_DRINKS_SCORE = "Tips_DrinkScore"
        PLACE_FROM = "Place_From"
        PLACE_TO = "Place_To"
        DISTANCE = "Distance"

    def __init__(self, path: str) -> None:
        """
        Constructing dataset.

        @param path: OS Path of the CSV file.

        @raises: ValueError, TypeError, OSError.
        """
        logger.debug("Constructing dataset from: %s.", path)
        if not path:
            raise ValueError("Path is required.")
        if not isinstance(path, str):
            raise ValueError("Expecting string path, got:", type(path))
        if not os.path.isfile(path):
            raise OSError("File not found:", path)
        self.__path = path
        self.__distances = None
        self.__dataset = pd.DataFrame.from_csv(self.__path,
                                               index_col=None)

    def __str__(sefl) -> str:
        """
        String serializer.
        """
        return "<Dataset: '{}'>".format(self.__path)

    def set_title(self) -> None:
        """
        Creates a new column that contains
        the title of the restaurant.
        """
        logger.debug("Generating restaurant title.")
        func = lambda x: re.sub("\(.*$", "", x)\
            .encode('ascii', 'ignore').decode()
        col = self.__dataset[self.Attribute.PLACE].apply(func)
        self.__dataset[self.MetaAttribute.PLACE_TITLE] = col

    def set_slugify_names(self) -> None:
        """
        Creates a new column that contains the
        slugified title of each restaurant.
        """
        logger.debug("Generating slugified restaurant name.")
        col = self.__dataset[self.MetaAttribute.PLACE_TITLE].apply(slugify)
        self.__dataset[self.MetaAttribute.PLACE_SLUG] = col

    def set_tips_food_score(self) -> None:
        """
        Creates a new column that contains
        the score based on food mentioned in the tips.

        The food score is calculated based on the amount
        of words in the tips, the amount of words that
        represent food and the polarity of the tip.
        In other words, negative tips reduce the score
        even if the tip contains lots of foods.

        This post suggests there are more corpora to
        use besides 'food.n.02'. However, things are
        kept simple because this is just a demo:
        https://github.com/JoshRosen/\
            cmps140_creative_cooking_assistant/blob/master/nlu/ingredients.py
        """
        logger.debug("Detecting food in tips.")
        lemmatizer = WordNetLemmatizer()
        food = {
            w.lower()
            for s in wn.synset('food.n.02').closure(lambda s: s.hyponyms())
            for w in s.lemma_names()
        }
        col = self.__dataset[self.Attribute.TIPS].apply(get_text_score,
                                                        corpora=food,
                                                        lemmatizer=lemmatizer)
        col *= self.__dataset[self.MetaAttribute.TIPS_SIGMOD_POLARITY]
        self.__dataset[self.MetaAttribute.TIPS_FOOD_SCORE] = col

    def set_restaurant_distances(self) -> None:
        """
        Creates a new column that contains
        the furhest and nearest restaurant.
        """
        logger.debug("Calculating restaurant distances.")
        d = self.__dataset.drop_duplicates(self.MetaAttribute.PLACE_SLUG)
        d_matrix = (
            (
                x[self.MetaAttribute.PLACE_SLUG],
                y[self.MetaAttribute.PLACE_SLUG],
                get_distance(x_lat=x[self.Attribute.LATITUDE],
                             y_lat=y[self.Attribute.LATITUDE],
                             x_long=x[self.Attribute.LONGITUDE],
                             y_long=y[self.Attribute.LONGITUDE])
            )
            for id_x, x in d.iterrows()
            for id_y, y in d.iterrows()
            # This triangulates the distances matrix:
            if id_y > id_x
            # This removes the diagonals of the distances matrix:
            and x[self.MetaAttribute.PLACE_SLUG] != y[self.MetaAttribute.PLACE_SLUG]
        )
        # Generating a new Pandas DataFrame.
        cols = [
            self.MetaAttribute.PLACE_FROM,
            self.MetaAttribute.PLACE_TO,
            self.MetaAttribute.DISTANCE,
        ]
        self.__distances = pd.DataFrame(d_matrix, columns=cols)

    def set_tips_drinks_score(self) -> None:
        """
        Creates a new column that contains
        the score based on DRINKS mentioned in the tips.

        This function is similar to the foods score but
        based on drinks mentioned in the tips.

        This function may force the algorithm to
        calculate again the tokenized version of the text
        but that way things are decoupled and each
        function can be later put into a different thread,
        taking advantage of parallelism.
        """
        logger.debug("Detecting drinks in tips.")
        lemmatizer = WordNetLemmatizer()
        drinks = {
            w.lower()
            for s in wn.synset('drink.n.01').closure(lambda s: s.hyponyms())
            for w in s.lemma_names()
        }
        col = self.__dataset[self.Attribute.TIPS].apply(get_text_score,
                                                        corpora=drinks,
                                                        lemmatizer=lemmatizer)
        col *= self.__dataset[self.MetaAttribute.TIPS_SIGMOD_POLARITY]
        self.__dataset[self.MetaAttribute.TIPS_DRINKS_SCORE] = col

    def set_polarity(self) -> None:
        """
        Creates a new column that contains
        the polarity of the tips.

        Polarity goes from -1 to 1.
        -1 indicates a negative opinion.
        1 indicates a positive opinon about the restaruant.

        Polarity is activated using the sigmoid function
        in order to normalize the value between 0 and 1.
        """
        logger.debug("Calculating polarity in tips.")
        col = self.__dataset[self.Attribute.TIPS].apply(get_polarity)
        col = col.apply(sigmoid)
        self.__dataset[self.MetaAttribute.TIPS_SIGMOD_POLARITY] = col

    def set_expensive_menu_items(self) -> None:
        """
        Creates a new column that contains
        True or False if they contain expensive products.
        """
        logger.debug("Calculating expensive products in tips.")
        col = self.__dataset[self.Attribute.TIPS].apply(find_expensive_items)
        self.__dataset[self.MetaAttribute.TIPS_EXPENSIVE_SCORE] = col

    def get_total_restaurants(self) -> int:
        """
        This method returns the amount of
        unique restaurants in the dataset.
        """
        logger.debug("Calculating total amount of restaurants")
        return len(self.__dataset)

    def get_total_unique_restaurants(self) -> int:
        """
        This method returns the amount of
        unique restaurants in the dataset.
        """
        logger.debug("Calculating total amount of unique restaurants")
        return len(self.__dataset[self.MetaAttribute.PLACE_SLUG].unique())

    def get_maximum_distance(self) -> np.ndarray:
        """
        This method returns the largest distance
        between 2 restaurants in the dataset.

        @raises: AttributeError:
        """
        logger.debug("Finding the maximum distance.")
        if self.__distances is None:
            raise AttributeError("Distances not calculated.")
        print(self.__distances)
        raise Exception(3)

    def get_expensive_restaurants(self,
                                  threshold: float=1,
                                  limit: int=10) -> np.ndarray:
        """
        This method returns those restaurants
        marked as expensive as per the comment in the tips.

        @raises: TypeError, ValueError.
        """
        logger.debug("Sorting restaurants by expensive score")
        if not isinstance(threshold, (float, int)):
            raise TypeError("Expeting numeric threshold, got:", type(threshold))
        if not isinstance(limit, int):
            raise TypeError("Expeting int limit, got:", type(limit))
        if limit < 1:
            raise ValueError("Limit is too low.")
        sort_by = self.MetaAttribute.TIPS_EXPENSIVE_SCORE
        cols = [self.MetaAttribute.PLACE_TITLE, sort_by]
        mask = self.__dataset[self.MetaAttribute.TIPS_EXPENSIVE_SCORE] \
            >= threshold
        query = self.__dataset[mask][cols]
        query = query.sort_values(sort_by, ascending=False)
        return query.head(limit)

    def get_best_restaurants_by_food(self, limit: int=10) -> np.ndarray:
        """
        This method returns the best restaurants
        based on the food score.

        @raises: TypeError, ValueError.
        """
        logger.debug("Sorting restaurants by food score")
        if not isinstance(limit, int):
            raise TypeError("Expeting int limit, got:", type(limit))
        if limit < 1:
            raise ValueError("Limit is too low.")
        sort_by = self.MetaAttribute.TIPS_FOOD_SCORE
        cols = [self.MetaAttribute.PLACE_TITLE, sort_by]
        query = self.__dataset[cols]
        query = query.sort_values(sort_by, ascending=False)
        return query.head(limit)

    def get_best_restaurants_by_drinks(self, limit: int=10) -> np.ndarray:
        """
        This method returns the best restaurants
        based on the drinks score.

        @raises: TypeError, ValueError.
        """
        logger.debug("Sorting restaurants by drinks score")
        if not isinstance(limit, int):
            raise TypeError("Expeting int limit, got:", type(limit))
        if limit < 1:
            raise ValueError("Limit is too low.")
        sort_by = self.MetaAttribute.TIPS_DRINKS_SCORE
        cols = [self.MetaAttribute.PLACE_TITLE, sort_by]
        query = self.__dataset[cols]
        query = query.sort_values(sort_by, ascending=False)
        return query.head(limit)

    def get_furthest_restaurants(self, limit: int=3) -> np.ndarray:
        """
        Filters restaurants by the furthest distance.

        @raises: TypeError, ValueError, AttributeError.
        """
        logger.debug("Sorting restaurants by largest distance.")
        if not isinstance(limit, int):
            raise TypeError("Expeting int limit, got:", type(limit))
        if limit < 1:
            raise ValueError("Limit is too low.")
        if self.__distances is None:
            raise AttributeError("Distances matrix not calculated.")
        sort_by = self.MetaAttribute.DISTANCE
        cols = [
            self.MetaAttribute.PLACE_TO,
            self.MetaAttribute.PLACE_FROM,
            sort_by
        ]
        query = self.__distances[cols]
        query = query.sort_values(sort_by, ascending=False)
        return query.head(limit)

    def get_nearest_restaurants(self, limit: int=3) -> np.ndarray:
        """
        Filters restaurants by the shortest distance.

        @raises: TypeError, ValueError, AttributeError.
        """
        logger.debug("Sorting restaurants by shorest distance.")
        if not isinstance(limit, int):
            raise TypeError("Expeting int limit, got:", type(limit))
        if limit < 1:
            raise ValueError("Limit is too low.")
        if self.__distances is None:
            raise AttributeError("Distances matrix not calculated.")
        sort_by = self.MetaAttribute.DISTANCE
        cols = [
            self.MetaAttribute.PLACE_TO,
            self.MetaAttribute.PLACE_FROM,
            sort_by
        ]
        query = self.__distances[cols]
        query = query.sort_values(sort_by, ascending=True)
        return query.head(limit)


class Task(object):
    """
    ETL task to be performed.
    """

    SEPARATOR = "-" * 30

    def __init__(self, path: str) -> None:
        """
        Constructing Task.
        """
        self.__path = path
        self.__dataset = None

    def extract(self) -> None:
        """
        Extract data from datasource.

        @raises: AttributeError.
        """
        logger.debug("Extracting data from datasource.")
        if not self.__path:
            raise AttributeError("Path not initialized.")
        self.__dataset = Dataset(self.__path)

    def transform(self) -> None:
        """
        Transforming data: Data cleaning, feature extraction, etc.

        @raises: AttributeError.
        """
        logger.debug("Transforming data.")
        if not self.__dataset:
            raise AttributeError("Dataset not initialized.")
        self.__dataset.set_title()
        self.__dataset.set_slugify_names()
        self.__dataset.set_restaurant_distances()
        self.__dataset.set_polarity()
        self.__dataset.set_expensive_menu_items()
        self.__dataset.set_tips_food_score()
        self.__dataset.set_tips_drinks_score()

    def load(self, limit: int=5) -> None:
        """
        Loading results into destination.

        @param limit: Amount of data to show.

        NOTE: Since this is a demo, the destintaion is stdout.
        """
        logger.debug("Loading data.")
        if not isinstance(limit, int):
            limit = int(limit)
        print(self.SEPARATOR)
        print("1.1: Total Restaurants:", self.__dataset.get_total_restaurants())
        print("1.2: Unique Restaurants:", self.__dataset.get_total_unique_restaurants())
        print(self.SEPARATOR)
        print("2.1: Furthest Restaurants:")
        print(self.__dataset.get_furthest_restaurants(limit=limit))
        print(self.SEPARATOR)
        print("2.2: Nearest Restaurants:")
        print(self.__dataset.get_nearest_restaurants(limit=limit))
        print(self.SEPARATOR)
        print("3: Expensive Restaurants:")
        print(self.__dataset.get_expensive_restaurants(limit=limit,
                                                       threshold=10))
        print(self.SEPARATOR)
        print("4.1: Restaurants by Food:")
        print(self.__dataset.get_best_restaurants_by_food(limit=limit))
        print(self.SEPARATOR)
        print("4.2: Restaurants by Drinks:")
        print(self.__dataset.get_best_restaurants_by_drinks(limit=limit))
        print(self.SEPARATOR)


@begin.start(lexical_order=True, short_args=True)
def run(debug: "Run script in debug mode."=False,
        limit: "Maximum amount of results to show."=5,
        path: "Datasource path."="./restaurants.csv"):
    """
    This method will be called by executing this script from the CLI.

    If script is executed in debug mode, logger
    messages will be printed to STDOUT.
    """
    if debug:
        logger.warning("Script is running in DEBUG mode.")
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
    try:
        t = Task(path)
        t.extract()
        t.transform()
        t.load(limit=limit)
    except:
        logger.exception("Something went wrong.")
        raise
    else:
        logger.debug("Finished running ETL job succesfully.")
    finally:
        logger.debug("End of ETL job.")
