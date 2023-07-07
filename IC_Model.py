import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def reading_of_data():

    # read in the data
    movie_reviews = pd.read_csv('Ratings.timed.csv', index_col='date', parse_dates=True).sort_values('date')
    movie_reviews_index = (movie_reviews.copy(deep=True)).reset_index(drop=True)
    network = pd.read_csv('network.txt', delimiter='  ', names=['origin', 'friend'], engine='python')

    network_for_printing = network.copy(deep=True)

    # set final dataframe with origin and number of friends
    full_origin_list_friends = network.groupby('origin')['friend'].unique()
    full_origin_list = full_origin_list_friends.index.tolist()
    full_origin_list_friend_count = full_origin_list_friends.str.len()

    # preprocessing - reducing input data set size
    # only people who have been infected must have gotten it from someone else who has reviewed a movie
    # this removes all people who have not reviewed a movie

    unique_reviewers = movie_reviews['userid'].unique()

    network = network[network['origin'].isin(unique_reviewers)]
    network = network[network['friend'].isin(unique_reviewers)]

    # group movies by all their reviewers
    movie_reviewers = movie_reviews.groupby('movieid')['userid'].unique()

    # find all originators of friends (possible disease originators)
    group_by_origin = network.groupby('friend')['origin'].unique()

    # group people by all the movies they have reviewed
    group_by_origin_movies = movie_reviews.groupby('userid')['movieid'].unique()

    # this algo deployed presumes that preprocess and broadcasting and set manipulation would be more efficient
    # in terms of time complexity as compared to "single pass" with data extrapolation since code using
    # aforementioned functions should be executed on C level.

    # in essence, we are forming two groups: origin of friends and friend's movie's reviewers, and finding the
    # intersection to find which two people have watched the same movie. Set intersection is computationally
    # efficient so this function was selected to cut down the possible infection list to a small list. The finding
    # of what and when they watched will be at a later stage.

    possible_infection_pairs = pd.Series(dtype=int)

    for person in group_by_origin.index.tolist():
        origin_friends = set(group_by_origin[person])
        origin_movies = group_by_origin_movies[person]

        origin_movie_friends = set()
        for movieid in origin_movies:
            origin_movie_friends.update(movie_reviewers[movieid])

        origin_movie_friends_intersect = origin_friends.intersection(origin_movie_friends)
        if len(origin_movie_friends_intersect) != 0:
            pairing = pd.Series([list(origin_movie_friends_intersect)], index=[person])
            possible_infection_pairs = possible_infection_pairs.append(pairing, verify_integrity=True)

    # now we find the movie that both parties watched and check for timing/infection and add if "infection"
    # we are using set intersection manipulation again for speed
    infection = []

    for friend in possible_infection_pairs.index.tolist():
        friend_movies = set(group_by_origin_movies[friend])
        for origin in possible_infection_pairs[friend]:
            origin_movies = set(group_by_origin_movies[origin])
            possible_infection_pair_movies = friend_movies.intersection(origin_movies)
            for movie in possible_infection_pair_movies:
                origin_review_index = (movie_reviews_index.index[(movie_reviews['movieid'] == movie)
                                                                 & (movie_reviews['userid'] == origin)])[0]
                friend_review_index = (movie_reviews_index.index[(movie_reviews['movieid'] == movie)
                                                                 & (movie_reviews['userid'] == friend)])[0]
                if origin_review_index < friend_review_index:
                    infection.append([origin, friend, movie])

    # convert data to dataframe
    infection = pd.DataFrame(infection, columns=['origin', 'friend', 'movie'])

    # count number of times each origin infected their friend, ie A_v2u
    infection = infection.groupby(['origin', 'friend']).size()
    infection = infection.to_frame()
    infection.stack().reset_index()

    origin_review_count = group_by_origin_movies.str.len()
    origin_review_count = origin_review_count.rename_axis('origin')

    result = network_for_printing.join(origin_review_count.to_frame(), on=['origin'])
    result = pd.merge(result, infection, how='outer', left_on=['origin', 'friend'], right_on=['origin', 'friend'])
    result.columns = result.columns.str.strip()
    result.columns = result.columns.fillna('A_v2u')
    result = result.rename(columns={'movieid': 'p*_v2u'})
    result = result.fillna(0)

    result['p*_v2u'] = result['A_v2u'] / result['p*_v2u']
    result = result.fillna(0)
    result = result.drop(['A_v2u'], axis=1)

    result.to_csv('q3_2.txt', index=False)

    # plot distribution to log y-scale
    plt.clf()
    plt.title('Weighted out-degree distribution (log scale)')
    plt.hist(result['p*_v2u'], bins=40)
    plt.yscale('log')
    plt.savefig('q3_3.png')


if __name__ == "__main__":
    reading_of_data()

    # infection.sort_index(ascending=True, inplace=True)

    # # remove the infection series initialization default value
    # infection = infection.iloc[1:]
    #
    # infection_count = infection.str.len()
    # infection_list_count = infection_count.reindex(full_origin_list, fill_value=0)
    #
    # # create final dataframe
    # infection_df = (pd.DataFrame({'origin': full_origin_list,
    #                               'Friend Count': full_origin_list_friend_count.tolist(),
    #                               'Infected': infection_list_count.tolist()})).set_index('origin')
    #
    # # preparing infection_df for merge
    # infection_df['mle'] = infection_df['Infected'] / infection_df['Friend Count']
    # infection_df = infection_df.drop(['Friend Count', 'Infected'], axis=1)
    #
    # # quality control
    # # print(infection_df.head(20))
    # # print(infection_df['Infected'].sum())
    #
    # result = pd.merge(network_for_printing, infection_df, how='outer', on=['origin'])
    # result.to_csv('q3_2.txt', index=False)
    #
    # # plot distribution to log y-scale
    # plt.clf()
    # plt.title('Weighted out-degree distribution (log scale)')
    # plt.hist(infection_df['mle'], bins=20)
    # plt.yscale('log')
    # plt.savefig('q3_3.png')