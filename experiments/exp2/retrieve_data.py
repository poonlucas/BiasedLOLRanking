import numpy as np
import requests
import csv

import config  # stores the api keys and puuids (omitted on github)


def get_feature_matrix(matchid):
    api_url = "https://americas.api.riotgames.com/lol/match/v5/matches/NA1_" + matchid
    api_url += "?api_key=" + config.api_key
    match_data = requests.get(api_url).json()['info']

    # Find which team we are on first
    found = False
    teamId = 0
    for i in range(0, 10):  # loop through all 10 players until we find someone that is on our team
        if not found:
            for puuid in config.puuids:
                if match_data.get('participants')[i].get('puuid') == puuid:
                    found = True
                    teamId = match_data.get('participants')[i].get('teamId')

    # the players are sorted, 0-4 is teamId = 100, 5-9 is teamId = 200
    offset = 0
    if teamId == 200:
        offset = 5
    participants = match_data.get('participants')

    # create feature matrix
    feature_matrix = np.zeros(shape=(5, 6), dtype=np.int64)
    for i in range(0, 5):
        feature_matrix[i][0] = participants[i + offset].get('kills')  # kills
        feature_matrix[i][1] = participants[i + offset].get('deaths')  # deaths
        feature_matrix[i][2] = participants[i + offset].get('assists')  # assists
        feature_matrix[i][3] = participants[i + offset].get('goldEarned')  # goldEarned
        feature_matrix[i][4] = match_data.get('gameDuration')  # gameDuration
        # teamPosition, we will convert into one hot vector later
        if participants[i + offset].get('teamPosition') == 'TOP':
            feature_matrix[i][5] = 0
        elif participants[i + offset].get('teamPosition') == 'JUNGLE':
            feature_matrix[i][5] = 1
        elif participants[i + offset].get('teamPosition') == 'MIDDLE':
            feature_matrix[i][5] = 2
        elif participants[i + offset].get('teamPosition') == 'BOTTOM':
            feature_matrix[i][5] = 3
        elif participants[i + offset].get('teamPosition') == 'UTILITY':
            feature_matrix[i][5] = 4

    feature_matrix = feature_matrix[feature_matrix[:, 5].argsort()]
    return feature_matrix


def generate_data(matchid, ranking):
    feature_matrix = get_feature_matrix(matchid)
    # we add the y labels to the front of the matrix to store in new csv file for later retrieval
    feature_matrix = np.c_[ranking.astype(np.int64).T, feature_matrix]
    with open("data.csv", "a") as data_csv:
        np.savetxt(data_csv, feature_matrix, fmt='%d', delimiter=',')


if __name__ == '__main__':
    with open('../../rankings.csv', newline='') as csvfile:
        rankings_csv = csv.reader(csvfile)
        next(rankings_csv)
        for row in rankings_csv:
            print(row)
            ranking = np.delete(row, 0)
            generate_data(row[0], ranking)
