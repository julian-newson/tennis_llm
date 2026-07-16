import pandas as pd
from llm_client import client, safe_llm_call
from rapidfuzz import process


def get_unique_players(df):
    players_p1 = df[["Player_1"]].rename(columns={"Player_1": "Player"})
    players_p2 = df[["Player_2"]].rename(columns={"Player_2": "Player"})
    combined_players = pd.concat([players_p1, players_p2])
    unique_players = combined_players["Player"].str.strip().unique()
    return unique_players

def get_unique_tournaments(df):
    unique_tournaments = df['Tournament'].unique().tolist()
    return unique_tournaments


def find_player(query, df, unique_players):

    matches = []
    
    # if enter player in the list we've found a match
    for player in unique_players:
        if query.lower() in player.lower():
            matches.append(player)
    
    # if we don't have a match get llm to format name correctly
    if not matches:
        # returns best match, confidence score, index in the array of the best match

        prompt = f"""
        From the query extract the referenced MEN'S ATP tennis player ONLY and return it in form e.g. Novak Djokovic -> Djokovic N.
        Their initial of their first name should be capitalised and come after their last name, followed by a full stop
        If they have middle names the initial of each should be capitalised and come straight after the first name initial,
        and its full stop, and have a full stop of its own, e.g. Struff J.L.
        Their last name should come first, and if consists of multiple words, each should have their first letter capitalised,
        e.g. De Minaur A.
        
        If the player isn't ATP men's e.g. WTA or unable to identify a tennis player only return the word unknown

        The query is: {query}
        """

        messages = [{"role": "user", "content": prompt}]
        response = safe_llm_call(messages, False)
 
        player_name = response.choices[0].message.content

        # final fuzzy safety net

        if player_name == "unknown":
            return None
        elif player_name in unique_players:
            return player_name
        else:
            best_match, score, _ = process.extractOne(player_name, unique_players)
            print(f"Fuzzy match: {best_match}, score: {score}")
            if score >= 90:
                return best_match
            else:
                return None
 
    # beyond here we HAVE a match
    # if we get exactly 1 match without llm just return it
    elif len(matches) == 1:
        return matches[0]
    # else we have multiple matches
    else:
        #print(matches)
        # either return a single match/multiple if games played cannot break the tie
        fil_matches = df[df["Player_1"].isin(matches) | df["Player_2"].isin(matches)]
        match_counts = {}
        for player in matches:
            match_count = len(fil_matches[(fil_matches["Player_1"] == player) | (fil_matches["Player_2"] == player)])
            match_counts[player] = match_count
        sorted_match_counts = sorted(match_counts.values(), reverse=True)

        if len(sorted_match_counts) < 2:
            return matches[0]
        top_candidate = sorted_match_counts[0]
        second_candidate = sorted_match_counts[1]
        if top_candidate > 5 * second_candidate:
            # find key with highest value from dict
            return max(match_counts, key = match_counts.get)
        else:
            return matches
        
def match_tournament(tournament, unique_tournaments):
    best_match, score, _ = process.extractOne(tournament, unique_tournaments)
    #print(f"Score: {score}")
    if score >= 70:
        return best_match
    else:
        return None


















        













