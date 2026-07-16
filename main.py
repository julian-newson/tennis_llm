from llm import classify_intent, extract_entities
from tennis_stats import get_h2h, get_surface_performance, get_player_stats, get_on_form_players, get_favourites, get_tournament_performance
from name_matching import find_player, get_unique_players, get_unique_tournaments
import pandas as pd

def handle_query(df, query, unique_players, unique_tournaments, history = [], cache = None):
    
    try:
        if cache is None:
            cache = {}


        print(query)
        # 1) classify intent

        intent = classify_intent(query, history)

        print(f"intent: {intent}")
        # 2) extract the entities

        if intent == "unknown":
            result = """I can only answer questions about the ATP tour on 
                        head-to-head records, player surface performance, player stats, on-form players, 
                        tournament favourites or player tournament performance."""
            return query, result
        
        # will either contain last assistant message OR be empty
        
        if len(history) >= 2: 
            last_message = history[-2]["content"]
        else:
            last_message = None

        entities = extract_entities(query, intent, unique_tournaments, last_message)

        print(f"entities: {entities}")

        if entities is None:
            result = "An error occurred in extracting entities"
            return query, result
        #3 ) if entities contain a name then resolve name

        if entities:
            for key in entities:
                if key in ["player", "player_1", "player_2"]:
                    potential_player = find_player(entities[key], df, unique_players)

                    if isinstance(potential_player, list):
                        result = f"Multiple players found, did you mean: {', '.join(potential_player)}?"
                        return query, result
                    elif potential_player is None:
                        result = f"Could not find player: {entities[key]}, please try again"
                        return query, result
                    else:
                        entities[key] = potential_player

            print(f"Resolved name: {entities}")


        cache_key = f"{intent}_{str(entities)}"

        # check if intent-entity pair is already in cache
        cached_result = None
        if cache_key in cache:
            cached_result = cache[cache_key]
            #print("hit")

        if cached_result is None:
    
            if intent == "h2h":
                result = get_h2h(df, entities["player_1"], entities["player_2"])
            elif intent == "surface_performance":
                # surface is either specific/all
                result = get_surface_performance(df, entities["player"], entities["surface"])
            elif intent == "player_stats":
                result = get_player_stats(df, entities["player"])
            elif intent == "on_form_players":
                # surface is either specific/all
                result = get_on_form_players(df, entities["surface"])
            elif intent == "tournament_favourites":
                # this gets the data
                result = get_favourites(df, entities["tournament"])
            elif intent == "tournament_performance":
                result = get_tournament_performance(df, entities["player"], entities["tournament"])
    
            else:
                # system error
                result = "An error occurred in classifying intent"

            cache[cache_key] = result
            
        else:
            result = cached_result

        #print(f"Result: {result}")
        #print(f"Cache: {cache}")
        return query, result
    # error messages either due to rate limiting or general error message
    except Exception as e:
        print(f"ERROR: {e}")
        if "rate_limit" in str(e):
            result = "Too many requests - please wait a moment and try again."
        else:
            result = "Something went wrong, please try again."

        return query, result












# if run file directly, for testing
if __name__ == "__main__":
    df = pd.read_csv("data/atp_tennis.csv")
    unique_players = get_unique_players(df)
    unique_tournaments = get_unique_tournaments(df)


    q1 = "who wins in A.Zverev vs taylor fritz?"
    q2 = "how does Novak Djokovic perform on grass"
    q3 = "Djokovic statistics"
    q4 = "who is playing well right now?"
    q5 = "who are the wimbledon favourites?"
    q6 = "what time is it?"
    q7 = "broady vs ruud"
    q8 = "liam broady clay"
    q9 = "liam broady stats"
    q10 = "best players rn"
    q11 = "who are the roland garros favourites"
    q12 = "murray roland garros"
    q13 = "djokovic height"

    q14 = "zverev stats"
    q15 = "serena williams stats"
    q16 = "john mcenroe stats"
    q17 = "rafael nadal stats"
    q18 = "norrie tournament performance at bastad open"
    q19 = "norrie surface performance"
    qs= [q11]
    for q in qs:
        query, result = handle_query(df, q, unique_players, unique_tournaments)
        print(result)