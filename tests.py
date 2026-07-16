import pandas as pd
import numpy as np
import pytest
from name_matching import get_unique_players, get_unique_tournaments, match_tournament
from tennis_stats import get_h2h, get_surface_performance, get_tournament_performance, get_player_stats, get_on_form_players, get_favourites

def test_get_unique_players_is_list():
    mock_df = pd.DataFrame({
        'Player_1': ['Djokovic N.', 'Nadal R.'],
        'Player_2': ['Federer R.', 'Murray A.']})
    
    result = get_unique_players(mock_df)
    # returns ArrowStringArray so check it is iterable
    assert hasattr(result, '__iter__')
                      
def test_get_unique_players_contains_all_players():
    mock_df = pd.DataFrame({
        'Player_1': ['Djokovic N.', 'Nadal R.'],
        'Player_2': ['Federer R.', 'Murray A.']})
    
    expected_players = ['Djokovic N.', 'Nadal R.', 'Federer R.', 'Murray A.']
    result = get_unique_players(mock_df)
    assert all(player in result for player in expected_players)

def test_get_unique_players_contains_no_duplicates():
    mock_df = pd.DataFrame({
        'Player_1': ['Djokovic N.', 'Nadal R.'],
        'Player_2': ['Nadal R.', 'Djokovic N.']})
    
    result = get_unique_players(mock_df)
    assert len(result) == 2

def test_get_unique_players_contains_no_whitespace():
    mock_df = pd.DataFrame({
        'Player_1': ['Djokovic N.', 'Nadal R.'],
        'Player_2': ['Federer R.', 'Murray A.']})
    
    result = get_unique_players(mock_df)
    assert all(player == player.strip() for player in result)

def test_get_unique_tournaments_is_list():
    mock_df = pd.DataFrame({'Tournament': ['French Open', 'Wimbledon', 'Miami Open', 'Australian Open']})
    
    result = get_unique_tournaments(mock_df)
    assert isinstance(result, (list, np.ndarray))

def test_get_unique_tournaments_contains_all_tournaments():
    mock_df = pd.DataFrame({'Tournament': ['French Open', 'Wimbledon', 'Miami Open', 'Australian Open']})
    
    expected_tournaments = ['French Open', 'Wimbledon', 'Miami Open', 'Australian Open']
    result = get_unique_tournaments(mock_df)
    assert all(tournament in result for tournament in expected_tournaments)

def test_get_unique_tournaments_contains_no_duplicates():
    mock_df = pd.DataFrame({'Tournament': ['French Open', 'Wimbledon', 'Wimbledon', 'Australian Open']})
    
    result = get_unique_tournaments(mock_df)
    assert len(result) == 3

def test_match_tournament_exact_match():
    tournaments = ["French Open", "Wimbledon", "US Open"]
    result = match_tournament("Wimbledon", tournaments)
    assert result == "Wimbledon"

def test_match_tournament_no_match():
    tournaments = ["French Open", "Wimbledon", "US Open"]
    result = match_tournament("xyznotreal", tournaments)
    assert result is None

def test_match_tournament_case_insensitive():
    tournaments = ["French Open", "Wimbledon", "US Open"]
    result = match_tournament("french open", tournaments)
    assert result == "French Open"

def test_match_tournament_slight_misspelling():
    tournaments = ["French Open", "Wimbledon", "US Open"]
    result = match_tournament("wimbeldon", tournaments) 
    assert result == "Wimbledon"

def test_match_tournament_too_different():
    tournaments = ["French Open", "Wimbledon", "US Open"]
    result = match_tournament("Roland Garros", tournaments)
    assert result is None



# more simpler functions

@pytest.fixture
def mock_df():
    return pd.DataFrame({'Player_1': ['Djokovic N.', 'Federer R.', 'Djokovic N.'],
                        'Player_2': ['Federer R.', 'Djokovic N.', 'Federer R.'],
                        'Winner':   ['Djokovic N.', 'Djokovic N.', 'Federer R.'],
                        'Surface':  ['Clay', 'Grass', 'Hard'],
                        'Tournament': ['French Open', 'Wimbledon', 'US Open'],
                        'Date': ['2020-01-01', '2021-01-01', '2022-01-01'],
                        'Rank_1': [1, 2, 1],
                        'Rank_2': [2, 1, 2]})

def test_get_h2h_contains_both_players(mock_df):
    result = get_h2h(mock_df, "Djokovic N.", "Federer R.")
    expected_players = ["Djokovic N.", "Federer R."]
    assert all(player in result for player in expected_players)

def test_get_h2h_djokovic_win_count(mock_df):
    result = get_h2h(mock_df, "Djokovic N.", "Federer R.")
    assert result["Djokovic N."] == 2

def test_get_h2h_federer_win_count(mock_df):
    result = get_h2h(mock_df, "Djokovic N.", "Federer R.")
    assert result["Federer R."] == 1

def test_get_h2h_no_matches_returns_empty(mock_df):
    result = get_h2h(mock_df, "Nadal R.", "Murray A.")
    assert len(result) == 0

def test_get_h2h_empty_df():
    empty_df = pd.DataFrame(columns=['Date', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Winner', 'Tournament', 'Surface'])
    result = get_h2h(empty_df, "Nadal R.", "Murray A.")
    assert result is None

def test_get_surface_performance_single_surface_played(mock_df):
    result = get_surface_performance(mock_df, "Djokovic N.", "Clay")
    assert result["surface_win_percentage"] == 100.0
    assert result["surface_matches"] == 1

def test_get_surface_performance_single_surface_not_played(mock_df):
    result = get_surface_performance(mock_df, "Djokovic N.", "Carpet")
    assert result["surface_matches"] == 0
    assert result["surface_win_percentage"] is None

def test_get_surface_performance_all_surfaces_returns_dict(mock_df):
    result = get_surface_performance(mock_df, "Djokovic N.", "All")
    assert isinstance(result, dict)
    assert "Grass" in result
    assert "Clay" in result
    assert "Hard" in result
    assert "Carpet" in result

def test_get_surface_performance_all_surfaces_returns_correct_values(mock_df):
    result = get_surface_performance(mock_df, "Djokovic N.", "All")
    assert result["Clay"]["win_rate"] == 100.0
    assert result["Grass"]["win_rate"] == 100.0
    assert result["Hard"]["win_rate"] == 0.0
    assert result["Carpet"] is None

def test_get_surface_performance_empty_df():
    empty_df = pd.DataFrame(columns=['Date', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Winner', 'Tournament', 'Surface'])
    result = get_surface_performance(empty_df, "Nadal R.", "Clay")
    assert result is None

def test_get_tournament_performance_correct_values(mock_df):
    result = get_tournament_performance(mock_df, "Djokovic N.", "Wimbledon")
    assert result["tournament_win_percentage"] == 100.0
    assert result["tournament_matches"] == 1

def test_get_tournament_performance_no_matches(mock_df):
    result = get_tournament_performance(mock_df, "Djokovic N.", "Australian Open")
    assert result is None

def test_get_tournament_performance_empty_df():
    empty_df = pd.DataFrame(columns=['Date', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Winner', 'Tournament', 'Surface'])
    result = get_tournament_performance(empty_df, "Nadal R.", "Wimbledon")
    assert result is None


# more complicated functions, based off df

@pytest.fixture
def df():
    return pd.read_csv("data/atp_tennis.csv")

def test_get_player_stats_returns_dict(df):
    result = get_player_stats(df, "Nadal R.")
    assert isinstance(result, dict)

def test_get_player_stats_win_rate(df):
    result = get_player_stats(df, "Nadal R.")
    assert result["overall_win_rate"] > 80

def test_get_player_stats_surface(df):
    result = get_player_stats(df, "Nadal R.")
    assert result["win_rate_by_surface"]["Clay"] > 85

def test_get_player_stats_higher_lower(df):
    result = get_player_stats(df, "Nadal R.")
    assert result["higher_lower_win_rates"]["win_rate_higher_rank"] < result["higher_lower_win_rates"]["win_rate_lower_rank"]

def test_get_player_stats_best_tournament(df):
    result = get_player_stats(df, "Nadal R.")
    assert result["best_tournament_stats"]["best_tournament"] == "French Open"

def test_get_player_stats_empty_df():
    empty_df = pd.DataFrame(columns=['Date', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Winner', 'Tournament', 'Surface'])
    result = get_player_stats(empty_df, "Nadal R.")
    assert result is None

def test_get_on_form_players_is_list(df):
    result = get_on_form_players(df)
    assert hasattr(result, '__iter__')

def test_get_on_form_players_returns_correct_number(df):
    result = get_on_form_players(df, on_form_number=3)
    assert len(result) == 3

def test_get_on_form_players_no_duplicates(df):
    result = get_on_form_players(df)
    assert len(result) == len(result.index.unique())

def test_get_on_form_players_sorted_descending(df):
    result = get_on_form_players(df)
    win_rates = result["win_rate"].tolist()
    assert win_rates == sorted(win_rates, reverse=True)

def test_get_on_form_players_min_matches(df):
    result = get_on_form_players(df)
    assert all(result["matches_played"] >= 10)

def test_get_on_form_players_no_surface_data(df):
    result = get_on_form_players(df, surface="Carpet")
    assert len(result) == 0

def test_get_on_form_players_surface_filter(df):
    result = get_on_form_players(df, surface="Grass")
    assert len(result) > 0

def test_get_on_form_players_empty_df():
    empty_df = pd.DataFrame(columns=['Date', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Winner', 'Tournament', 'Surface'])
    result = get_on_form_players(empty_df)
    assert result is None

def test_get_favourites_returns_dict(df):
    result = get_favourites(df, "Wimbledon")
    assert isinstance(result, dict)

def test_get_favourites_returns_five(df):
    result = get_favourites(df, "Wimbledon")
    assert len(result) == 5

def test_get_favourites_no_duplicates(df):
    result = get_favourites(df, "Wimbledon")
    players = list(result.values())
    assert len(players) == len(set(players))

def test_get_favourites_correct_keys(df):
    result = get_favourites(df, "Wimbledon")
    assert "Favourite 1" in result

def test_get_favourites_unknown_tournament(df):
    result = get_favourites(df, "abcde")
    assert result is None

def test_get_favourites_empty_df():
    empty_df = pd.DataFrame(columns=['Date', 'Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Winner', 'Tournament', 'Surface'])
    result = get_favourites(empty_df, "Wimbledon")
    assert result is None