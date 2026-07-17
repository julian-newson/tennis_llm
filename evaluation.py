"""
evaluation.py

Evaluation framework for intent classification accuracy testing.
Tests classify_intent() against 104 labelled queries spanning 7 intents
including both manually curated and LLM-generated test cases.

Achieves 92.3% overall accuracy using batched LLM classification for efficiency.

Usage:
    python evaluation.py

To regenerate test cases, uncomment generate_cases() in __main__.
"""

from llm import classify_intent_batch
from llm_client import safe_llm_call
import json
import random
import os

manual_cases = [

    #obvious ones

    {"query": "who wins Nadal vs Federer", "expected_intent": "h2h", "source": "manual"},
    {"query": "head to head of sinner and alcaraz", "expected_intent": "h2h", "source": "manual"},

    {"query": "how does Djokovic perform on clay", "expected_intent": "surface_performance", "source": "manual"},
    {"query": "how does Dimitrov play on grass", "expected_intent": "surface_performance", "source": "manual"},
    
    {"query": "what are fritz statistics", "expected_intent": "player_stats", "source": "manual"},
    {"query": "statistics of Auger-Alliasime", "expected_intent": "player_stats", "source": "manual"},

    {"query": "who is on form right now?", "expected_intent": "on_form_players", "source": "manual"},
    {"query": "who are the on-form players currently?", "expected_intent": "on_form_players", "source": "manual"},

    {"query": "who are the tournament favourites for US open?", "expected_intent": "tournament_favourites", "source": "manual"},
    {"query": "who are the clear favourites for Wimbledon?", "expected_intent": "tournament_favourites", "source": "manual"},

    {"query": "how does zverev perform at the Australian Open?", "expected_intent": "tournament_performance", "source": "manual"},
    {"query": "what is Tommy Paul's performance like at Indian Wells?", "expected_intent": "tournament_performance", "source": "manual"},

    {"query": "what time is it", "expected_intent": "unknown", "source": "manual"},
    {"query": "capital of France", "expected_intent": "unknown", "source": "manual"},

    # slightly ambiguous ones

    {"query": "Alexander Z vs novak", "expected_intent": "h2h", "source": "manual"},
    {"query": "who wins tomy paul or perricard?", "expected_intent": "h2h", "source": "manual"},

    {"query": "taylor f clay", "expected_intent": "surface_performance"},
    {"query": "stan warinka on hard courts", "expected_intent": "surface_performance", "source": "manual"},

    {"query": "how good was Federer", "expected_intent": "player_stats", "source": "manual"},
    {"query": "is Shelton a good player?", "expected_intent": "player_stats", "source": "manual"},

    {"query": "best grass players", "expected_intent": "on_form_players", "source": "manual"},
    {"query": "best players right now", "expected_intent": "on_form_players", "source": "manual"},

    {"query": "who should i bet on for paris masters", "expected_intent": "tournament_favourites", "source": "manual"},
    {"query": "who will win ATP finals?", "expected_intent": "tournament_favourites", "source": "manual"},

    {"query": "murray roland garros", "expected_intent": "tournament_performance", "source": "manual"},
    {"query": "Tommy Paul at Queen's club", "expected_intent": "tournament_performance", "source": "manual"},
 
    {"query": "who coaches Draper?", "expected_intent": "unknown", "source": "manual"},
    {"query": "how tall is djokovic", "expected_intent": "unknown", "source": "manual"},

    # ambiguous ones

    {"query": "jannik and carlos", "expected_intent": "h2h", "source": "manual"},
    {"query": "medvedev hard", "expected_intent": "surface_performance", "source": "manual"},
    {"query": "Tommy P summary", "expected_intent": "player_stats", "source": "manual"},
    {"query": "current clay court specialists", "expected_intent": "on_form_players", "source": "manual"},
    {"query": "who should i watch at Miami", "expected_intent": "tournament_favourites", "source": "manual"},
    {"query": "sinner wimbledon match stats", "expected_intent": "unknown", "source": "manual"},
]

def generate_test_cases_llm(intent, intent_description, n=10):

    """
    Generates n test cases for a given intent using the LLM.

    Args:
        intent: Intent name e.g. 'h2h', 'surface_performance'
        intent_description: Description of the intent to guide generation
        n: Number of test cases to generate, defaults to 10

    Returns:
        List of dicts with 'query', 'expected_intent' and 'source' keys
    """
    prompt = f"""Generate {n} diverse, realistic queries about ATP men's tour only for intent: {intent}
            Based off intent description: {intent_description}
            Mix obvious, slightly ambiguous and ambiguous ones where intent is only implied. Return as a list, one per line."""

    messages = [{"role": "user", "content": prompt}]
    response = safe_llm_call(messages, False)

    # parse LLM response into list of dicts, stripping numbering e.g. "1. query" -> "query"

    queries = response.choices[0].message.content.strip().split("\n")
    test_cases_llm = []
    for query in queries:
        test_cases_llm.append({"query": query.strip().lstrip("0123456789. "), "expected_intent": intent, "source": "llm"})

    return test_cases_llm

def evaluate(batch):

    """
    Evaluates a batch of test cases against the intent classifier.

    Args:
        batch: List of test case dicts with 'query' and 'expected_intent' keys

    Returns:
        Tuple of (correct_count, failures, intent_results) where:
            - correct_count: number of correct classifications
            - failures: list of dicts with 'query', 'expected' and 'predicted' keys
            - intent_results: dict tracking correct/total counts per intent
    """
    correct = 0
    failures = []
    queries = []
    intent_results = {}

    # extract just the query strings for batch classification
    for test in batch:
        queries.append(test["query"])
    
    predicted_intents = classify_intent_batch(queries)

    for j, test in enumerate(batch):
        predicted_intent = predicted_intents[j]
        expected_intent = test["expected_intent"]
        
        # track per-intent accuracy
        if expected_intent not in intent_results:
            intent_results[expected_intent] = {"correct": 0, "total": 0}

        intent_results[expected_intent]["total"] += 1

        if predicted_intent == expected_intent:
            correct += 1
            intent_results[expected_intent]["correct"] += 1
        else:
            failures.append({
                "query": test["query"],
                "expected": expected_intent,
                "predicted": predicted_intent
            })
    
    return correct, failures, intent_results

def run_evaluation(batch_size = 10):

    """
    Runs the full evaluation across all test cases in batches.

    Args:
        batch_size: Number of test cases per batch API call, defaults to 10

    Returns:
        Tuple of (accuracy, total_failures, total_intent_results) where:
            - accuracy: overall accuracy percentage e.g. 92.3
            - total_failures: list of all failed test cases with expected vs predicted
            - total_intent_results: dict with per-intent correct/total counts
    """

    test_cases = load_cases()
    # shuffle to avoid ordering bias in evaluation
    random.shuffle(test_cases)
    print(f"manual_cases: {len(manual_cases)}, llm_cases: {len(test_cases)-len(manual_cases)}")
    
    total_correct = 0
    total_failures = []
    total_intent_results = {}

    for i in range(0, len(test_cases), batch_size):
        print(f"Evaluating batch {i//batch_size + 1}/{len(test_cases)//batch_size + 1}...")
        batch = test_cases[i:i+batch_size]
        correct, failures, intent_results = evaluate(batch)
        total_correct += correct
        total_failures.extend(failures)

        # aggregate per-intent results across batches
        for intent, results in intent_results.items():
            if intent not in total_intent_results:
                total_intent_results[intent] = {"correct": 0, "total": 0}
            total_intent_results[intent]["correct"] += results["correct"]
            total_intent_results[intent]["total"] += results["total"]

    accuracy = round((total_correct/len(test_cases))*100,1)
    return accuracy, total_failures, total_intent_results
    
def generate_cases(n=10):

    """
    Generates and saves test cases combining manual and LLM-generated cases.
    Run once to create test_cases.json — uncomment in __main__ to regenerate.
    
    Args:
        n: Number of LLM-generated cases per intent, defaults to 10
    """

    intent_descriptions = {
    "h2h": "general head to head record between two players only across all tournaments",
    "surface_performance": "how a player performs on a specific surface or all surfaces",
    "player_stats": "overall win rate, win rates on each surface, win rate vs higher/lower ranked opponents, best tournament, recent form",
    "on_form_players": "which players are currently in form, best performing players recently, either for all surfaces/specific one",
    "tournament_favourites": "who is likely to win an upcoming tournament",
    "tournament_performance": "how a specific player has performed at a specific tournament",
    "unknown": "questions about rankings, grand slam titles, prize money, coaching, playing style, or anything not covered by the other intents"
        }   
    
    llm_cases = []
    for intent, description in intent_descriptions.items():
        llm_cases.extend(generate_test_cases_llm(intent, description,n))
    all_cases = manual_cases + llm_cases
    with open("test_cases.json", "w") as f:
        json.dump(all_cases, f, indent=4)

def load_cases():

    """
    Loads test cases from test_cases.json.
    
    Returns:
        List of test case dicts with 'query', 'expected_intent' and 'source' keys
    """

    if not os.path.exists("test_cases.json"):
        raise FileNotFoundError("test_cases.json not found - run generate_cases() first")
    with open("test_cases.json", "r") as f:
        all_cases = json.load(f)
    return all_cases

if __name__ == "__main__":
    # uncomment to generate test cases
    #generate_cases(10)
    accuracy, total_failures, total_intent_results = run_evaluation(10)
    print(f"accuracy: {accuracy}%")
    print(total_intent_results)
    # uncomment to see individual failure cases:
    #print(f"failures: {total_failures}")
    
   