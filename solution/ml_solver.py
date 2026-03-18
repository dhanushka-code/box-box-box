import json
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def extract_features(race_config, strategy):
    # 1. Track Conditions
    track_temp = race_config['track_temp']
    total_laps = race_config['total_laps']
    pit_lane_time = race_config['pit_lane_time']
    
    # 2. Driver Strategy
    soft_laps = 0
    med_laps = 0
    hard_laps = 0
    
    current_tire = strategy['starting_tire']
    pit_stops = {stop['lap']: stop for stop in strategy['pit_stops']}
    num_pits = len(pit_stops)
    
    for lap in range(1, total_laps + 1):
        if current_tire == "SOFT": soft_laps += 1
        elif current_tire == "MEDIUM": med_laps += 1
        elif current_tire == "HARD": hard_laps += 1
        
        if lap in pit_stops:
            current_tire = pit_stops[lap]['to_tire']
            
    return [track_temp, total_laps, pit_lane_time, num_pits, soft_laps, med_laps, hard_laps]

def main():
    print("Loading Historical Data to Train the AI...")
    hist_dir = os.path.join("..", "data", "historical_races")
    
    X_train = []
    y_train = []
    
    # Load 5 JSON files (5,000 races = 100,000 driver strategies)
    for filename in sorted(os.listdir(hist_dir))[:5]:
        with open(os.path.join(hist_dir, filename), 'r') as f:
            races = json.load(f)
            for race in races:
                config = race['race_config']
                finishing_positions = race['finishing_positions']
                
                for pos, strategy in race['strategies'].items():
                    d_id = strategy['driver_id']
                    features = extract_features(config, strategy)
                    
                    # The Target: What rank did this strategy get? (0 = 1st, 19 = Last)
                    rank = finishing_positions.index(d_id)
                    
                    X_train.append(features)
                    y_train.append(rank)
                    
    print("Training Random Forest Regressor... (Let your LOQ cook!)")
    # n_jobs=-1 tells scikit-learn to use all the CPU cores on your laptop
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Testing against the Final Exam...")
    inputs_dir = os.path.join("..", "data", "test_cases", "inputs")
    expected_dir = os.path.join("..", "data", "test_cases", "expected_outputs")
    
    total_error = 0
    
    for filename in sorted(os.listdir(inputs_dir)):
        if not filename.endswith(".json"): continue
            
        with open(os.path.join(inputs_dir, filename)) as f_in:
            race_data = json.load(f_in)
        with open(os.path.join(expected_dir, filename)) as f_out:
            expected_data = json.load(f_out)
            
        actual_order = expected_data['finishing_positions']
        config = race_data['race_config']
        
        # Predict scores for each driver in the test case
        driver_scores = {}
        for pos, strategy in race_data['strategies'].items():
            d_id = strategy['driver_id']
            features = extract_features(config, strategy)
            
            # Ask the AI what rank it predicts for this driver
            predicted_rank = model.predict([features])[0]
            driver_scores[d_id] = predicted_rank
            
        # Sort drivers by their predicted rank
        predicted_order = [d[0] for d in sorted(driver_scores.items(), key=lambda x: x[1])]
        
        # Calculate error
        if predicted_order != actual_order:
            for driver in predicted_order:
                total_error += abs(predicted_order.index(driver) - actual_order.index(driver))
                
    print("\n========================================================")
    print("🤖 MACHINE LEARNING RESULTS 🤖")
    print("========================================================")
    print(f"Total Error Points: {total_error}")
    if total_error == 0:
        print("🏆 PERFECT SCORE! The AI learned the pattern!")
    else:
        print("Still missing some nuance. We might need more features.")

if __name__ == '__main__':
    main()