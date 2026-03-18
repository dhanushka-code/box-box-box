import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_driver_features(race_config, strategy):
    track_temp = race_config['track_temp']
    pit_lane_time = race_config['pit_lane_time']
    
    soft_laps = med_laps = hard_laps = 0
    current_tire = strategy['starting_tire']
    pit_stops = {stop['lap']: stop for stop in strategy['pit_stops']}
    num_pits = len(pit_stops)
    
    # Calculate exactly how many laps were driven on each compound
    for lap in range(1, race_config['total_laps'] + 1):
        if current_tire == "SOFT": soft_laps += 1
        elif current_tire == "MEDIUM": med_laps += 1
        elif current_tire == "HARD": hard_laps += 1
        
        if lap in pit_stops:
            current_tire = pit_stops[lap]['to_tire']
            
    # The Ultimate Feature Vector (including temperature interactions)
    return np.array([
        soft_laps, med_laps, hard_laps, num_pits,
        soft_laps * track_temp, med_laps * track_temp, hard_laps * track_temp,
        num_pits * pit_lane_time
    ])

def main():
    print("Loading Historical Data for 1v1 Battles...")
    hist_dir = os.path.join("..", "data", "historical_races")
    
    X_train = []
    y_train = []
    
    # Training on 2,000 races to generate hundreds of thousands of 1v1 matchups
    for filename in sorted(os.listdir(hist_dir))[:2]:
        with open(os.path.join(hist_dir, filename), 'r') as f:
            races = json.load(f)
            for race in races:
                config = race['race_config']
                finishing_positions = race['finishing_positions']
                
                # Pre-calculate features and ranks for all drivers in this race
                driver_data = {}
                for pos, strategy in race['strategies'].items():
                    d_id = strategy['driver_id']
                    driver_data[d_id] = {
                        'features': extract_driver_features(config, strategy),
                        'rank': finishing_positions.index(d_id)
                    }
                
                # Create the 1v1 matchups
                drivers = list(driver_data.keys())
                for i in range(len(drivers)):
                    for j in range(i + 1, len(drivers)):
                        d1 = drivers[i]
                        d2 = drivers[j]
                        
                        f1 = driver_data[d1]['features']
                        f2 = driver_data[d2]['features']
                        
                        # We train the AI on the DIFFERENCE between their stats
                        if driver_data[d1]['rank'] < driver_data[d2]['rank']:
                            X_train.append(f1 - f2) # d1 beats d2
                            y_train.append(1)
                            X_train.append(f2 - f1) # d2 loses to d1
                            y_train.append(0)
                        else:
                            X_train.append(f1 - f2) # d1 loses to d2
                            y_train.append(0)
                            X_train.append(f2 - f1) # d2 beats d1
                            y_train.append(1)
                            
    print(f"Generated {len(X_train)} pairwise battles. Training Classifier...")
    # n_jobs=-1 forces your Lenovo LOQ to use every single CPU core it has!
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Running the Final Exam Tournament...")
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
        
        # Extract features for this specific test case
        test_drivers = {}
        for pos, strategy in race_data['strategies'].items():
            d_id = strategy['driver_id']
            test_drivers[d_id] = extract_driver_features(config, strategy)
            
        # Simulate a round-robin tournament for this race
        driver_wins = {d_id: 0 for d_id in test_drivers}
        drivers = list(test_drivers.keys())
        
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                d1 = drivers[i]
                d2 = drivers[j]
                
                diff = test_drivers[d1] - test_drivers[d2]
                
                # Predict who wins the 1v1 based on their feature differences
                if model.predict([diff])[0] == 1:
                    driver_wins[d1] += 1
                else:
                    driver_wins[d2] += 1
                    
        # Sort the drivers by whoever got the most 1v1 wins
        predicted_order = sorted(driver_wins.keys(), key=lambda x: driver_wins[x], reverse=True)
        
        if predicted_order != actual_order:
            for driver in predicted_order:
                total_error += abs(predicted_order.index(driver) - actual_order.index(driver))
                
    print("\n========================================================")
    print("⚔️ PAIRWISE ML TOURNAMENT RESULTS ⚔️")
    print("========================================================")
    print(f"Total Error Points: {total_error}")

if __name__ == '__main__':
    main()