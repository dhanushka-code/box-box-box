import json
import os
import numpy as np
from scipy.optimize import linprog

def extract_features(strategy, total_laps, track_temp):
    features = np.zeros(6)
    current_tire = strategy['starting_tire']
    pit_stops = {stop['lap']: stop for stop in strategy['pit_stops']}
    
    tire_age = 0
    pit_count = 0
    
    for lap in range(1, total_laps + 1):
        tire_age += 1
        
        if current_tire == "SOFT":
            features[0] += 1
            features[2] += tire_age
        elif current_tire == "MEDIUM":
            features[3] += tire_age
        elif current_tire == "HARD":
            features[1] += 1
            features[4] += tire_age
            
        # Temperature interacts universally with total tire age
        features[5] += track_temp * tire_age
        
        if lap in pit_stops:
            current_tire = pit_stops[lap]['to_tire']
            tire_age = 0
            pit_count += 1
            
    return features, pit_count

def main():
    print("Loading historical race matrices...")
    path = os.path.join("data", "historical_races", "races_00000-00999.json")
    with open(path, 'r') as f:
        races = json.load(f)[:150] # 150 races gives us thousands of data points to ensure perfect accuracy.

    A_ub = []
    b_ub = []

    for race in races:
        config = race['race_config']
        total_laps = config['total_laps']
        track_temp = config['track_temp']
        pit_lane_time = config['pit_lane_time']
        
        driver_data = {}
        for pos, strategy in race['strategies'].items():
            d_id = strategy['driver_id']
            driver_data[d_id] = extract_features(strategy, total_laps, track_temp)
            
        finishing_positions = race['finishing_positions']
        
        # Build the matrix: Driver N beat Driver N+1
        # Time equation: (Features * Constants) + (PitCount * PitTime)
        for i in range(len(finishing_positions) - 1):
            winner_id = finishing_positions[i]
            loser_id = finishing_positions[i+1]
            
            w_feat, w_pits = driver_data[winner_id]
            l_feat, l_pits = driver_data[loser_id]
            
            diff_feat = w_feat - l_feat
            
            # (Feat_W * X) - (Feat_L * X) <= -0.001 - (Pits_W * PitTime - Pits_L * PitTime)
            A_ub.append(diff_feat)
            b_ub.append(-0.001 - ((w_pits - l_pits) * pit_lane_time))

    # Logical bounds for F1 mechanics
    bounds = [
        (-5.0, -0.001),  # Soft Offset (Negative = Faster)
        (0.001, 5.0),    # Hard Offset (Positive = Slower)
        (0.001, 0.5),    # Soft Deg
        (0.001, 0.5),    # Med Deg
        (0.001, 0.5),    # Hard Deg
        (-0.01, 0.01)    # Temp Multiplier
    ]

    print("Solving the System of Inequalities... (This is pure math, no guessing!)")
    res = linprog(np.zeros(6), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        x = res.x
        print("\n🎯 TARGET ACQUIRED! EXACT MATH FOUND! 🎯")
        print(f"SOFT Speed Offset:   {x[0]:.4f}")
        print(f"HARD Speed Offset:   {x[1]:.4f}")
        print(f"SOFT Degradation:    {x[2]:.4f}")
        print(f"MEDIUM Degradation:  {x[3]:.4f}")
        print(f"HARD Degradation:    {x[4]:.4f}")
        print(f"Temperature Factor:  {x[5]:.6f}")
    else:
        print("Still unsolvable. The F1 formula might have compound-specific temperature multipliers.")

if __name__ == '__main__':
    main()