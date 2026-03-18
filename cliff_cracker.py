import json
import os
from scipy.optimize import differential_evolution

def load_training_data():
    path = os.path.join("data", "historical_races", "races_00000-00999.json")
    with open(path, 'r') as f:
        return json.load(f)[:50] 

def simulate_race(race, params):
    # Unpack 9 parameters including the new Grace Periods
    (offset_soft, offset_hard, 
     grace_soft, grace_med, grace_hard, 
     deg_soft, deg_med, deg_hard, 
     temp_factor) = params
     
    # Grace periods must be whole laps
    grace_soft = int(round(grace_soft))
    grace_med = int(round(grace_med))
    grace_hard = int(round(grace_hard))
    
    config = race['race_config']
    base_lap_time = config['base_lap_time']
    track_temp = config['track_temp']
    pit_lane_time = config['pit_lane_time']
    total_laps = config['total_laps']
    
    driver_times = {}
    
    for driver_pos, strategy in race['strategies'].items():
        driver_id = strategy['driver_id']
        current_tire = strategy['starting_tire']
        pit_stops = {stop['lap']: stop for stop in strategy['pit_stops']}
        
        total_time = 0.0
        tire_age = 0
        
        for lap in range(1, total_laps + 1):
            tire_age += 1
            
            # 1. Base + Offset
            if current_tire == "SOFT":
                lap_time = base_lap_time + offset_soft
                grace = grace_soft
                deg_rate = deg_soft
            elif current_tire == "MEDIUM":
                lap_time = base_lap_time + 0.0
                grace = grace_med
                deg_rate = deg_med
            else: # HARD
                lap_time = base_lap_time + offset_hard
                grace = grace_hard
                deg_rate = deg_hard
                
            # 2. Degradation Cliff (Initial Performance Period)
            if tire_age > grace:
                effective_age = tire_age - grace
                # Track temp increases the severity of the degradation
                actual_deg = deg_rate + (track_temp * temp_factor)
                lap_time += actual_deg * effective_age
                
            total_time += lap_time
            
            # 3. Pit Stops
            if lap in pit_stops:
                total_time += pit_lane_time
                current_tire = pit_stops[lap]['to_tire']
                tire_age = 0
                
        driver_times[driver_id] = total_time
        
    sorted_drivers = sorted(driver_times.items(), key=lambda x: x[1])
    return [d[0] for d in sorted_drivers]

def objective_function(params, races):
    total_error = 0
    for race in races:
        predicted = simulate_race(race, params)
        actual = race['finishing_positions']
        for driver in predicted:
            total_error += abs(predicted.index(driver) - actual.index(driver))
    return total_error

def main():
    print("Loading historical F1 data...")
    races = load_training_data()
    
    print("Running Advanced AI Cracker (Accounting for Tire Cliffs)...")
    
    # Let the AI search within these logical bounds
    bounds = [
        (-3.0, -0.1),   # offset_soft (Soft is faster, negative)
        (0.1, 3.0),     # offset_hard (Hard is slower, positive)
        (1, 15),        # grace_soft (Laps before Soft drops off)
        (5, 25),        # grace_med  (Laps before Med drops off)
        (10, 40),       # grace_hard (Laps before Hard drops off)
        (0.01, 0.5),    # deg_soft
        (0.01, 0.5),    # deg_med
        (0.01, 0.5),    # deg_hard
        (0.0001, 0.01)  # temp_factor
    ]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(races,), 
        strategy='best1bin', 
        maxiter=50, 
        popsize=20, 
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        disp=True
    )
    
    print("\n========================================================")
    print("🎯 OPTIMIZATION COMPLETE - TIRE CLIFFS FOUND!")
    print("========================================================")
    print(f"SOFT Offset:   {result.x[0]:.3f} | Grace Laps: {int(round(result.x[2]))} | Deg: {result.x[5]:.4f}")
    print(f"MED Offset:    0.000  | Grace Laps: {int(round(result.x[3]))} | Deg: {result.x[6]:.4f}")
    print(f"HARD Offset:   {result.x[1]:.3f} | Grace Laps: {int(round(result.x[4]))} | Deg: {result.x[7]:.4f}")
    print(f"Temp Factor:   {result.x[8]:.6f}")
    print(f"Total Error:   {result.fun}")

if __name__ == "__main__":
    main()