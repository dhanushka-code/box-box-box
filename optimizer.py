import json
import os
from scipy.optimize import differential_evolution

# Load a batch of historical races to train our model
def load_training_data():
    path = os.path.join("data", "historical_races", "races_00000-00999.json")
    with open(path, 'r') as f:
        # We only need about 50 races to find the exact math
        return json.load(f)[:50] 

def simulate_race(race, params):
    # Unpack the AI's current guessed parameters
    offset_soft, offset_hard, deg_soft, deg_medium, deg_hard, temp_factor = params
    
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
            
            # Apply the AI's current guessed math
            if current_tire == "SOFT":
                lap_time = base_lap_time + offset_soft + (deg_soft * tire_age)
            elif current_tire == "MEDIUM":
                lap_time = base_lap_time + 0.0 + (deg_medium * tire_age) # Medium is our baseline 0.0
            else: # HARD
                lap_time = base_lap_time + offset_hard + (deg_hard * tire_age)
                
            # Add temperature degradation effect (rulebook says temp impacts degradation)
            lap_time += (track_temp - 28) * temp_factor * tire_age
            
            total_time += lap_time
            
            if lap in pit_stops:
                total_time += pit_lane_time
                current_tire = pit_stops[lap]['to_tire']
                tire_age = 0
                
        driver_times[driver_id] = total_time
        
    # Sort drivers by total time to get predicted order
    sorted_drivers = sorted(driver_times.items(), key=lambda x: x[1])
    return [d[0] for d in sorted_drivers]

def objective_function(params, races):
    total_error = 0
    
    for race in races:
        predicted_order = simulate_race(race, params)
        actual_order = race['finishing_positions']
        
        # Calculate how far off our prediction is from reality
        for driver in predicted_order:
            pred_idx = predicted_order.index(driver)
            act_idx = actual_order.index(driver)
            total_error += abs(pred_idx - act_idx)
            
    return total_error

def main():
    print("Loading historical F1 data...")
    races = load_training_data()
    
    print("Starting Evolutionary Algorithm... (This might take a minute, let your LOQ cook!)")
    
    # Define the logical bounds for our search (min, max)
    bounds = [
        (-3.0, -0.1),  # offset_soft (Softs should be faster/negative)
        (0.1, 3.0),    # offset_hard (Hards should be slower/positive)
        (0.01, 0.2),   # deg_soft (High degradation)
        (0.01, 0.2),   # deg_medium (Medium degradation)
        (0.001, 0.1),  # deg_hard (Low degradation)
        (0.0001, 0.01) # temp_factor
    ]
    
    # Run the differential evolution optimizer
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(races,), 
        strategy='best1bin', 
        maxiter=30, 
        popsize=15, 
        tol=0.01, 
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        disp=True
    )
    
    print("\n========================================================")
    print("🎯 OPTIMIZATION COMPLETE - WE FOUND THE SECRET MATH!")
    print("========================================================")
    print(f"SOFT Speed Offset:   {result.x[0]:.4f}")
    print(f"HARD Speed Offset:   {result.x[1]:.4f}")
    print(f"SOFT Degradation:    {result.x[2]:.4f}")
    print(f"MEDIUM Degradation:  {result.x[3]:.4f}")
    print(f"HARD Degradation:    {result.x[4]:.4f}")
    print(f"Temperature Factor:  {result.x[5]:.6f}")
    print("\nPlug these exact numbers into solution.py!")

if __name__ == "__main__":
    main()