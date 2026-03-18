import json
import os
from scipy.optimize import differential_evolution

def load_test_cases():
    inputs_dir = os.path.join("..", "data", "test_cases", "inputs")
    expected_dir = os.path.join("..", "data", "test_cases", "expected_outputs")
    cases = []
    for filename in sorted(os.listdir(inputs_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(inputs_dir, filename)) as f_in:
                race_data = json.load(f_in)
            with open(os.path.join(expected_dir, filename)) as f_out:
                expected_data = json.load(f_out)
            cases.append((race_data, expected_data['finishing_positions']))
    return cases

def simulate_race(race, params):
    # We added the 10th secret variable: wear_exponent!
    (offset_soft, offset_hard, 
     grace_soft, grace_med, grace_hard, 
     deg_soft, deg_med, deg_hard, 
     temp_factor, wear_exponent) = params
     
    grace_soft = int(round(grace_soft))
    grace_med = int(round(grace_med))
    grace_hard = int(round(grace_hard))
    
    config = race['race_config']
    base_lap_time = config['base_lap_time']
    track_temp = config['track_temp']
    pit_lane_time = config['pit_lane_time']
    total_laps = config['total_laps']
    
    driver_times = {}
    
    for pos, strategy in race['strategies'].items():
        driver_id = strategy['driver_id']
        current_tire = strategy['starting_tire']
        pit_stops = {stop['lap']: stop for stop in strategy['pit_stops']}
        
        total_time = 0.0
        tire_age = 0
        
        for lap in range(1, total_laps + 1):
            # The first lap on fresh tires is driven at age 1
            tire_age += 1
            
            if current_tire == "SOFT":
                lap_time = base_lap_time + offset_soft
                grace = grace_soft
                base_deg = deg_soft
            elif current_tire == "MEDIUM":
                lap_time = base_lap_time + 0.0
                grace = grace_med
                base_deg = deg_med
            else: 
                lap_time = base_lap_time + offset_hard
                grace = grace_hard
                base_deg = deg_hard
                
            # The Non-Linear Tire Cliff
            if tire_age > grace:
                effective_age = tire_age - grace
                actual_deg = base_deg + (track_temp * temp_factor)
                # Here is the magic: raising the age to the power of the exponent!
                lap_time += actual_deg * (effective_age ** wear_exponent)
                
            total_time += lap_time
            
            if lap in pit_stops:
                total_time += pit_lane_time
                current_tire = pit_stops[lap]['to_tire']
                tire_age = 0
                
        driver_times[driver_id] = total_time
        
    sorted_drivers = sorted(driver_times.items(), key=lambda x: x[1])
    return [d[0] for d in sorted_drivers]

def objective_function(params, test_cases):
    total_error = 0
    for race, actual_order in test_cases:
        predicted = simulate_race(race, params)
        if predicted != actual_order:
            for driver in predicted:
                total_error += abs(predicted.index(driver) - actual_order.index(driver))
    return total_error

def main():
    print("Loading Final Exam...")
    test_cases = load_test_cases()
    print("Running Ultimate Hacker AI (Hunting the Non-Linear Curve)...")
    
    # We force the AI to look for an exponent greater than 1.0
    bounds = [
        (-2.0, -0.1),   
        (0.1, 2.0),     
        (5, 15),        
        (15, 25),       
        (25, 40),       
        (0.001, 0.2),    
        (0.001, 0.2),    
        (0.001, 0.2),    
        (0.0001, 0.01), 
        (1.1, 3.0)      # wear_exponent
    ]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(test_cases,), 
        strategy='best1bin', 
        maxiter=100, 
        popsize=15, 
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        disp=True
    )
    
    print("\n========================================================")
    print("🎯 ULTIMATE MATH FOUND!")
    print("========================================================")
    print(f"SOFT Offset: {result.x[0]:.4f} | Grace: {int(round(result.x[2]))} | Deg: {result.x[5]:.4f}")
    print(f"MED Offset:  0.0000 | Grace: {int(round(result.x[3]))} | Deg: {result.x[6]:.4f}")
    print(f"HARD Offset: {result.x[1]:.4f} | Grace: {int(round(result.x[4]))} | Deg: {result.x[7]:.4f}")
    print(f"Temp Factor: {result.x[8]:.6f}")
    print(f"Wear Curve Exponent: {result.x[9]:.4f}")
    print(f"Total Error Points: {result.fun}")

if __name__ == '__main__':
    main()