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
    # 10 Variables. We dropped the basic temp multiplier and added temp_grace_loss
    (offset_soft, offset_hard, 
     base_grace_soft, base_grace_med, base_grace_hard, 
     temp_grace_loss,  # NEW: How many grace laps are lost per degree of heat
     deg_soft, deg_med, deg_hard, 
     wear_exponent) = params
     
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
            tire_age += 1
            
            if current_tire == "SOFT":
                lap_time = base_lap_time + offset_soft
                base_grace = base_grace_soft
                deg_rate = deg_soft
            elif current_tire == "MEDIUM":
                lap_time = base_lap_time + 0.0
                base_grace = base_grace_med
                deg_rate = deg_med
            else: 
                lap_time = base_lap_time + offset_hard
                base_grace = base_grace_hard
                deg_rate = deg_hard
                
            # THE SECRET MECHANIC: Heat destroys the grace period
            # Assuming 25C is a baseline temperate track
            heat_penalty = max(0, track_temp - 25) * temp_grace_loss
            actual_grace = max(1.0, base_grace - heat_penalty)
            
            # The Non-Linear Cliff
            if tire_age > actual_grace:
                effective_age = tire_age - actual_grace
                lap_time += deg_rate * (effective_age ** wear_exponent)
                
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
    print("Running Grandmaster AI (Hunting the Heat Penalty)...")
    
    bounds = [
        (-2.0, -0.1),   # offset_soft
        (0.1, 2.0),     # offset_hard
        (8, 15),        # base_grace_soft
        (15, 25),       # base_grace_med
        (25, 40),       # base_grace_hard
        (0.01, 0.5),    # temp_grace_loss (Fraction of a lap lost per degree)
        (0.01, 0.3),    # deg_soft
        (0.01, 0.3),    # deg_med
        (0.001, 0.1),   # deg_hard
        (1.5, 2.5)      # wear_exponent (Usually quadratic in games, close to 2.0)
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
    print("🏆 GRANDMASTER MATH FOUND! 🏆")
    print("========================================================")
    print(f"SOFT Offset: {result.x[0]:.4f} | Base Grace: {result.x[2]:.2f} | Deg: {result.x[6]:.4f}")
    print(f"MED Offset:  0.0000 | Base Grace: {result.x[3]:.2f} | Deg: {result.x[7]:.4f}")
    print(f"HARD Offset: {result.x[1]:.4f} | Base Grace: {result.x[4]:.2f} | Deg: {result.x[8]:.4f}")
    print(f"Laps lost per degree over 25C: {result.x[5]:.4f}")
    print(f"Wear Curve Exponent: {result.x[9]:.4f}")
    print(f"Total Error Points: {result.fun}")

if __name__ == '__main__':
    main()