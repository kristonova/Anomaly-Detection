import os
import glob
from collections import defaultdict

def profile_dataset(base_path, is_eval=False):
    print(f"\nProfiling {base_path}...")
    train_path = os.path.join(base_path, "train", "*.wav")
    test_path = os.path.join(base_path, "test", "*.wav")
    
    train_files = glob.glob(train_path)
    test_files = glob.glob(test_path)
    
    print(f"Found {len(train_files)} training files.")
    print(f"Found {len(test_files)} testing files.")
    
    stats = defaultdict(lambda: {'train_normal': 0, 'test_unknown': 0, 'test_normal': 0, 'test_anomaly': 0})
    
    for f in train_files:
        basename = os.path.basename(f)
        parts = basename.replace('.wav', '').split('_')
        if len(parts) >= 4:
            condition = parts[0]
            machine_id = parts[2]
            if condition == "normal":
                stats[machine_id]['train_normal'] += 1
            else:
                print(f"WARNING: Anomalous file in train set! {basename}")
                
    for f in test_files:
        basename = os.path.basename(f)
        parts = basename.replace('.wav', '').split('_')
        
        if is_eval:
            # Eval files might not have condition in name, e.g., id_01_00000000.wav
            if len(parts) == 3 and parts[0] == "id":
                machine_id = parts[1]
                stats[machine_id]['test_unknown'] += 1
            else:
                print(f"WARNING: Unexpected format in eval test set! {basename}")
        else:
            if len(parts) >= 4:
                condition = parts[0]
                machine_id = parts[2]
                if condition == "normal":
                    stats[machine_id]['test_normal'] += 1
                elif condition == "anomaly":
                    stats[machine_id]['test_anomaly'] += 1
                else:
                    print(f"WARNING: Unknown condition in test set! {basename}")
                
    print("\n--- Dataset Profile ---")
    if is_eval:
        print(f"{'Machine ID':<12} | {'Train (Normal)':<15} | {'Test (Unknown)':<15}")
        print("-" * 50)
        for m_id in sorted(stats.keys()):
            s = stats[m_id]
            print(f"ID {m_id:<9} | {s['train_normal']:<15} | {s['test_unknown']:<15}")
    else:
        print(f"{'Machine ID':<12} | {'Train (Normal)':<15} | {'Test (Normal)':<15} | {'Test (Anomaly)':<15}")
        print("-" * 65)
        for m_id in sorted(stats.keys()):
            s = stats[m_id]
            print(f"ID {m_id:<9} | {s['train_normal']:<15} | {s['test_normal']:<15} | {s['test_anomaly']:<15}")

if __name__ == "__main__":
    dev_dir = "/Users/mymac/Study Abroad/Master Computer Science EURECOM/AML/Anomaly Detection/dataset/dev_data/dev_data/slider"
    eval_dir = "/Users/mymac/Study Abroad/Master Computer Science EURECOM/AML/Anomaly Detection/dataset/eval_data/eval_data/slider"
    
    profile_dataset(dev_dir, is_eval=False)
    profile_dataset(eval_dir, is_eval=True)
