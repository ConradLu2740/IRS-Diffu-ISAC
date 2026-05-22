import numpy as np
import os

def chamfer_distance(pc1, pc2):
    """Compute symmetric Chamfer distance between two point clouds."""
    # pc1: (N, D), pc2: (M, D)
    # For each point in pc1, find nearest in pc2
    diff1 = np.expand_dims(pc1, axis=1) - np.expand_dims(pc2, axis=0)  # (N, M, D)
    dist1_sq = np.sum(diff1 ** 2, axis=2)  # (N, M)
    min_dist1 = np.min(dist1_sq, axis=1)   # (N,)
    
    # For each point in pc2, find nearest in pc1
    diff2 = np.expand_dims(pc2, axis=1) - np.expand_dims(pc1, axis=0)  # (M, N, D)
    dist2_sq = np.sum(diff2 ** 2, axis=2)  # (M, N)
    min_dist2 = np.min(dist2_sq, axis=1)   # (M,)
    
    cd = np.mean(min_dist1) + np.mean(min_dist2)
    return cd, np.mean(min_dist1), np.mean(min_dist2)

def describe_array(arr, name):
    """Describe array statistics."""
    print(f"  {name}:")
    print(f"    Shape:      {arr.shape}")
    print(f"    Num points: {arr.shape[0]}")
    print(f"    Dtype:      {arr.dtype}")
    if arr.ndim == 2:
        print(f"    X: min={arr[:,0].min():.6f}, max={arr[:,0].max():.6f}, mean={arr[:,0].mean():.6f}")
        print(f"    Y: min={arr[:,1].min():.6f}, max={arr[:,1].max():.6f}, mean={arr[:,1].mean():.6f}")
        if arr.shape[1] >= 3:
            print(f"    Z: min={arr[:,2].min():.6f}, max={arr[:,2].max():.6f}, mean={arr[:,2].mean():.6f}")
    else:
        print(f"    All: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")

print("=" * 80)
print("POINT CLOUD STATISTICS")
print("=" * 80)

# Pair 1: PC_gt_0001 vs PC_hat_0001
print("\n--- Pair 1: PC_gt_0001 vs PC_hat_0001 ---")
gt1 = np.load(r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_npy\PC_gt_0001.npy")
pred1 = np.load(r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_npy\PC_hat_0001.npy")

describe_array(gt1, "GT")
describe_array(pred1, "Pred")
cd1, cd1_gt2pred, cd1_pred2gt = chamfer_distance(gt1, pred1)
print(f"    Chamfer Distance (GT<->Pred): {cd1:.8f}")
print(f"    Chamfer GT->Pred (mean):      {cd1_gt2pred:.8f}")
print(f"    Chamfer Pred->GT (mean):      {cd1_pred2gt:.8f}")

# Pair 2: ldm_no_irs gt vs pred
print("\n--- Pair 2: LDM No-IRS sample_0001_gt vs sample_0001_pred ---")
gt2 = np.load(r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_ldm_no_irs\result\sample_0001_gt.npy")
pred2 = np.load(r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_ldm_no_irs\result\sample_0001_pred.npy")

describe_array(gt2, "GT")
describe_array(pred2, "Pred")
cd2, cd2_gt2pred, cd2_pred2gt = chamfer_distance(gt2, pred2)
print(f"    Chamfer Distance (GT<->Pred): {cd2:.8f}")
print(f"    Chamfer GT->Pred (mean):      {cd2_gt2pred:.8f}")
print(f"    Chamfer Pred->GT (mean):      {cd2_pred2gt:.8f}")

# Pair 3: ldm_random_irs gt vs pred
print("\n--- Pair 3: LDM Random-IRS sample_0001_gt vs sample_0001_pred ---")
gt3 = np.load(r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_ldm_random_irs\result\sample_0001_gt.npy")
pred3 = np.load(r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_ldm_random_irs\result\sample_0001_pred.npy")

describe_array(gt3, "GT")
describe_array(pred3, "Pred")
cd3, cd3_gt2pred, cd3_pred2gt = chamfer_distance(gt3, pred3)
print(f"    Chamfer Distance (GT<->Pred): {cd3:.8f}")
print(f"    Chamfer GT->Pred (mean):      {cd3_gt2pred:.8f}")
print(f"    Chamfer Pred->GT (mean):      {cd3_pred2gt:.8f}")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE - Chamfer Distance")
print("=" * 80)
print(f"{'Pair':<45} {'CD_total':>14} {'CD_gt->pred':>14} {'CD_pred->gt':>14}")
print("-" * 87)
print(f"{'1: PC_gt vs PC_hat':<45} {cd1:>14.8f} {cd1_gt2pred:>14.8f} {cd1_pred2gt:>14.8f}")
print(f"{'2: LDM No-IRS gt vs pred':<45} {cd2:>14.8f} {cd2_gt2pred:>14.8f} {cd2_pred2gt:>14.8f}")
print(f"{'3: LDM Random-IRS gt vs pred':<45} {cd3:>14.8f} {cd3_gt2pred:>14.8f} {cd3_pred2gt:>14.8f}")

# ===================================================================
# LOSS HISTORY FILES
# ===================================================================
print("\n" + "=" * 80)
print("LOSS HISTORY STATISTICS")
print("=" * 80)

loss_files = [
    (r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\model\ldm_history.npy", "Train LDM"),
    (r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_ldm_no_irs\check\ldm_history.npy", "LDM No-IRS"),
    (r"c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete\outputs_ldm_random_irs\check\ldm_history.npy", "LDM Random-IRS"),
]

for path, label in loss_files:
    print(f"\n--- {label} ---")
    print(f"  File: {path}")
    data = np.load(path, allow_pickle=True)
    print(f"  Type: {type(data)}")
    
    if isinstance(data, np.ndarray):
        if data.dtype == np.dtype('O'):
            # It's an object array, likely contains a dict
            item = data.item()
            if isinstance(item, dict):
                data = item
            else:
                print(f"  Shape: {data.shape}")
                print(f"  Content (object array, first element type: {type(data.flat[0])})")
                continue
        else:
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min:   {data.min()}")
            print(f"  Max:   {data.max()}")
            print(f"  Mean:  {data.mean()}")
            
            if data.ndim >= 1:
                print(f"  First 5 values: {data.flatten()[:5]}")
                print(f"  Last 5 values:  {data.flatten()[-5:]}")
            
            if data.ndim >= 2:
                for col in range(min(data.shape[1], 5)):
                    col_data = data[:, col]
                    print(f"  Column {col}: min={col_data.min():.6f}, max={col_data.max():.6f}, mean={col_data.mean():.6f}, final={col_data[-1]:.6f}")
            continue
    
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                arr = np.asarray(v, dtype=np.float64)
                print(f"  [{k}]: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
                print(f"         First 3: {arr.flatten()[:3]}, Last 3: {arr.flatten()[-3:]}")
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=np.float64)
                print(f"  [{k}]: len={len(arr)}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
                if len(arr) > 0:
                    print(f"         First 3: {arr[:3]}, Last 3: {arr[-3:]}")
            else:
                print(f"  [{k}]: {v}")
    elif isinstance(data, (list, tuple)):
        arr = np.asarray(data, dtype=np.float64)
        print(f"  Length: {len(arr)}")
        print(f"  Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}")
    else:
        print(f"  Content: {data}")

print("\nDone.")
