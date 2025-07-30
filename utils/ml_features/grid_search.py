from sklearn.model_selection import cross_val_score
from itertools import product
from tqdm import tqdm
import time
import numpy as np

def grid_search(model_class, param_grid, X, y, cv=5, scoring="accuracy"):
    total_time = 0

    best_score = -np.inf
    best_model = None
    best_params = None

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(product(*values))

    print("Denenecek hiperparametre kombinasyon sayısı:", len(all_combos))

    for combo in tqdm(all_combos, desc=f"{model_class.__class__.__name__} Grid Search"):
        params = dict(zip(keys, combo))
        model = model_class.set_params(**params)

        start = time.time()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        duration = time.time() - start
        total_time += duration

        avg_score = np.mean(scores)
        print("Parametreler:", params, "| Skor:", avg_score, "| Süre:", format_duration(duration))

        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_params = params

    best_model.fit(X, y)
    return best_model, best_score, best_params, format_duration(total_time)

def format_duration(seconds):
    if seconds < 60:
        secs = int(seconds // 1)
        return str(secs) + " sn"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return str(minutes) + " dk " + str(secs) + " sn"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return str(hours) + " sa " + str(minutes) + " dk " + str(secs) + " sn"