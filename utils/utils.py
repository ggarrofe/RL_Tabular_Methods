import pandas as pd

def append_results(total_rewards, iteration=None, df=None, epsilon=None, alpha=None, mse=None, classifier=None):
    
    results = {
        "Total rewards": total_rewards, 
        "Episodes": list(range(1, len(total_rewards)+1))
    }

    if iteration is not None: results["Iteration"] = [iteration]*len(total_rewards)
    if epsilon is not None: results["Epsilon"] = [epsilon]*len(total_rewards)
    if alpha is not None: results["Alpha"] = [alpha]*len(total_rewards)
    if epsilon is not None: results["epsilon"] = [epsilon]*len(total_rewards)
    if mse is not None: results["MSE"] = mse
    if classifier is not None: results["Classifier"] = [classifier]*len(total_rewards)

    try:
        df = df.append(pd.DataFrame(results), ignore_index=True)
    except (NameError, AttributeError) as e:
        df = pd.DataFrame(results)

    return df
