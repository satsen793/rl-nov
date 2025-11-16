"""results_pipeline.py
Utility to run experiments or aggregate existing results from `rl_education_suite.py`.

Usage (on Lightning AI):
  python results_pipeline.py --steps 30000 --seeds 5 --outdir results

The script will:
- Run each algorithm (dqn_per, ppo, pets, mbpo) for given seeds (unless JSON results exist)
- Compute metrics: AUC @10k steps, time-to-mastery (min per run), final return @30k, variance across seeds, wall-clock
- Save per-run CSV and LaTeX-formatted tables: `tables.tex` and CSVs under `results/`

Note: running full experiments can be compute-intensive; you can instead generate results
by running experiments separately and placing per-run JSON results in `results/<alg>_<seed>.json`.
"""
import os
import json
import argparse
import numpy as np
from collections import defaultdict

try:
    # import training functions from repo
    from rl_education_suite import train_dqn_per, train_ppo, train_pets, train_mbpo, EduEnv
except Exception as e:
    print("Warning: could not import training functions from rl_education_suite.py:", e)
    train_dqn_per = train_ppo = train_pets = train_mbpo = None

ALG_FUNCS = {
    "dqn_per": train_dqn_per,
    "ppo": train_ppo,
    "pets": train_pets,
    "mbpo": train_mbpo,
}


def auc_at_k(learning_curve, k=10000):
    # learning_curve is cumulative reward per step (running sum); we integrate first k points
    n = min(len(learning_curve), k)
    if n == 0:
        return float('nan')
    x = np.arange(n)
    y = np.array(learning_curve[:n], dtype=np.float64)
    return float(np.trapz(y, x) / float(max(1, n)))


def summarize_algo_results(results_list, auc_k=10000, final_step=30000):
    # results_list: list of dicts returned by training functions
    metrics = []
    for res in results_list:
        lc = res.get('learning_curve', [])
        auc = auc_at_k(lc, k=auc_k)
        ttm_arr = res.get('time_to_mastery', np.array([np.nan]))
        ttm = float(np.nanmin(ttm_arr)) if np.any(np.isfinite(ttm_arr)) else float('nan')
        final = float(lc[min(len(lc)-1, final_step-1)]) if len(lc) > 0 else float('nan')
        var = float(np.var(lc, ddof=1)) if len(lc) > 1 else 0.0
        wall = float(res.get('wall_clock', float('nan')))
        metrics.append({
            'AUC': auc,
            'TTM': ttm,
            'FinalReturn': final,
            'Var': var,
            'Wall': wall,
        })
    return metrics


def save_csv(results_map, outdir):
    os.makedirs(outdir, exist_ok=True)
    for alg, metrics in results_map.items():
        path = os.path.join(outdir, f"{alg}_by_seed.csv")
        with open(path, 'w') as f:
            f.write('seed,AUC,TTM,FinalReturn,Var,Wall\n')
            for i, m in enumerate(metrics):
                f.write(f"{i},{m['AUC']},{m['TTM']},{m['FinalReturn']},{m['Var']},{m['Wall']}\n")
    print(f"Saved CSVs to {outdir}")


def make_latex_table(results_map, outpath, auc_k=10000):
    # Create three tables similar to the figures in your attachment
    # Table I: metrics by algorithm (AUC @10k, Time-to-Mastery, Final Return @30k, Variance, Compute)
    algs = list(results_map.keys())
    with open(outpath, 'w') as f:
        f.write('% Auto-generated tables\n')
        f.write('\\begin{table}[ht]\\centering\n')
        f.write('\\caption{DQN vs PPO on the simulated education MDP}\\label{tab:alg_compare}\n')
        f.write('\\begin{tabular}{lrrrrr}\\toprule\n')
        f.write('Metric & ' + ' & '.join([a.upper() for a in algs]) + ' \\\\ \\midrule\n')
        # AUC
        f.write('AUC (reward) @%dk steps &' % auc_k)
        f.write(' & '.join([f"{np.mean([m['AUC'] for m in results_map[a]]):.3f}" for a in algs]))
        f.write(' \\\\ \n')
        # Time to Mastery
        f.write('Time-to-Mastery (steps) & ')
        f.write(' & '.join([f"{np.nanmean([m['TTM'] for m in results_map[a]]):.1f}" for a in algs]))
        f.write(' \\\\ \n')
        # Final Return
        f.write('Final Return @30k steps & ')
        f.write(' & '.join([f"{np.mean([m['FinalReturn'] for m in results_map[a]]):.3f}" for a in algs]))
        f.write(' \\\\ \n')
        # Variance
        f.write('Reward Variance (seeds) & ')
        f.write(' & '.join([f"{np.nanvar([m['FinalReturn'] for m in results_map[a]], ddof=1):.3f}" for a in algs]))
        f.write(' \\\\ \n')
        # Wall-clock
        f.write('Compute (wall-clock, s) & ')
        f.write(' & '.join([f"{np.mean([m['Wall'] for m in results_map[a]]):.1f}" for a in algs]))
        f.write(' \\\\ \n')
        f.write('\\bottomrule\\end{tabular}\\end{table}\n\n')

        # Table II and III can be added similarly; for brevity we only generate Table I here.
    print(f"Wrote LaTeX table to {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--no-run', action='store_true', help='do not run training; only aggregate existing JSONs')
    args = parser.parse_args()

    algorithms = ['dqn_per', 'ppo', 'pets', 'mbpo']
    results_map = defaultdict(list)
    os.makedirs(args.outdir, exist_ok=True)

    for alg in algorithms:
        for s in range(args.seeds):
            json_path = os.path.join(args.outdir, f"{alg}_{s}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as fh:
                    res = json.load(fh)
                results_map[alg].append(res)
                continue

            if args.no_run or ALG_FUNCS.get(alg) is None:
                print(f"Skipping run for {alg} seed {s} (no JSON and no run)")
                continue

            print(f"Running {alg} seed {s} ...")
            fn = ALG_FUNCS[alg]
            # construct a proper env_ctor that builds EduEnv with the given seed
            def make_env_ctor(sd):
                return lambda seed=sd: EduEnv(K=8, max_steps=100, beta=0.5, gamma=0.05, step_cost=0.01, seed=sd)
            env_ctor = make_env_ctor(s)
            # Some train_* functions expect an env-factory (env_ctor), others expect an env instance.
            # Try calling with the factory first; on failure, call with an env instance.
            try:
                res = fn(env_ctor, steps=args.steps, seed=s)
            except (TypeError, AttributeError) as e:
                try:
                    env = env_ctor()
                    res = fn(env, steps=args.steps, seed=s)
                except Exception as e2:
                    print(f"Error running {alg} seed {s}: {e2}")
                    res = None
            if res is None:
                 print(f"No result for {alg} seed {s}")
                 continue
             with open(json_path, 'w') as fh:
                 json.dump(res, fh)
             results_map[alg].append(res)

    # Summarize
    summarized = {}
    for alg, lst in results_map.items():
        summarized[alg] = summarize_algo_results(lst)

    save_csv(summarized, args.outdir)
    make_latex_table(summarized, os.path.join(args.outdir, 'tables.tex'))


if __name__ == '__main__':
    main()
