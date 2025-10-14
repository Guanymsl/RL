import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

os.environ['WANDB_API_KEY'] = '4459c56f2331259c595f33fc6b3e1191f07ad335'

wandb.login(key=os.environ['WANDB_API_KEY'])
wandb.init(
    project="rl_hw2",
    name="rl"
)

def last10_avg(arr):
    arr = np.array(arr)
    result = np.empty_like(arr, dtype=float)
    result[:] = np.nan

    for i in range(9, len(arr)):
        result[i] = np.mean(arr[i-9:i+1])

    return result

def plotwandb(
    MC_Bias, MC_Var, TD0_Bias, TD0_Var,
    MC_learn, MC_el, SARSA_learn, SARSA_el, Q_learn, Q_el
):
    states = np.arange(len(MC_Bias))
    width = 0.35

    _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(states - width/2, MC_Bias, width, label='MC', alpha=0.8, color='orange')
    ax.bar(states + width/2, TD0_Bias, width, label='TD(0)', alpha=0.8, color='green')

    ax.set_xlabel('State')
    ax.set_title('Bias')
    ax.legend(loc='upper left')
    plt.tight_layout()

    _, ay = plt.subplots(figsize=(6, 4))
    ay.bar(states - width/2, MC_Var, width, label='MC', alpha=0.8, color='orange')
    ay.bar(states + width/2, TD0_Var, width, label='TD(0)', alpha=0.8, color='green')

    ay.set_xlabel('State')
    ay.set_title('Variance')
    ay.legend(loc='upper left')
    plt.tight_layout()

    plt.show()

    def log_eps_curves(prefix, list):
        epsilons = [0.1, 0.2, 0.3, 0.4]
        length = len(list[0])
        for t in range(length):
            log_dict = {}
            for i, eps in enumerate(epsilons):
                log_dict[f"{prefix}/ε={eps}"] = list[i][t]
            wandb.log(log_dict)

    log_eps_curves("MC", MC_learn)
    log_eps_curves("MC", MC_el)
    log_eps_curves("SARSA", SARSA_learn)
    log_eps_curves("SARSA", SARSA_el)
    log_eps_curves("Q-Learning", Q_learn)
    log_eps_curves("Q-Learning", Q_el)

if __name__ == "__main__":
    MC_bias     = np.load("./results/Bias_MC.npy")
    MC_var      = np.load("./results/Var_MC.npy")
    TD0_bias    = np.load("./results/Bias_TD.npy")
    TD0_var     = np.load("./results/Var_TD.npy")
    MC_learn    = np.load("./results/MC_learn.npy")
    MC_el       = np.load("./results/MC_el.npy")
    SARSA_learn = np.load("./results/SARSA_learn.npy")
    SARSA_el    = np.load("./results/SARSA_el.npy")
    Q_learn     = np.load("./results/Q_learn.npy")
    Q_el        = np.load("./results/Q_el.npy")

    MC_learn_avg    = [last10_avg(x) for x in MC_learn]
    MC_el_avg       = [last10_avg(x) for x in MC_el]
    SARSA_learn_avg = [last10_avg(x) for x in SARSA_learn]
    SARSA_el_avg    = [last10_avg(x) for x in SARSA_el]
    Q_learn_avg     = [last10_avg(x) for x in Q_learn]
    Q_el_avg        = [last10_avg(x) for x in Q_el]

    plotwandb(
        MC_bias, MC_var, TD0_bias, TD0_var,
        MC_learn_avg, MC_el_avg, SARSA_learn_avg, SARSA_el_avg, Q_learn_avg, Q_el_avg
    )

    print("Plotted to wandb.")
