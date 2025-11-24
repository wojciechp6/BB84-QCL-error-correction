import numpy as np
import matplotlib.pyplot as plt


def plt_fab_fae(
    bob_qbers,
    eve_qbers,
    target_fab,
    title="BB84 QCL – porównanie z PCCM",
):
    """
    bob_qbers, eve_qbers – listy/ndarray z końcowymi QBER-ami
                           (np. z różnych f_value albo różnych ataków)
    labels – opcjonalnie etykiety dla każdego punktu
    """

    bob_qbers = np.asarray(bob_qbers, dtype=float)
    eve_qbers = np.asarray(eve_qbers, dtype=float)

    target = (target_fab, FAE_max(target_fab))

    # Fidelności z QBER-ów
    fab = 1.0 - bob_qbers
    fae = 1.0 - eve_qbers

    # Krzywa PCCM (analityczna), jak w artykule:
    theta = np.linspace(0, np.pi / 2, 400)
    fab_pccm = (1.0 + np.cos(theta)) / 2.0
    fae_pccm = (1.0 + np.sin(theta)) / 2.0

    plt.figure(figsize=(6, 5))

    # Krzywa teoretyczna
    plt.plot(
        fab_pccm,
        fae_pccm,
        label="PCCM (analityczne)",
        linewidth=2,
    )

    plt.scatter(fab, fae, s=40, zorder=3, alpha=0.3, label="symulacje")
    plt.scatter(
        *target,
        s=100,
        zorder=4,
        label="docelowy punkt",
        color="red",
        marker="x",
    )

    plt.xlabel(r"$F_{AB}$")
    plt.ylabel(r"$F_{AE}$")
    plt.title(title)

    # zakresy osi podobne do Fig.4
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    return plt

def FAE_max(FAB):
    cos_theta = 2*FAB - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    return (1 + sin_theta) / 2
