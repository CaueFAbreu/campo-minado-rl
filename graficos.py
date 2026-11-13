"""Gera os dois gráficos dos slides a partir de resultados/.

    python graficos.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = Path("resultados")
FUNDO, TINTA, SUAVE, GRADE = "#FAF8F2", "#16324F", "#566573", "#DDE3E8"
COR = {"Ao acaso": "#8A9AA6", "Resolvedor lógico": "#B84A00", "DQN": "#16324F"}
NOME_TAB = {"5x5": "5×5, 4 minas", "9x9": "9×9, 10 minas"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "text.color": TINTA,
    "axes.edgecolor": GRADE, "axes.labelcolor": SUAVE, "xtick.color": SUAVE, "ytick.color": SUAVE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
})


def pct(v, casas=0):
    return f"{100 * v:.{casas}f}%".replace(".", ",")


def eixo_porcentagem(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.grid(axis="y", color=GRADE, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def curva(resumo):
    fig, eixos = plt.subplots(1, 2, figsize=(16, 5.8), sharey=True)
    for ax, tab in zip(eixos, NOME_TAB):
        dados = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((PASTA / tab).glob("dqn_semente*.json"))]
        corridas = [d["curva"] for d in dados]
        # Média e faixa enquanto todas as sementes têm dados; depois, a que treinou por mais tempo.
        n = min(len(c) for c in corridas)
        x = np.array([p["episodio"] for p in corridas[0][:n]]) / 1000
        y = np.array([[p["vitorias"] for p in c[:n]] for c in corridas])
        i_longa = max(range(len(corridas)), key=lambda i: len(corridas[i]))
        longa = corridas[i_longa]
        continua = len(longa) - n > 2  # uma semente treinou bem mais que as outras
        x_fim = (longa if continua else corridas[0][:n])[-1]["episodio"] / 1000

        eixo_porcentagem(ax)
        for nome in ("Ao acaso", "Resolvedor lógico"):
            v = resumo[tab][nome]
            ax.axhline(v, color=COR[nome], lw=2)
            # o resolvedor fica à esquerda, onde a curva da DQN ainda está abaixo dele
            esquerda = nome == "Resolvedor lógico"
            ax.text(x_fim * 0.01 if esquerda else x_fim, v + 0.02, f"{nome}: {pct(v)}",
                    ha="left" if esquerda else "right", va="bottom", fontsize=14, color=TINTA)
        ax.fill_between(x, y.min(0), y.max(0), color=COR["DQN"], alpha=0.12, lw=0)
        ax.plot(x, y.mean(0), color=COR["DQN"], lw=2.5, solid_capstyle="round")
        if continua:
            xs = [x[-1]] + [p["episodio"] / 1000 for p in longa[n:]]
            ys = [y.mean(0)[-1]] + [p["vitorias"] for p in longa[n:]]
            ax.plot(xs, ys, color=COR["DQN"], lw=2.5, solid_capstyle="round")
            ax.axvline(x[-1], color=GRADE, lw=1, zorder=0)
            ax.text(x[-1] + x_fim * 0.01, 0.03, f"daqui em diante,\nsó a semente {dados[i_longa]['config']['semente']}", ha="left", va="bottom",
                    fontsize=13, color=SUAVE, linespacing=1.2)
        ax.set_title(NOME_TAB[tab], loc="left", fontsize=18, fontweight="bold", color=TINTA, pad=12)
        ax.set_xlabel("Partidas de treino (mil)")
        ax.set_xlim(0, x_fim)
    eixos[0].set_ylabel("Vitórias em 1.000 tabuleiros de teste")

    linhas = [plt.Line2D([], [], color=COR[k], lw=2.5) for k in COR]
    linhas.append(plt.Rectangle((0, 0), 1, 1, color=COR["DQN"], alpha=0.12))
    fig.legend(linhas, ["Ao acaso", "Resolvedor lógico", "DQN (média das sementes)", "DQN: pior e melhor semente"],
               loc="upper right", ncol=4, frameon=False, fontsize=14, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3)
    fig.savefig(PASTA / "curva_aprendizado.png", dpi=150)
    plt.close(fig)


def comparacao(resumo):
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    eixo_porcentagem(ax)
    largura = 0.2
    nomes = ["Ao acaso", "Resolvedor lógico", "DQN"]
    for g, tab in enumerate(NOME_TAB):
        valores = [resumo[tab]["Ao acaso"], resumo[tab]["Resolvedor lógico"], resumo[tab]["DQN"]["vitorias"]]
        for k, (nome, v) in enumerate(zip(nomes, valores)):
            xc = g + (k - 1) * (largura + 0.03)
            ax.bar(xc, v, largura, color=COR[nome], lw=0)
            ax.text(xc, v + 0.02, pct(v, 1), ha="center", va="bottom", fontsize=16,
                    fontweight="bold" if nome == "DQN" else "normal", color=TINTA)
    ax.set_xticks(range(len(NOME_TAB)))
    ax.set_xticklabels([NOME_TAB[t] for t in NOME_TAB], fontsize=17, color=TINTA)
    ax.set_xlim(-0.6, len(NOME_TAB) - 0.4)
    ax.set_ylabel("Vitórias em 10.000 tabuleiros")
    caixas = [plt.Rectangle((0, 0), 1, 1, color=COR[n]) for n in nomes]
    ax.legend(caixas, nomes, loc="upper left", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(PASTA / "comparacao_final.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    resumo = json.loads((PASTA / "resumo.json").read_text(encoding="utf-8"))
    curva(resumo)
    comparacao(resumo)
    print("gráficos salvos em", PASTA)
