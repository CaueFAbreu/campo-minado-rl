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
        # Retreinos que continuam a mesma rede entram como continuação da curva, quando começam
        # perto do fim do trecho anterior: primeiro resultados/ajuste, depois resultados/mse.
        retreinos = []  # (partida em que começa, rótulo)
        semente = dados[i_longa]["config"]["semente"]
        for pasta_r, rotulo in (("ajuste", "retreino: simetrias e\ntaxa decrescente →"),
                                ("mse", "erro\nquadrático →")):
            arq = PASTA / pasta_r / tab / f"dqn_semente{semente}.json"
            if not arq.exists():
                continue
            cont = json.loads(arq.read_text(encoding="utf-8"))["curva"]
            if cont[0]["episodio"] >= 0.9 * longa[-1]["episodio"]:
                retreinos.append((cont[0]["episodio"] / 1000, rotulo))
                longa = [p for p in longa if p["episodio"] < cont[0]["episodio"]] + cont
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
        for inicio, rotulo in retreinos:
            ax.axvline(inicio, color=GRADE, lw=1, zorder=0)
            ax.text(inicio - x_fim * 0.008, 0.97, rotulo, ha="right", va="top", fontsize=12.5, color=SUAVE,
                    linespacing=1.15)
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
    """Barras deitadas, um painel por tabuleiro: as referências, a DQN antes e depois das melhorias
    e a combinação da lógica com o chute da DQN."""
    barras = [  # (rótulo, cor, como obter o valor)
        ("Ao acaso", COR["Ao acaso"], lambda r: r["Ao acaso"]),
        ("Resolvedor lógico", COR["Resolvedor lógico"], lambda r: r["Resolvedor lógico"]),
        ("Probabilidade exata", "#6A5A93", lambda r: r["Probabilidade exata"]),
        ("DQN inicial", "#7F9BB8", lambda r: r["variantes"]["DQN"]["vitorias"]),
        ("DQN final", COR["DQN"], lambda r: r["DQN final"]["vitorias"]),
        ("Lógica + chute da DQN", "#2E7D6F", lambda r: r["Lógica + chute da DQN"]),
    ]
    fig, eixos = plt.subplots(2, 1, figsize=(9.6, 6.6), sharex=True)
    for ax, tab in zip(eixos, NOME_TAB):
        y = np.arange(len(barras))[::-1]
        for yi, (rotulo, cor, valor) in zip(y, barras):
            v = valor(resumo[tab])
            ax.barh(yi, v, 0.62, color=cor, lw=0)
            ax.text(v + 0.012, yi, pct(v, 1), va="center", fontsize=15, color=TINTA,
                    fontweight="bold" if rotulo == "DQN final" else "normal")
        ax.set_yticks(y)
        ax.set_yticklabels([b[0] for b in barras], fontsize=15, color=TINTA)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
        ax.grid(axis="x", color=GRADE, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(NOME_TAB[tab], loc="left", fontsize=17, fontweight="bold", color=TINTA, pad=8)
    eixos[-1].set_xlabel("Vitórias em 10.000 tabuleiros de teste")
    fig.tight_layout(h_pad=1.5)
    fig.savefig(PASTA / "comparacao_final.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    resumo = json.loads((PASTA / "resumo.json").read_text(encoding="utf-8"))
    curva(resumo)
    comparacao(resumo)
    print("gráficos salvos em", PASTA)
