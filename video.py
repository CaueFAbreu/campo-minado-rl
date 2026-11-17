"""Grava um vídeo da DQN jogando em quatro momentos do treino, nos mesmos tabuleiros.

    python video.py --tabuleiro 9x9
Usa as redes salvas com `treinar.py --checkpoints` e gera entrega-02-resultados/treino_<tabuleiro>.mp4.
"""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

from campo_minado.ambiente import FECHADA, CampoMinado
from campo_minado.dqn import PoliticaDQN, RedeQ
from treinar import TABULEIROS

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams["font.family"] = "DejaVu Sans"

# Cores dos slides
FUNDO, TINTA, SUAVE = "#FAF8F2", "#16324F", "#566573"
ABERTA, FECHADA_COR, DESTAQUE, MINA = "#E9EFF3", "#16324F", "#B84A00", "#9A3E00"
COR_NUMERO = {1: "#1F5F8B", 2: "#2E7D4F", 3: "#9A3E00", 4: "#5B2A86"}

FPS = 2
SEMENTES_VIDEO = [10_000_000, 10_000_001, 10_000_002]  # tabuleiros de teste


def _misturar(cor_a, cor_b, t):
    a = [int(cor_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(cor_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * float(t)):02x}" for x, y in zip(a, b))


def desenhar(ax, env, ultima=None, perdeu=False, chances=None):
    """Desenha o tabuleiro. Com `chances`, cada casa fechada é sombreada pela chance de mina estimada."""
    ax.clear()
    ax.set_xlim(0, env.colunas)
    ax.set_ylim(env.linhas, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for l in range(env.linhas):
        for c in range(env.colunas):
            v = env.visivel[l, c]
            mina_visivel = perdeu and env.minas[l, c]
            cor = MINA if (mina_visivel and (l, c) == ultima) else (ABERTA if v != FECHADA else FECHADA_COR)
            if v == FECHADA and chances is not None and not mina_visivel:
                cor = _misturar("#9FB4C8", FECHADA_COR, chances[l * env.colunas + c])
            ax.add_patch(Rectangle((c + 0.04, l + 0.04), 0.92, 0.92, color=cor, lw=0))
            if mina_visivel:
                ax.add_patch(plt.Circle((c + 0.5, l + 0.5), 0.2, color=FUNDO if (l, c) == ultima else "#E07B39"))
            elif v > 0:
                ax.text(c + 0.5, l + 0.53, str(v), ha="center", va="center", fontsize=22 if env.linhas > 5 else 34,
                        fontweight="bold", color=COR_NUMERO.get(int(v), TINTA))
            elif v == FECHADA and chances is not None:
                pc = chances[l * env.colunas + c]
                ax.text(c + 0.5, l + 0.52, f"{pc:.0%}", ha="center", va="center", fontsize=11 if env.linhas > 5 else 16,
                        color=FUNDO if pc > 0.25 else TINTA)
    if ultima is not None:
        l, c = ultima
        ax.add_patch(Rectangle((c + 0.04, l + 0.04), 0.92, 0.92, fill=False, ec=DESTAQUE, lw=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabuleiro", choices=TABULEIROS, default="9x9")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--momentos", type=int, nargs="+", default=[0, 6000, 20000, 100000],
                   help="episódios de treino a mostrar (usa o checkpoint mais próximo)")
    p.add_argument("--resultados", default="resultados")
    args = p.parse_args()

    linhas, colunas, minas = TABULEIROS[args.tabuleiro]
    pasta = Path(args.resultados) / args.tabuleiro
    info = json.loads((pasta / f"dqn_semente{args.semente}.json").read_text(encoding="utf-8"))
    cfg = info["config"]
    taxa_em = {p["episodio"]: p["vitorias"] for p in info["curva"]}
    ckpts = {int(f.stem[2:]): f for f in (pasta / f"checkpoints_semente{args.semente}").glob("ep*.pt")}

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=FUNDO)
    ax = fig.add_axes([0.04, 0.08, 0.5, 0.84])
    texto = fig.add_axes([0.58, 0.08, 0.4, 0.84])
    texto.axis("off")

    saida = Path("entrega-02-resultados") / f"treino_{args.tabuleiro}.mp4"
    escritor = FFMpegWriter(fps=FPS, bitrate=2500)
    with escritor.saving(fig, str(saida), dpi=100):
        for momento in args.momentos:
            ep = min(ckpts, key=lambda e: abs(e - momento))
            rede = RedeQ(cfg["canais"], cfg["camadas"])
            rede.load_state_dict(torch.load(ckpts[ep]))
            politica = PoliticaDQN(rede)
            for n, semente in enumerate(SEMENTES_VIDEO, 1):
                env = CampoMinado(linhas, colunas, minas)
                estado = env.reset(semente)

                def quadro(ultima=None, perdeu=False, resultado="", segurar=1):
                    desenhar(ax, env, ultima, perdeu)
                    texto.clear()
                    texto.axis("off")
                    texto.text(0, 0.95, f"Campo Minado {linhas}×{colunas} · {minas} minas", fontsize=18,
                               color=DESTAQUE, fontweight="bold", transform=texto.transAxes)
                    texto.text(0, 0.86, f"{ep:,} partidas".replace(",", ".") + "\nde treino",
                               fontsize=32, color=TINTA, fontweight="bold", va="top", linespacing=1.15,
                               transform=texto.transAxes)
                    texto.text(0, 0.60, f"Vence {taxa_em.get(ep, 0):.0%} dos tabuleiros de teste",
                               fontsize=20, color=TINTA, transform=texto.transAxes)
                    texto.text(0, 0.47, f"Partida {n} de {len(SEMENTES_VIDEO)}   ·   jogada {env.jogadas}",
                               fontsize=18, color=SUAVE, transform=texto.transAxes)
                    if resultado:
                        texto.text(0, 0.32, resultado, fontsize=30, fontweight="bold",
                                   color=TINTA if "Venceu" in resultado else MINA, transform=texto.transAxes)
                    texto.text(0, 0.02, "Contorno laranja: casa que o agente acabou de abrir.",
                               fontsize=14, color=SUAVE, transform=texto.transAxes)
                    for _ in range(segurar):
                        escritor.grab_frame(facecolor=FUNDO)

                quadro(segurar=2)
                terminou = False
                while not terminou:
                    acao = politica.escolher(estado)
                    estado, _, terminou, i = env.step(acao)
                    ultima = divmod(acao, colunas)
                    if terminou:
                        perdeu = not i["venceu"]
                        quadro(ultima, perdeu, "Abriu uma mina" if perdeu else "Venceu!", segurar=4)
                    else:
                        quadro(ultima)
    print("vídeo salvo em", saida)


if __name__ == "__main__":
    main()
