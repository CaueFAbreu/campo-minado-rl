"""Grava um vídeo da DQN final jogando tabuleiros novos, um atrás do outro.

    python video_final.py                 # 9×9, 6 partidas
    python video_final.py --tabuleiro 5x5 --partidas 8

Os tabuleiros são os primeiros de uma faixa de sementes não usada no treino, na validação, no
teste nem na demo: nenhuma partida é escolhida a dedo. Cada casa fechada mostra a chance de mina
que a DQN estima (quanto mais escura, mais perigosa).
"""

import argparse
import json
from pathlib import Path

import torch
from matplotlib.animation import FFMpegWriter

from campo_minado.ambiente import CampoMinado
from campo_minado.dqn import PoliticaDQN, valores_q
from demo import carregar_rede, chance_de_mina
from treinar import TABULEIROS
from video import DESTAQUE, FPS, FUNDO, MINA, SUAVE, TINTA, desenhar, plt

SEMENTE_INICIAL = 60_000_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabuleiro", choices=TABULEIROS, default="9x9")
    p.add_argument("--partidas", type=int, default=6)
    p.add_argument("--saida", default="entrega-02-resultados")
    args = p.parse_args()

    linhas, colunas, minas = TABULEIROS[args.tabuleiro]
    rede, simetria = carregar_rede(args.tabuleiro)
    politica = PoliticaDQN(rede, simetria)
    taxa = json.loads(Path("resultados/resumo.json").read_text(encoding="utf-8"))[args.tabuleiro]["DQN final"]["vitorias"]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=FUNDO)
    ax = fig.add_axes([0.04, 0.08, 0.5, 0.84])
    texto = fig.add_axes([0.58, 0.08, 0.4, 0.84])
    texto.axis("off")

    Path(args.saida).mkdir(parents=True, exist_ok=True)
    saida = Path(args.saida) / f"agente_final_{args.tabuleiro}.mp4"
    vitorias = 0
    escritor = FFMpegWriter(fps=FPS, bitrate=2500)
    with escritor.saving(fig, str(saida), dpi=100):
        for n in range(1, args.partidas + 1):
            env = CampoMinado(linhas, colunas, minas)
            estado = env.reset(SEMENTE_INICIAL + n - 1)

            def quadro(ultima=None, perdeu=False, resultado="", segurar=1, mostrar_chances=True):
                chances = None
                if mostrar_chances:
                    with torch.no_grad():
                        chances = chance_de_mina(valores_q(rede, env.visivel[None], simetria)[0].numpy())
                desenhar(ax, env, ultima, perdeu, chances)
                texto.clear()
                texto.axis("off")
                texto.text(0, 0.95, f"Campo Minado {linhas}×{colunas} · {minas} minas", fontsize=18,
                           color=DESTAQUE, fontweight="bold", transform=texto.transAxes)
                texto.text(0, 0.86, "DQN final", fontsize=32, color=TINTA, fontweight="bold", va="top",
                           transform=texto.transAxes)
                texto.text(0, 0.70, "Vence " + f"{taxa:.1%}".replace(".", ",") + " dos 10.000\ntabuleiros de teste",
                           fontsize=20, color=TINTA, va="top", linespacing=1.2, transform=texto.transAxes)
                texto.text(0, 0.50, f"Partida {n} de {args.partidas}   ·   jogada {env.jogadas}",
                           fontsize=18, color=SUAVE, transform=texto.transAxes)
                texto.text(0, 0.42, f"Placar: {vitorias} vitória{'s' if vitorias != 1 else ''}"
                                    f" em {n - 1 + bool(resultado)} partida{'s' if n - 1 + bool(resultado) != 1 else ''}",
                           fontsize=18, color=SUAVE, transform=texto.transAxes)
                if resultado:
                    texto.text(0, 0.28, resultado, fontsize=30, fontweight="bold",
                               color=TINTA if "Venceu" in resultado else MINA, transform=texto.transAxes)
                texto.text(0, 0.02, "Casas fechadas: chance de mina estimada\npela DQN (mais escura = mais perigosa).\n"
                                    "Contorno laranja: última jogada.",
                           fontsize=13, color=SUAVE, linespacing=1.3, transform=texto.transAxes)
                for _ in range(segurar):
                    escritor.grab_frame(facecolor=FUNDO)

            quadro(segurar=2)
            terminou = False
            while not terminou:
                acao = politica.escolher(estado)
                estado, _, terminou, info = env.step(acao)
                ultima = divmod(acao, colunas)
                if terminou:
                    vitorias += info["venceu"]
                    perdeu = not info["venceu"]
                    quadro(ultima, perdeu, "Abriu uma mina" if perdeu else "Venceu!", segurar=4,
                           mostrar_chances=False)
                else:
                    quadro(ultima)
    print(f"vídeo salvo em {saida} ({vitorias} vitórias em {args.partidas} partidas)")


if __name__ == "__main__":
    main()
