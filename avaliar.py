"""Compara o acaso, o resolvedor lógico e as versões da DQN nos mesmos 10.000 tabuleiros de teste.

    python avaliar.py

Para cada treino, a rede comparada é escolhida entre as salvas durante o treino
(`treinar.py --checkpoints`) pela taxa de vitórias em 2.000 tabuleiros de VALIDAÇÃO, que não fazem
parte do teste. A versão chamada de "DQN final" também é escolhida pela validação.
Gera resultados/resumo.json (usado pelos gráficos) e resultados/<tabuleiro>/dqn_final.pt.
"""

import json
import shutil
from pathlib import Path

import torch

from campo_minado.ambiente import CampoMinado
from campo_minado.dqn import LogicaComChuteDQN, RedeQ, avaliar
from campo_minado.politicas import Aleatoria, Probabilistico, Resolvedor, jogar
from treinar import TABULEIROS

SEMENTES_TESTE = range(10_000_000, 10_010_000)
SEMENTES_VALIDACAO = range(20_000_000, 20_002_000)
PASTA = Path("resultados")
SEMENTE_DQN = 0  # a execução que salvou as redes intermediárias

# Treinos da DQN: (nome, pasta com dqn_semente0.json e checkpoints_semente0/)
TREINOS = [
    ("DQN", PASTA),
    ("DQN retreinada", PASTA / "ajuste"),  # simetrias no treino e taxa de aprendizado decrescente
    ("DQN maior", PASTA / "grande"),  # 64 canais e 6 camadas, já com simetrias e taxa decrescente
    ("DQN com erro quadrático", PASTA / "mse"),  # perda MSE em vez de Huber
    ("DQN com erro quadrático e γ = 0", PASTA / "mse_g0"),
]


def melhor_checkpoint(pasta, tab, linhas, colunas, minas):
    cfg = json.loads((pasta / tab / f"dqn_semente{SEMENTE_DQN}.json").read_text(encoding="utf-8"))["config"]
    melhor = None
    for arq in sorted((pasta / tab / f"checkpoints_semente{SEMENTE_DQN}").glob("ep*.pt")):
        rede = RedeQ(cfg["canais"], cfg["camadas"])
        rede.load_state_dict(torch.load(arq))
        validacao = avaliar(rede, linhas, colunas, minas, SEMENTES_VALIDACAO)
        if melhor is None or validacao > melhor[0]:
            melhor = (validacao, arq, rede)
    return melhor


def main():
    resumo = {}
    for tab, (linhas, colunas, minas) in TABULEIROS.items():
        env = CampoMinado(linhas, colunas, minas)
        r = {"Ao acaso": jogar(Aleatoria(0), env, SEMENTES_TESTE),
             "Resolvedor lógico": jogar(Resolvedor(0), env, SEMENTES_TESTE),
             "Probabilidade exata": jogar(Probabilistico(minas), env, SEMENTES_TESTE),
             "variantes": {}}
        redes = {}
        for nome, pasta in TREINOS:
            if not (pasta / tab / f"checkpoints_semente{SEMENTE_DQN}").exists():
                continue
            _, arq, rede = melhor_checkpoint(pasta, tab, linhas, colunas, minas)
            for simetria in (False, True):
                rotulo = nome + (" + média nas simetrias" if simetria else "")
                r["variantes"][rotulo] = {
                    "vitorias": avaliar(rede, linhas, colunas, minas, SEMENTES_TESTE, simetria=simetria),
                    "validacao": avaliar(rede, linhas, colunas, minas, SEMENTES_VALIDACAO, simetria=simetria),
                    "checkpoint": str(arq.relative_to(PASTA)).replace("\\", "/"),
                    "simetria": simetria,
                }
                redes[rotulo] = (rede, arq, simetria)
                print(tab, rotulo, r["variantes"][rotulo], flush=True)

        # Versão final: a de melhor validação
        final = max(r["variantes"], key=lambda k: r["variantes"][k]["validacao"])
        rede, arq, simetria = redes[final]
        shutil.copy(arq, PASTA / tab / "dqn_final.pt")
        r["DQN final"] = dict(r["variantes"][final], versao=final)
        r["Lógica + chute da DQN"] = jogar(LogicaComChuteDQN(rede, simetria), env, SEMENTES_TESTE)
        print(tab, "final:", final, "| lógica + chute da DQN:", r["Lógica + chute da DQN"], flush=True)
        resumo[tab] = r
    (PASTA / "resumo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
