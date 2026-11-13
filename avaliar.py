"""Compara o acaso, o resolvedor lógico e a DQN nos mesmos 10.000 tabuleiros de teste.

    python avaliar.py

A rede da DQN que entra na comparação é escolhida entre as salvas durante o treino
(`treinar.py --checkpoints`) pela taxa de vitórias em 2.000 tabuleiros de VALIDAÇÃO, que
não fazem parte do teste. Assim a escolha não usa os tabuleiros em que o resultado é medido.
Gera resultados/resumo.json (usado pelos gráficos) e resultados/<tabuleiro>/dqn_final.pt.
"""

import json
import shutil
from pathlib import Path

import torch

from campo_minado.ambiente import CampoMinado
from campo_minado.dqn import RedeQ, avaliar
from campo_minado.politicas import Aleatoria, Resolvedor, jogar
from treinar import TABULEIROS

SEMENTES_TESTE = range(10_000_000, 10_010_000)
SEMENTES_VALIDACAO = range(20_000_000, 20_002_000)
PASTA = Path("resultados")
SEMENTE_DQN = 0  # a execução que salvou as redes intermediárias


def main():
    resumo = {}
    for tab, (linhas, colunas, minas) in TABULEIROS.items():
        env = CampoMinado(linhas, colunas, minas)
        r = {"Ao acaso": jogar(Aleatoria(0), env, SEMENTES_TESTE),
             "Resolvedor lógico": jogar(Resolvedor(0), env, SEMENTES_TESTE)}

        cfg = json.loads((PASTA / tab / f"dqn_semente{SEMENTE_DQN}.json").read_text(encoding="utf-8"))["config"]
        melhor = None
        for arq in sorted((PASTA / tab / f"checkpoints_semente{SEMENTE_DQN}").glob("ep*.pt")):
            rede = RedeQ(cfg["canais"], cfg["camadas"])
            rede.load_state_dict(torch.load(arq))
            validacao = avaliar(rede, linhas, colunas, minas, SEMENTES_VALIDACAO)
            if melhor is None or validacao > melhor[0]:
                melhor = (validacao, arq, rede)
        validacao, arq, rede = melhor
        shutil.copy(arq, PASTA / tab / "dqn_final.pt")
        r["DQN"] = {"vitorias": avaliar(rede, linhas, colunas, minas, SEMENTES_TESTE),
                    "semente": SEMENTE_DQN, "episodio": int(arq.stem[2:]), "validacao": validacao}
        resumo[tab] = r
        print(tab, r, flush=True)
    (PASTA / "resumo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
