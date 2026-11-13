"""Treina a DQN no Campo Minado e salva a curva de aprendizado e a rede.

Exemplos:
    python treinar.py --tabuleiro 5x5 --semente 0
    python treinar.py --tabuleiro 9x9 --semente 0
    python treinar.py --tabuleiro 9x9 --checkpoints --continuar-de resultados/9x9/checkpoints_semente0/ep0028000.pt
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from campo_minado.ambiente import CampoMinado
from campo_minado.dqn import Memoria, RedeQ, aumentar, avaliar, codificar, mascara_fechadas, melhores_acoes

TABULEIROS = {"5x5": (5, 5, 4), "9x9": (9, 9, 10)}
SEMENTES_TESTE = range(10_000_000, 10_001_000)  # nunca usadas no treino


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabuleiro", choices=TABULEIROS, default="5x5")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--episodios", type=int, default=100_000)
    p.add_argument("--paralelos", type=int, default=32, help="partidas jogadas ao mesmo tempo")
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=None,
                   help="se informado, a taxa de aprendizado cai (cosseno) de --lr até este valor")
    p.add_argument("--aumentar", action="store_true",
                   help="gira/espelha cada lote do replay ao acaso (o jogo é simétrico)")
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--canais", type=int, default=32)
    p.add_argument("--camadas", type=int, default=5)
    p.add_argument("--memoria", type=int, default=200_000)
    p.add_argument("--epsilon-final", type=float, default=0.01)
    p.add_argument("--fracao-exploracao", type=float, default=0.3,
                   help="fração dos episódios em que epsilon cai de 1 até o valor final")
    p.add_argument("--atualizar-alvo", type=int, default=1000, help="passos de treino entre cópias")
    p.add_argument("--avaliar-cada", type=int, default=2000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--checkpoints", action="store_true",
                   help="salva a rede em cada avaliação (usado no vídeo)")
    p.add_argument("--continuar-de", metavar="CHECKPOINT",
                   help="retoma o treino de uma rede salva (o número de partidas vem do nome, ex.: ep0028000.pt)")
    p.add_argument("--saida", default="resultados")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.semente)
    rng = np.random.default_rng(args.semente)
    linhas, colunas, minas = TABULEIROS[args.tabuleiro]

    online = RedeQ(args.canais, args.camadas)
    alvo = RedeQ(args.canais, args.camadas)
    alvo.load_state_dict(online.state_dict())
    otimizador = torch.optim.Adam(online.parameters(), lr=args.lr)
    memoria = Memoria(args.memoria, (linhas, colunas))

    envs = [CampoMinado(linhas, colunas, minas) for _ in range(args.paralelos)]
    for i, env in enumerate(envs):  # sementes de treino separadas das de teste
        env.rng = np.random.default_rng([args.semente, i])
    estados = np.stack([env.reset() for env in envs])

    episodios, passos_treino = 0, 0
    proxima_avaliacao = 0
    curva = []
    pasta = Path(args.saida) / args.tabuleiro
    pasta_ckpt = pasta / f"checkpoints_semente{args.semente}"
    if args.checkpoints:
        pasta_ckpt.mkdir(parents=True, exist_ok=True)
    vitorias_recentes = []
    inicio = time.time()
    nome = f"dqn_semente{args.semente}"

    if args.continuar_de:
        # Retoma da rede salva. O replay e o otimizador recomeçam vazios; epsilon segue o cronograma.
        online.load_state_dict(torch.load(args.continuar_de))
        alvo.load_state_dict(online.state_dict())
        episodios = proxima_avaliacao = int(re.search(r"ep(\d+)", Path(args.continuar_de).stem)[1])
        anterior = pasta / f"{nome}.json"
        if anterior.exists():
            dados = json.loads(anterior.read_text(encoding="utf-8"))
            curva = [p for p in dados["curva"] if p["episodio"] < episodios]
            inicio -= dados.get("tempo_s", 0)
        print(f"retomando de {args.continuar_de} ({episodios} partidas)", flush=True)
    episodio_inicial = episodios

    def calcular_epsilon():
        return max(args.epsilon_final,
                   1 - (1 - args.epsilon_final) * episodios / (args.fracao_exploracao * args.episodios))

    def salvar():
        """Salva a curva e a rede a cada avaliação: se o treino parar, o que já foi feito fica salvo."""
        pasta.mkdir(parents=True, exist_ok=True)
        torch.save(online.state_dict(), pasta / f"{nome}.pt")
        with open(pasta / f"{nome}.json", "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "curva": curva,
                       "tempo_s": round(time.time() - inicio)}, f, indent=1)

    while episodios < args.episodios:
        # Avaliação periódica nos tabuleiros de teste, sem exploração
        epsilon = calcular_epsilon()
        if episodios >= proxima_avaliacao:
            taxa = avaliar(online, linhas, colunas, minas, SEMENTES_TESTE)
            curva.append({"episodio": episodios, "vitorias": taxa})
            treino = np.mean(vitorias_recentes[-2000:]) if vitorias_recentes else 0
            print(f"ep {episodios:>7}  teste {taxa:6.1%}  treino {treino:6.1%}  "
                  f"eps {epsilon:.3f}  {time.time() - inicio:6.0f}s" if episodios else
                  f"ep {episodios:>7}  teste {taxa:6.1%}", flush=True)
            proxima_avaliacao += args.avaliar_cada
            if args.checkpoints:
                torch.save(online.state_dict(), pasta_ckpt / f"ep{episodios:07d}.pt")
            salvar()

        # epsilon-guloso, só entre casas fechadas
        mascara = mascara_fechadas(estados)
        with torch.no_grad():
            acoes = melhores_acoes(online(codificar(estados)), mascara).numpy()
        for i in np.flatnonzero(rng.random(len(envs)) < epsilon):
            acoes[i] = rng.choice(np.flatnonzero(mascara[i].numpy()))

        for i, env in enumerate(envs):
            proximo, r, terminou, info = env.step(acoes[i])
            memoria.guardar(estados[i], acoes[i], r, proximo, terminou)
            if terminou:
                episodios += 1
                vitorias_recentes.append(info["venceu"])
                proximo = env.reset()
            estados[i] = proximo

        if memoria.tamanho < 5 * args.lote:
            continue

        # Um passo de treino (Double DQN) por rodada de jogadas
        s, a, r, s2, fim = memoria.amostrar(args.lote, rng)
        if args.aumentar:
            s, a, s2 = aumentar(s, a, s2, rng)
        if args.lr_final is not None:
            progresso = (episodios - episodio_inicial) / (args.episodios - episodio_inicial)
            lr = args.lr_final + (args.lr - args.lr_final) * 0.5 * (1 + math.cos(math.pi * min(progresso, 1)))
            for grupo in otimizador.param_groups:
                grupo["lr"] = lr
        q = online(codificar(s)).gather(1, torch.as_tensor(a)[:, None]).squeeze(1)
        with torch.no_grad():
            mascara2 = mascara_fechadas(s2)
            a2 = melhores_acoes(online(codificar(s2)), mascara2)
            q2 = alvo(codificar(s2)).gather(1, a2[:, None]).squeeze(1)
            fim = torch.as_tensor(fim) | ~mascara2.any(1)
            objetivo = torch.as_tensor(r) + args.gamma * torch.where(fim, 0.0, q2)
        perda = F.smooth_l1_loss(q, objetivo)
        otimizador.zero_grad()
        perda.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 10)
        otimizador.step()
        passos_treino += 1
        if passos_treino % args.atualizar_alvo == 0:
            alvo.load_state_dict(online.state_dict())

    taxa = avaliar(online, linhas, colunas, minas, SEMENTES_TESTE)
    curva.append({"episodio": episodios, "vitorias": taxa})
    print(f"final: {taxa:.1%} de vitórias em {time.time() - inicio:.0f}s", flush=True)
    salvar()


if __name__ == "__main__":
    main()
