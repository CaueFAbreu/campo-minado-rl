"""DQN convolucional para o Campo Minado.

A rede recebe o tabuleiro visível inteiro e devolve um valor Q para cada casa. Ela só usa
convoluções, então cada casa é avaliada pelo mesmo filtro, olhando a região em volta dela.
O agente só escolhe entre casas fechadas (as abertas são mascaradas).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .ambiente import FECHADA, CampoMinado

N_CANAIS_ENTRADA = 10  # casa fechada + números de 0 a 8


def codificar(estados):
    """(B, L, C) com -1..8  ->  (B, 10, L, C) em one-hot."""
    x = torch.as_tensor(np.asarray(estados), dtype=torch.long) + 1
    return F.one_hot(x, N_CANAIS_ENTRADA).permute(0, 3, 1, 2).float()


def mascara_fechadas(estados):
    return torch.as_tensor(np.asarray(estados).reshape(len(estados), -1) == FECHADA)


class RedeQ(nn.Module):
    def __init__(self, canais=64, camadas=6):
        super().__init__()
        partes = [nn.Conv2d(N_CANAIS_ENTRADA, canais, 3, padding=1), nn.ReLU()]
        for _ in range(camadas - 1):
            partes += [nn.Conv2d(canais, canais, 3, padding=1), nn.ReLU()]
        partes.append(nn.Conv2d(canais, 1, 1))
        self.rede = nn.Sequential(*partes)

    def forward(self, x):
        return self.rede(x).flatten(1)  # (B, L*C)


def melhores_acoes(q, mascara):
    return q.masked_fill(~mascara, -torch.inf).argmax(1)


class Memoria:
    """Replay buffer circular."""

    def __init__(self, capacidade, forma):
        self.capacidade, self.tamanho, self.pos = capacidade, 0, 0
        self.s = np.zeros((capacidade, *forma), dtype=np.int8)
        self.s2 = np.zeros((capacidade, *forma), dtype=np.int8)
        self.a = np.zeros(capacidade, dtype=np.int64)
        self.r = np.zeros(capacidade, dtype=np.float32)
        self.fim = np.zeros(capacidade, dtype=bool)

    def guardar(self, s, a, r, s2, fim):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.fim[i] = s, a, r, s2, fim
        self.pos = (i + 1) % self.capacidade
        self.tamanho = min(self.tamanho + 1, self.capacidade)

    def amostrar(self, n, rng):
        i = rng.integers(0, self.tamanho, n)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.fim[i]


@torch.no_grad()
def avaliar(rede, linhas, colunas, minas, sementes, paralelos=200):
    """Taxa de vitórias da política gulosa (sem exploração) nas sementes de teste."""
    rede.eval()
    vitorias = 0
    sementes = list(sementes)
    for inicio in range(0, len(sementes), paralelos):
        lote = sementes[inicio:inicio + paralelos]
        envs = [CampoMinado(linhas, colunas, minas) for _ in lote]
        estados = [env.reset(s) for env, s in zip(envs, lote)]
        ativos = list(range(len(envs)))
        while ativos:
            est = np.stack([estados[i] for i in ativos])
            acoes = melhores_acoes(rede(codificar(est)), mascara_fechadas(est)).tolist()
            seguem = []
            for i, a in zip(ativos, acoes):
                estados[i], _, terminou, info = envs[i].step(a)
                if terminou:
                    vitorias += info["venceu"]
                else:
                    seguem.append(i)
            ativos = seguem
    rede.train()
    return vitorias / len(sementes)


class PoliticaDQN:
    """Adapta a rede treinada à mesma interface das políticas de comparação."""

    nome = "DQN"

    def __init__(self, rede):
        self.rede = rede.eval()

    @torch.no_grad()
    def escolher(self, visivel):
        est = visivel[None]
        return int(melhores_acoes(self.rede(codificar(est)), mascara_fechadas(est))[0])
