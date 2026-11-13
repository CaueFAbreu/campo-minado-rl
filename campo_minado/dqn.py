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


# O jogo não muda se o tabuleiro for girado ou espelhado: são 8 simetrias (4 rotações x espelho).
SIMETRIAS = [(k, espelho) for k in range(4) for espelho in (False, True)]


def transformar(x, k, espelho):
    """Gira as duas últimas dimensões k vezes 90 graus e, se pedido, espelha."""
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if espelho else x


def desfazer(x, k, espelho):
    if espelho:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


def valores_q(rede, estados, simetria=False):
    """Valores Q de cada casa. Com `simetria`, tira a média da rede nas 8 versões do tabuleiro."""
    x = codificar(estados)
    if not simetria:
        return rede(x)
    lado = x.shape[-1]
    total = 0
    for k, espelho in SIMETRIAS:
        q = rede(transformar(x, k, espelho)).view(-1, lado, lado)
        total = total + desfazer(q, k, espelho)
    return (total / len(SIMETRIAS)).flatten(1)


def aumentar(s, a, s2, rng):
    """Aplica a mesma simetria aleatória a um lote do replay (estados, ações e próximos estados)."""
    k, espelho = SIMETRIAS[rng.integers(len(SIMETRIAS))]
    lado = s.shape[-1]
    s = transformar(torch.as_tensor(s), k, espelho).numpy().copy()
    s2 = transformar(torch.as_tensor(s2), k, espelho).numpy().copy()
    antiga = transformar(torch.arange(lado * lado).view(lado, lado), k, espelho).flatten()
    nova = torch.empty_like(antiga)
    nova[antiga] = torch.arange(lado * lado)  # casa antiga -> casa nova
    return s, nova[torch.as_tensor(a)].numpy(), s2


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
def avaliar(rede, linhas, colunas, minas, sementes, paralelos=200, simetria=False):
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
            acoes = melhores_acoes(valores_q(rede, est, simetria), mascara_fechadas(est)).tolist()
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

    def __init__(self, rede, simetria=False):
        self.rede = rede.eval()
        self.simetria = simetria

    @torch.no_grad()
    def escolher(self, visivel):
        est = visivel[None]
        return int(melhores_acoes(valores_q(self.rede, est, self.simetria), mascara_fechadas(est))[0])


class LogicaComChuteDQN:
    """Combinação: abre as casas que a lógica garante e só usa a DQN quando é preciso chutar.

    Mede quanto a DQN acerta nos chutes: é o resolvedor lógico com o chute ao acaso trocado pela
    casa de maior valor Q entre as que a lógica não provou ser mina.
    """

    nome = "Lógica + chute da DQN"

    def __init__(self, rede, simetria=False):
        from .politicas import Resolvedor
        self.rede = rede.eval()
        self.simetria = simetria
        self.resolvedor = Resolvedor()

    @torch.no_grad()
    def escolher(self, visivel):
        minas, seguras = self.resolvedor._deduzir(visivel)
        if seguras:
            return min(seguras)
        q = valores_q(self.rede, visivel[None], self.simetria)[0]
        permitidas = torch.as_tensor(visivel.ravel() == FECHADA)
        permitidas[list(minas)] = False
        return int(q.masked_fill(~permitidas, -torch.inf).argmax())
