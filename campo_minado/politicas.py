"""Políticas de comparação: clicar ao acaso (piso) e o resolvedor lógico (referência alta).

Cada política recebe o tabuleiro visível e devolve a casa que vai abrir.
"""

import numpy as np

from .ambiente import FECHADA


class Aleatoria:
    nome = "Ao acaso"

    def __init__(self, semente=0):
        self.rng = np.random.default_rng(semente)

    def escolher(self, visivel):
        return int(self.rng.choice(np.flatnonzero(visivel.ravel() == FECHADA)))


class Resolvedor:
    """Abre o que é certamente seguro e chuta ao acaso quando não há certeza.

    Usa duas regras sobre cada número aberto:
    1. Se o número é igual ao de casas fechadas em volta, todas são minas; se as minas
       em volta já completam o número, as outras fechadas são seguras.
    2. Se as casas desconhecidas de um número estão todas em volta de outro número,
       a diferença entre os dois diz quantas minas há nas casas que sobram.
    """

    nome = "Resolvedor lógico"

    def __init__(self, semente=0):
        self.rng = np.random.default_rng(semente)

    def escolher(self, visivel):
        linhas, colunas = visivel.shape
        minas, seguras = self._deduzir(visivel)
        if seguras:
            return min(seguras)
        candidatas = [i for i in np.flatnonzero(visivel.ravel() == FECHADA) if i not in minas]
        return int(self.rng.choice(candidatas))

    def _deduzir(self, visivel):
        linhas, colunas = visivel.shape
        numeros = []  # (número, conjunto das casas fechadas vizinhas)
        for l in range(linhas):
            for c in range(colunas):
                v = visivel[l, c]
                if v <= 0:
                    continue
                fechadas = set()
                for dl in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nl, nc = l + dl, c + dc
                        if 0 <= nl < linhas and 0 <= nc < colunas and visivel[nl, nc] == FECHADA:
                            fechadas.add(nl * colunas + nc)
                if fechadas:
                    numeros.append((int(v), fechadas))

        minas, seguras = set(), set()
        mudou = True
        while mudou:
            mudou = False
            restricoes = []  # (minas que faltam, casas ainda desconhecidas)
            for v, fechadas in numeros:
                desconhecidas = fechadas - minas - seguras
                if desconhecidas:
                    restricoes.append((v - len(fechadas & minas), frozenset(desconhecidas)))

            # Regra 1
            for faltam, casas in restricoes:
                if faltam == 0:
                    seguras |= casas
                    mudou = True
                elif faltam == len(casas):
                    minas |= casas
                    mudou = True
            if mudou:
                continue

            # Regra 2
            for fa, a in restricoes:
                for fb, b in restricoes:
                    if a is b or not a < b:
                        continue
                    resto, faltam = b - a, fb - fa
                    if faltam == 0:
                        seguras |= resto
                        mudou = True
                    elif faltam == len(resto):
                        minas |= resto
                        mudou = True
        return minas, seguras


def jogar(politica, ambiente, sementes):
    """Joga uma partida por semente e devolve a taxa de vitórias."""
    vitorias = 0
    for semente in sementes:
        estado = ambiente.reset(semente)
        terminou = False
        while not terminou:
            estado, _, terminou, info = ambiente.step(politica.escolher(estado))
        vitorias += info["venceu"]
    return vitorias / len(sementes)
