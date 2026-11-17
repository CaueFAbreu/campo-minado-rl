"""Políticas de comparação: clicar ao acaso (piso) e o resolvedor lógico (referência alta).

Cada política recebe o tabuleiro visível e devolve a casa que vai abrir.
"""

from math import comb

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


class Probabilistico:
    """Calcula a probabilidade exata de mina em cada casa fechada e abre a mais segura.

    Enumera todas as distribuições de minas nas casas da fronteira (fechadas e vizinhas de um
    número) que respeitam os números visíveis. As casas fechadas longe dos números entram por
    contagem: se sobram r minas para n casas, há comb(n, r) maneiras. Cada distribuição pesa
    pelo número de tabuleiros completos que ela representa.
    É a referência forte: escolhe sempre a jogada mais segura agora (não planeja as próximas).
    """

    nome = "Probabilidade exata"

    def __init__(self, minas):
        self.minas = minas

    def probabilidades(self, visivel):
        linhas, colunas = visivel.shape
        fechadas = [i for i in range(linhas * colunas) if visivel.flat[i] == FECHADA]

        restricoes = []  # (casas fechadas vizinhas, número)
        for l in range(linhas):
            for c in range(colunas):
                v = int(visivel[l, c])
                if v < 0:
                    continue
                viz = [nl * colunas + nc
                       for nl in range(max(0, l - 1), min(linhas, l + 2))
                       for nc in range(max(0, c - 1), min(colunas, c + 2))
                       if visivel[nl, nc] == FECHADA]
                if viz:
                    restricoes.append((viz, v))

        fronteira = sorted({i for viz, _ in restricoes for i in viz})
        interior = [i for i in fechadas if i not in set(fronteira)]

        # Componentes independentes da fronteira (casas ligadas por um número em comum)
        pai = {i: i for i in fronteira}

        def raiz(i):
            while pai[i] != i:
                pai[i] = pai[pai[i]]
                i = pai[i]
            return i

        for viz, _ in restricoes:
            for i in viz[1:]:
                pai[raiz(i)] = raiz(viz[0])
        grupos = {}
        for i in fronteira:
            grupos.setdefault(raiz(i), []).append(i)

        componentes = []  # para cada um: {k minas: (n soluções, {casa: n soluções com mina})}
        for casas in grupos.values():
            conjunto = set(casas)
            regras = [(viz, v) for viz, v in restricoes if viz[0] in conjunto]
            componentes.append(self._enumerar(casas, regras))

        # Distribuição do total de minas na fronteira, sem um dos componentes
        def convolucao(lista):
            dist = {0: 1}
            for comp in lista:
                nova = {}
                for k1, n1 in dist.items():
                    for k2, (n2, _) in comp.items():
                        nova[k1 + k2] = nova.get(k1 + k2, 0) + n1 * n2
                dist = nova
            return dist

        n_int = len(interior)

        def maneiras_interior(k):
            r = self.minas - k
            return comb(n_int, r) if 0 <= r <= n_int else 0

        total_dist = convolucao(componentes)
        total = sum(n * maneiras_interior(k) for k, n in total_dist.items())
        prob = {}
        for j, comp in enumerate(componentes):
            outros = convolucao(componentes[:j] + componentes[j + 1:])
            for k, (_, por_casa) in comp.items():
                peso = sum(n * maneiras_interior(k + ko) for ko, n in outros.items())
                for casa, n in por_casa.items():
                    prob[casa] = prob.get(casa, 0) + n * peso
        for casa in fronteira:
            prob[casa] = prob.get(casa, 0) / total
        if n_int:
            media = sum(n * maneiras_interior(k) * (self.minas - k) for k, n in total_dist.items()) / total
            for casa in interior:
                prob[casa] = media / n_int
        return prob

    @staticmethod
    def _enumerar(casas, regras):
        """Backtracking: todas as atribuições mina/livre das casas que respeitam as regras."""
        indice = {c: i for i, c in enumerate(casas)}
        regras_da_casa = [[] for _ in casas]
        for r, (viz, _) in enumerate(regras):
            for c in viz:
                regras_da_casa[indice[c]].append(r)
        minas_na_regra = [0] * len(regras)
        livres_na_regra = [len(viz) for viz, _ in regras]
        alvo = [v for _, v in regras]
        atual = [0] * len(casas)
        resultado = {}

        def passo(i, k):
            if i == len(casas):
                n, por_casa = resultado.get(k, (0, {}))
                for c, m in zip(casas, atual):
                    if m:
                        por_casa[c] = por_casa.get(c, 0) + 1
                resultado[k] = (n + 1, por_casa)
                return
            for mina in (0, 1):
                ok = True
                for r in regras_da_casa[i]:
                    minas_na_regra[r] += mina
                    livres_na_regra[r] -= 1
                    if minas_na_regra[r] > alvo[r] or minas_na_regra[r] + livres_na_regra[r] < alvo[r]:
                        ok = False
                if ok:
                    atual[i] = mina
                    passo(i + 1, k + mina)
                for r in regras_da_casa[i]:
                    minas_na_regra[r] -= mina
                    livres_na_regra[r] += 1
            atual[i] = 0

        passo(0, 0)
        return resultado

    def escolher(self, visivel):
        prob = self.probabilidades(visivel)
        linhas, colunas = visivel.shape

        def fechadas_em_volta(i):
            l, c = divmod(i, colunas)
            return sum(visivel[nl, nc] == FECHADA
                       for nl in range(max(0, l - 1), min(linhas, l + 2))
                       for nc in range(max(0, c - 1), min(colunas, c + 2)))

        menor = min(prob.values())
        empatadas = [i for i, p in prob.items() if p <= menor + 1e-12]
        # Empate: a casa com menos vizinhas fechadas (cantos e bordas abrem mais vezes um 0)
        return int(min(empatadas, key=lambda i: (fechadas_em_volta(i), i)))
