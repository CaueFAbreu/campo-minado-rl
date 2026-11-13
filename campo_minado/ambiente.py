"""O jogo de Campo Minado usado como ambiente de aprendizado por reforço.

O tabuleiro visível é uma matriz de inteiros:
    -1      casa fechada
    0 a 8   casa aberta, com o número de minas vizinhas

As ações são os índices das casas, linha a linha: a casa (linha, coluna) é a ação
linha * colunas + coluna.
"""

from collections import deque

import numpy as np

FECHADA = -1

# Recompensas da modelagem (entrega 1)
RECOMPENSA_VITORIA = 10.0
RECOMPENSA_SEGURA = 1.0
RECOMPENSA_REPETIDA = -1.0
RECOMPENSA_MINA = -10.0


class CampoMinado:
    def __init__(self, linhas=5, colunas=5, minas=4, limite_jogadas=None):
        self.linhas = linhas
        self.colunas = colunas
        self.n_minas = minas
        self.n_casas = linhas * colunas
        # Sem limite informado, o agente pode jogar uma vez por casa.
        self.limite_jogadas = limite_jogadas or self.n_casas
        self.rng = np.random.default_rng()

    def reset(self, semente=None):
        """Começa uma partida nova. Com a mesma semente, as minas são as mesmas."""
        if semente is not None:
            self.rng = np.random.default_rng(semente)
        posicoes = self.rng.choice(self.n_casas, self.n_minas, replace=False)
        self.minas = np.zeros((self.linhas, self.colunas), dtype=bool)
        self.minas.flat[posicoes] = True
        self._contar_vizinhas()
        self.visivel = np.full((self.linhas, self.colunas), FECHADA, dtype=np.int8)
        self.fechadas_seguras = self.n_casas - self.n_minas
        self.jogadas = 0
        self.primeira_jogada = True
        return self.visivel.copy()

    def _contar_vizinhas(self):
        """Calcula, para cada casa, quantas minas há nas até 8 casas vizinhas."""
        borda = np.pad(self.minas, 1).astype(np.int8)
        soma = np.zeros((self.linhas, self.colunas), dtype=np.int8)
        for dl in (0, 1, 2):
            for dc in (0, 1, 2):
                if dl == 1 and dc == 1:
                    continue
                soma += borda[dl:dl + self.linhas, dc:dc + self.colunas]
        self.vizinhas = soma

    def _vizinhos(self, l, c):
        for dl in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nl, nc = l + dl, c + dc
                if (dl or dc) and 0 <= nl < self.linhas and 0 <= nc < self.colunas:
                    yield nl, nc

    def _abrir(self, l, c):
        """Abre a casa e, se ela for 0, abre as vizinhas em cadeia (busca em largura)."""
        fila = deque([(l, c)])
        self.visivel[l, c] = self.vizinhas[l, c]
        self.fechadas_seguras -= 1
        while fila:
            l, c = fila.popleft()
            if self.vizinhas[l, c] != 0:
                continue
            for nl, nc in self._vizinhos(l, c):
                if self.visivel[nl, nc] == FECHADA:
                    self.visivel[nl, nc] = self.vizinhas[nl, nc]
                    self.fechadas_seguras -= 1
                    fila.append((nl, nc))

    def step(self, acao):
        """Abre a casa `acao`. Devolve (estado, recompensa, terminou, info)."""
        l, c = divmod(int(acao), self.colunas)
        self.jogadas += 1

        # A primeira casa aberta é sempre segura: se houver uma mina nela, a mina vai
        # para a primeira casa livre, linha a linha (como no Campo Minado do Windows).
        # Isso não usa sorteio, então a mesma semente gera o mesmo jogo para todos.
        if self.primeira_jogada:
            self.primeira_jogada = False
            if self.minas[l, c]:
                livre = np.flatnonzero(~self.minas.ravel())
                livre = livre[livre != acao][0]
                self.minas[l, c] = False
                self.minas.flat[livre] = True
                self._contar_vizinhas()

        if self.visivel[l, c] != FECHADA:
            recompensa, terminou, venceu = RECOMPENSA_REPETIDA, False, False
        elif self.minas[l, c]:
            recompensa, terminou, venceu = RECOMPENSA_MINA, True, False
        else:
            self._abrir(l, c)
            venceu = self.fechadas_seguras == 0
            terminou = venceu
            recompensa = RECOMPENSA_VITORIA if venceu else RECOMPENSA_SEGURA

        if not terminou and self.jogadas >= self.limite_jogadas:
            terminou = True
        return self.visivel.copy(), recompensa, terminou, {"venceu": venceu}

    def casas_fechadas(self):
        return np.flatnonzero(self.visivel.ravel() == FECHADA)

    def mostrar(self):
        simbolos = {FECHADA: "#", 0: "."}
        for linha in self.visivel:
            print(" ".join(simbolos.get(int(v), str(int(v))) for v in linha))
