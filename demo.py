"""Demonstração ao vivo: a DQN treinada joga Campo Minado numa janela.

    python demo.py                  # 9×9 com 10 minas
    python demo.py --tabuleiro 5x5

Botões: "Próxima jogada" (passo a passo), "Jogar sozinho" (automático) e "Nova partida".
Com "Mostrar o que a DQN pensa" ligado, cada casa fechada mostra a chance de mina que a DQN
estima a partir do valor Q: quanto mais escura a casa, mais perigosa a DQN acha que ela é.
Os tabuleiros são novos (sementes que não foram usadas no treino nem no teste).
"""

import argparse
import json
import tkinter as tk
from pathlib import Path

import numpy as np
import torch

from campo_minado.ambiente import FECHADA, RECOMPENSA_MINA, RECOMPENSA_SEGURA, CampoMinado
from campo_minado.dqn import RedeQ, mascara_fechadas, melhores_acoes, valores_q
from treinar import TABULEIROS

FUNDO, TINTA, SUAVE = "#FAF8F2", "#16324F", "#566573"
ABERTA, DESTAQUE, MINA, VITORIA = "#E9EFF3", "#B84A00", "#9A3E00", "#2E7D6F"
COR_NUMERO = {1: "#1F5F8B", 2: "#2E7D4F", 3: "#9A3E00", 4: "#5B2A86", 5: "#7A1F1F", 6: "#1F6F6F", 7: TINTA, 8: SUAVE}
SEMENTE_INICIAL = 50_000_000  # longe das sementes de treino, validação e teste


def carregar_rede(tab):
    """Carrega a DQN final do tabuleiro. A arquitetura sai do próprio arquivo de pesos."""
    pesos = torch.load(Path("resultados") / tab / "dqn_final.pt")
    convs = sorted({k.split(".")[1] for k in pesos if k.endswith(".weight")}, key=int)
    canais = pesos[f"rede.{convs[0]}.weight"].shape[0]
    rede = RedeQ(canais, len(convs) - 1)
    rede.load_state_dict(pesos)
    simetria = False
    resumo = Path("resultados") / "resumo.json"
    if resumo.exists():
        simetria = json.loads(resumo.read_text(encoding="utf-8"))[tab]["DQN final"].get("simetria", False)
    return rede.eval(), simetria


def chance_de_mina(q, gamma=0.1):
    """Converte o valor Q numa estimativa da chance de mina.

    Abrir uma casa segura vale +1 (mais um pouco das jogadas seguintes); uma mina vale −10.
    Então Q ≈ (1 − p)·(1 + γ) + p·(−10), e daí sai p. É só uma estimativa para visualizar.
    """
    seguro = RECOMPENSA_SEGURA * (1 + gamma)
    return np.clip((seguro - q) / (seguro - RECOMPENSA_MINA), 0, 1)


class Demo:
    def __init__(self, raiz, tab, ms):
        self.raiz, self.tab, self.ms = raiz, tab, ms
        linhas, colunas, minas = TABULEIROS[tab]
        self.env = CampoMinado(linhas, colunas, minas)
        self.rede, self.simetria = carregar_rede(tab)
        self.lado = 64 if linhas > 5 else 96
        self.semente = SEMENTE_INICIAL
        self.partidas = self.vitorias = 0
        self.automatico = False
        self.ultima = None

        raiz.title(f"DQN jogando Campo Minado {linhas}×{colunas}")
        raiz.configure(bg=FUNDO)
        self.canvas = tk.Canvas(raiz, width=colunas * self.lado, height=linhas * self.lado, bg=FUNDO,
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=24, pady=24)

        painel = tk.Frame(raiz, bg=FUNDO)
        painel.grid(row=0, column=1, sticky="n", padx=(0, 24), pady=24)
        fonte = ("Segoe UI", 13)
        tk.Label(painel, text=f"Campo Minado {linhas}×{colunas} · {minas} minas", font=("Segoe UI", 13, "bold"),
                 fg=DESTAQUE, bg=FUNDO).pack(anchor="w")
        tk.Label(painel, text="DQN treinada" + (" (média nas 8 simetrias)" if self.simetria else ""),
                 font=("Segoe UI", 20, "bold"), fg=TINTA, bg=FUNDO).pack(anchor="w", pady=(4, 12))
        self.placar = tk.Label(painel, font=("Segoe UI", 15), fg=TINTA, bg=FUNDO, justify="left")
        self.placar.pack(anchor="w")
        self.status = tk.Label(painel, font=("Segoe UI", 22, "bold"), fg=TINTA, bg=FUNDO)
        self.status.pack(anchor="w", pady=(12, 16))

        botoes = tk.Frame(painel, bg=FUNDO)
        botoes.pack(anchor="w")
        estilo = dict(font=fonte, bg=TINTA, fg=FUNDO, activebackground=DESTAQUE, activeforeground=FUNDO,
                      relief="flat", padx=12, pady=6, cursor="hand2")
        tk.Button(botoes, text="Próxima jogada", command=self.jogada, **estilo).pack(fill="x", pady=3)
        self.botao_auto = tk.Button(botoes, text="Jogar sozinho", command=self.alternar, **estilo)
        self.botao_auto.pack(fill="x", pady=3)
        tk.Button(botoes, text="Nova partida", command=self.nova_partida, **estilo).pack(fill="x", pady=3)

        self.mostrar = tk.BooleanVar(value=True)
        tk.Checkbutton(painel, text="Mostrar o que a DQN pensa", variable=self.mostrar, command=self.desenhar,
                       font=fonte, fg=TINTA, bg=FUNDO, activebackground=FUNDO, selectcolor=FUNDO
                       ).pack(anchor="w", pady=(16, 4))
        tk.Label(painel, text="Número na casa fechada: chance de mina\nque a DQN estima. Mais escura = mais\n"
                              "perigosa. Contorno laranja: última jogada.",
                 font=("Segoe UI", 11), fg=SUAVE, bg=FUNDO, justify="left").pack(anchor="w")

        tk.Label(painel, text="Velocidade", font=fonte, fg=TINTA, bg=FUNDO).pack(anchor="w", pady=(16, 0))
        self.velocidade = tk.Scale(painel, from_=50, to=1500, orient="horizontal", length=220, showvalue=False,
                                   bg=FUNDO, fg=TINTA, highlightthickness=0, troughcolor=ABERTA)
        self.velocidade.set(1550 - ms)
        self.velocidade.pack(anchor="w")
        tk.Label(painel, text="devagar ← → rápido", font=("Segoe UI", 10), fg=SUAVE, bg=FUNDO).pack(anchor="w")

        raiz.bind("<space>", lambda e: self.jogada())
        raiz.bind("<Return>", lambda e: self.alternar())
        self.nova_partida()

    @torch.no_grad()
    def q(self):
        return valores_q(self.rede, self.env.visivel[None], self.simetria)[0].numpy()

    def nova_partida(self):
        self.estado = self.env.reset(self.semente)
        self.semente += 1
        self.terminou, self.venceu, self.ultima = False, False, None
        self.status.config(text="")
        self.desenhar()

    def jogada(self):
        if self.terminou:
            self.nova_partida()
            return
        est = self.estado[None]
        acao = int(melhores_acoes(torch.as_tensor(self.q())[None], mascara_fechadas(est))[0])
        self.estado, _, self.terminou, info = self.env.step(acao)
        self.ultima = divmod(acao, self.env.colunas)
        if self.terminou:
            self.venceu = info["venceu"]
            self.partidas += 1
            self.vitorias += self.venceu
            self.status.config(text="Venceu!" if self.venceu else "Abriu uma mina",
                               fg=VITORIA if self.venceu else MINA)
        self.desenhar()

    def alternar(self):
        self.automatico = not self.automatico
        self.botao_auto.config(text="Pausar" if self.automatico else "Jogar sozinho")
        if self.automatico:
            self.passo_automatico()

    def passo_automatico(self):
        if not self.automatico:
            return
        espera = 1550 - self.velocidade.get()
        fim_antes = self.terminou
        self.jogada()
        # Segura o tabuleiro final um pouco mais, para dar tempo de ver o resultado
        self.raiz.after(espera * 3 if self.terminou and not fim_antes else espera, self.passo_automatico)

    def desenhar(self):
        c, lado, env = self.canvas, self.lado, self.env
        c.delete("all")
        chances = chance_de_mina(self.q()) if self.mostrar.get() and not self.terminou else None
        for l in range(env.linhas):
            for col in range(env.colunas):
                x, y = col * lado, l * lado
                v = int(env.visivel[l, col])
                mina = self.terminou and not self.venceu and env.minas[l, col]
                if v != FECHADA:
                    cor = ABERTA
                elif chances is not None:
                    cor = _misturar("#9FB4C8", TINTA, chances[l * env.colunas + col])
                else:
                    cor = TINTA
                if mina and (l, col) == self.ultima:
                    cor = MINA
                c.create_rectangle(x + 2, y + 2, x + lado - 2, y + lado - 2, fill=cor, outline="")
                if mina:
                    r = lado * 0.18
                    c.create_oval(x + lado / 2 - r, y + lado / 2 - r, x + lado / 2 + r, y + lado / 2 + r,
                                  fill=FUNDO if (l, col) == self.ultima else "#E07B39", outline="")
                elif v > 0:
                    c.create_text(x + lado / 2, y + lado / 2, text=str(v), fill=COR_NUMERO[v],
                                  font=("Segoe UI", int(lado * 0.38), "bold"))
                elif v == FECHADA and chances is not None:
                    p = chances[l * env.colunas + col]
                    c.create_text(x + lado / 2, y + lado / 2, text=f"{p:.0%}",
                                  fill=FUNDO if p > 0.25 else TINTA, font=("Segoe UI", int(lado * 0.2)))
        if self.ultima is not None:
            l, col = self.ultima
            c.create_rectangle(col * lado + 2, l * lado + 2, (col + 1) * lado - 2, (l + 1) * lado - 2,
                               outline=DESTAQUE, width=4)
        taxa = f"{self.vitorias / self.partidas:.0%}".replace(".", ",") if self.partidas else "–"
        self.placar.config(text=f"Partidas: {self.partidas}\nVitórias: {self.vitorias}\nTaxa de vitórias: {taxa}")


def _misturar(cor_a, cor_b, t):
    a = [int(cor_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(cor_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * float(t)):02x}" for x, y in zip(a, b))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tabuleiro", choices=TABULEIROS, default="9x9")
    p.add_argument("--ms", type=int, default=600, help="tempo entre jogadas no modo automático")
    args = p.parse_args()
    raiz = tk.Tk()
    Demo(raiz, args.tabuleiro, args.ms)
    raiz.mainloop()
