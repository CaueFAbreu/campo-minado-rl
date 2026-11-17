# Campo Minado com Aprendizado por Reforço

Agente de aprendizado por reforço que aprende a jogar Campo Minado: a cada jogada, escolhe uma casa
para abrir, tentando abrir todas as casas seguras sem acertar uma mina.

## Problema

Os números do tabuleiro permitem deduzir parte das jogadas, mas nem todas. Às vezes, há mais de uma
distribuição de minas possível, e é preciso arriscar. O agente não precisa saber onde está cada mina:
ele aprende quais casas têm mais chance de ser seguras.

## Modelagem (MDP)

| | |
|---|---|
| Ambiente | tabuleiro 5×5 com 4 minas e depois 9×9 com 10 minas; as minas são sorteadas a cada partida |
| Estado | tabuleiro visível: casas fechadas e números das casas abertas |
| Ações | escolher uma casa para abrir (25 no 5×5 e 81 no 9×9) |
| Recompensa | +10 ao vencer · +1 ao abrir uma casa segura · −1 ao clicar em uma casa já aberta · −10 ao abrir uma mina |

Os valores da recompensa são um ponto de partida e serão testados.

Apresentação da entrega 1 em [slides.pdf](entrega-01-modelagem/slides.pdf).

## Avaliação

A métrica principal é a taxa de vitórias, nos mesmos tabuleiros de teste. O agente é comparado com
clicar ao acaso (piso) e com um resolvedor lógico (referência alta).

## Resultados da DQN (entrega 2)

Vitórias em 10.000 tabuleiros de teste, que o agente não viu no treino:

| Política | 5×5, 4 minas | 9×9, 10 minas |
|---|---|---|
| Ao acaso | 2,9% | 0,0% |
| Resolvedor lógico (lógica + chute ao acaso) | 61,2% | 79,2% |
| Probabilidade exata (referência forte) | 75,2% | 90,2% |
| DQN inicial | 65,2% | 67,2% |
| DQN retreinada (simetrias no treino, taxa decrescente) | 65,2%¹ | 71,6% |
| DQN com erro quadrático | 64,4% | 82,3% |
| DQN maior (64 canais, 6 camadas) | 63,4% | – |
| **DQN final** (melhor versão na validação) | **65,2%** | **87,3%** |
| Lógica + chute da DQN | 74,0% | 89,7% |

¹ No 5×5, o retreino não superou a rede original na validação, então a rede comparada é a mesma.
Todas as versões, com e sem a média nas simetrias, estão em `resultados/resumo.json`.

No 9×9, a DQN final é a rede treinada com erro quadrático, jogando com a média nas 8 simetrias.
No 5×5, é a rede do primeiro treino, sem a média.

![Curva de aprendizado](resultados/curva_aprendizado.png)

![Comparação final](resultados/comparacao_final.png)

- **Treino:** 3 sementes por tabuleiro. A primeira rodada foi interrompida: o 5×5 parou em cerca de
  70 mil partidas, já estável, e o 9×9 em cerca de 28 mil. As curvas dessa rodada foram reconstruídas
  dos registros em `resultados/logs/`. Depois, a semente 0 do 9×9 continuou em três etapas:
  1. até 100 mil partidas (`--continuar-de`);
  2. retreino de 98 mil a 140 mil (`resultados/ajuste/`), com `--aumentar` (cada lote do replay é girado
     ou espelhado ao acaso) e `--lr-final` (taxa de aprendizado caindo até 10⁻⁵ em forma de cosseno);
  3. de 134 mil a 176 mil com `--perda mse` (`resultados/mse/`). Esta etapa foi planejada até 194 mil
     e parou antes; a rede escolhida é a de 166 mil.
- **Erro quadrático em vez de Huber:** com recompensas de +1 e −10, a perda de Huber trata os erros
  grandes de forma linear e deixa quase iguais os valores de uma casa garantida e de uma arriscada. O
  erro quadrático aprende a média das recompensas e separa melhor os riscos. No 9×9, a DQN sozinha foi
  de 71,6% para 82,3%. No 5×5, não houve ganho, e γ = 0 foi pior.
- **Média nas simetrias:** o jogo não muda se o tabuleiro for girado ou espelhado. Na hora de jogar, a
  rede avalia as 8 versões do tabuleiro, e a média dos valores Q decide a jogada. No 9×9, isso rendeu de
  5 a 14 pontos sem treino a mais. No 5×5, não ajudou.
- **Rede maior:** no 5×5, 64 canais e 6 camadas não superaram a rede de 32 canais, então não a
  treinamos no 9×9.
- **Probabilidade exata** ([campo_minado/politicas.py](campo_minado/politicas.py)): enumera todas as
  distribuições de minas compatíveis com os números e abre a casa com menor chance de mina. É a
  referência forte. A DQN final alcança 87% (5×5) e 97% (9×9) do resultado dela.
- **Onde a DQN perde:** no 9×9, 69% das derrotas da DQN inicial aconteceram quando havia uma casa que a
  lógica garantia ser segura (no 5×5, 50%). Ou seja, ela errava deduções.
- **Lógica + chute da DQN:** abre as casas que a lógica garante e só usa a DQN quando é preciso
  chutar. Fica a menos de 1,5 ponto da probabilidade exata, então a DQN chuta quase tão bem quanto o
  cálculo exato.
- **Rede comparada:** em cada treino, escolhida entre as salvas pela taxa de vitórias em 2.000
  tabuleiros de validação, separados dos de teste. A versão final também é escolhida pela validação.
- **Vídeos** (em [entrega-02-resultados/](entrega-02-resultados/), com o slide da apresentação em PDF):
  a DQN inicial joga os mesmos 3 tabuleiros em 4 momentos do treino
  ([5×5](entrega-02-resultados/treino_5x5.mp4) e [9×9](entrega-02-resultados/treino_9x9.mp4)), e a DQN
  final joga 6 tabuleiros 9×9 novos ([agente_final_9x9.mp4](entrega-02-resultados/agente_final_9x9.mp4),
  gerado por `python video_final.py`).
- **Demonstração ao vivo:** `python demo.py` abre uma janela em que a DQN final joga tabuleiros novos,
  mostrando a chance de mina que ela estima para cada casa.

### Detalhes da implementação

- **Jogo** ([campo_minado/ambiente.py](campo_minado/ambiente.py)): a primeira casa aberta nunca é
  mina. Se for, a mina vai para a primeira casa livre, então a mesma semente gera o mesmo tabuleiro
  para todas as políticas.
- **Resolvedor** ([campo_minado/politicas.py](campo_minado/politicas.py)): deduz casas seguras e minas
  a partir de cada número e de pares de números. Quando não há certeza, chuta ao acaso.
- **DQN** ([campo_minado/dqn.py](campo_minado/dqn.py)): rede só de convoluções, com 5 camadas de 32
  filtros. Ela recebe o tabuleiro em one-hot (fechada, 0 a 8) e dá um valor Q por casa.
- **Treino** ([treinar.py](treinar.py)): Double DQN com replay de 200 mil jogadas, rede-alvo copiada a
  cada 1.000 passos e γ = 0,1. Jogamos 32 partidas em paralelo, e ε cai de 1 até 0,01 nos primeiros
  30% do treino.
- **Recompensas:** as da modelagem. O agente só escolhe entre casas fechadas, então o −1 por clicar
  numa casa aberta nunca acontece no treino.

### Como rodar

No PowerShell, cada comando vai numa linha só. Use `--saida` com uma pasta nova para não
sobrescrever os resultados em `resultados/`.

```
pip install -r requirements.txt
python demo.py                                   # a DQN final jogando, numa janela (--tabuleiro 5x5 para o menor)
python treinar.py --tabuleiro 5x5 --episodios 2000 --saida teste    # treino curto de teste
python treinar.py --tabuleiro 9x9 --checkpoints --saida meu_treino  # treino completo (100 mil partidas)
python avaliar.py                                # compara as políticas e gera resultados/resumo.json
python graficos.py                               # gera os dois gráficos
python video.py --tabuleiro 9x9 --momentos 0 10000 30000 100000
```

Opções do `treinar.py` usadas nos retreinos: `--continuar-de <rede.pt>`, `--aumentar`,
`--lr-final 1e-5`, `--perda mse`, `--canais 64 --camadas 6`.

## Equipe

- Cauê Fernando Faria Abreu
- Paulo Fernando Azevedo de Carvalho

Disciplina: Inteligência Artificial
