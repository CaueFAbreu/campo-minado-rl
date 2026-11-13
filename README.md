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

| Tabuleiro | Ao acaso | Resolvedor lógico | DQN |
|---|---|---|---|
| 5×5, 4 minas | 2,9% | 61,2% | **65,2%** |
| 9×9, 10 minas | 0,0% | 79,2% | **50,2%** (treino parcial) |

![Curva de aprendizado](resultados/curva_aprendizado.png)

![Comparação final](resultados/comparacao_final.png)

- **Treino:** 3 sementes por tabuleiro. O treino foi interrompido antes das 100 mil partidas
  planejadas: o 5×5 parou em cerca de 70 mil, e o 9×9 em cerca de 28 mil, ainda melhorando. As
  curvas foram reconstruídas dos registros em `resultados/logs/`.
- **Rede comparada:** escolhida entre as salvas durante o treino da semente 0, pela taxa de vitórias em
  2.000 tabuleiros de validação separados dos de teste. No 5×5 foi a de 32 mil partidas; no 9×9, a de
  28 mil.
- **Vídeos:** a DQN joga os mesmos 3 tabuleiros em 4 momentos do treino:
  [5×5](resultados/treino_5x5.mp4) e [9×9](resultados/treino_9x9.mp4).

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

```bash
pip install -r requirements.txt
python treinar.py --tabuleiro 5x5 --semente 0 --checkpoints   # também 9x9 e outras sementes
python avaliar.py    # compara as políticas e gera resultados/resumo.json
python graficos.py   # gera os dois gráficos
python video.py --tabuleiro 9x9 --momentos 0 6000 14000 28000
```

## Equipe

- Cauê Fernando Faria Abreu
- Paulo Fernando Azevedo de Carvalho

Disciplina: Inteligência Artificial
