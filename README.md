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

## Equipe

- Cauê Fernando Faria Abreu
- Paulo Fernando Azevedo de Carvalho

Disciplina: Inteligência Artificial
