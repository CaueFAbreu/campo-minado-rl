# Entrega 2: resultados da DQN

| Arquivo | O que é |
|---|---|
| [apresentação 2.pdf](apresentação%202.pdf) | Apresentação em 2 slides: como a DQN foi treinada (estado, ações, recompensas, γ, ε, rede e ciclo da Double DQN) e os resultados (curva de aprendizado, comparação, tempo de treino e onde ela erra) |
| [slide-treino-e-resultado.pdf](slide-treino-e-resultado.pdf) | Slide da apresentação: como a DQN foi treinada, o que fizemos além dela e o resultado |
| [agente_final_9x9.mp4](agente_final_9x9.mp4) | A DQN final jogando 6 tabuleiros 9×9 novos, sem escolher partidas (venceu 5). Cada casa fechada mostra a chance de mina que a DQN estima |
| [treino_9x9.mp4](treino_9x9.mp4) | A DQN jogando os mesmos 3 tabuleiros 9×9 em 4 momentos do treino: 0, 10 mil, 30 mil e 98 mil partidas |
| [treino_5x5.mp4](treino_5x5.mp4) | O mesmo para o 5×5: 0, 4 mil, 10 mil e 32 mil partidas |

Resultado nos 10.000 tabuleiros de teste: a DQN final vence **87,3%** no 9×9 (resolvedor lógico: 79,2%;
probabilidade exata: 90,2%) e **65,2%** no 5×5 (resolvedor lógico: 61,2%; probabilidade exata: 75,2%).

Detalhes, gráficos e como rodar estão no [README principal](../README.md). Para ver a DQN jogando ao
vivo, rode `python demo.py` na raiz do repositório.
