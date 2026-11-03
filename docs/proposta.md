# Proposta de novo tema: agente de RL para bateria residencial com energia solar


## O problema

Uma casa tem **painel solar** e uma **bateria**. A cada hora, o agente decide se a bateria
**carrega, descarrega ou fica parada**, com o objetivo de **minimizar a conta de luz**.

A decisão é difícil porque três coisas variam ao longo do dia:

| Fator | Comportamento |
|---|---|
| Preço da energia | Na **tarifa branca** o kWh é bem mais caro no horário de ponta (fim de tarde/início da noite, dias úteis) do que no resto do dia |
| Geração solar | Máxima ao meio-dia, zero à noite, incerta por causa de nuvens |
| Consumo da casa | Costuma ter pico à noite, quando não há sol e a energia está mais cara |

O ganho vem de **guardar energia barata** (sobra de sol ou horário fora de ponta) e **usá-la na ponta**.
O que torna o problema sequencial é que **carregar agora reduz o que dá para carregar depois**:
cada decisão muda as opções das próximas horas.

---

## Modelagem como MDP

| Elemento | Proposta |
|---|---|
| **Episódio** | 1 dia, com passos de 1 hora (24 decisões) |
| **Estado** | carga da bateria (%), hora do dia, dia útil ou fim de semana, geração solar atual, consumo atual; opcionalmente, previsão de geração para as próximas horas |
| **Ações** | carregar 2 kW · carregar 1 kW · parar · descarregar 1 kW · descarregar 2 kW (5 ações discretas) |
| **Transição** | carga da bateria muda de forma determinística (com perda por eficiência, ~10%); sol e consumo mudam de forma aleatória, a partir de dados reais |
| **Recompensa** | − (energia comprada × tarifa do horário) + (valor da energia injetada na rede) − (custo de desgaste da bateria) |

**Propriedade de Markov:** sol e consumo dependem do que aconteceu nas horas anteriores
(um dia nublado tende a continuar nublado). Por isso o estado inclui a geração atual e, se possível,
a previsão. É um ponto bom para discutir na apresentação.

---

## Como vamos avaliar o agente

Vamos comparar o agente com três políticas, que funcionam como piso, referência e teto:

| Política | Papel |
|---|---|
| Sem bateria | **Piso**: quanto custa não fazer nada |
| Regra simples: "carrega com sobra de sol, descarrega na ponta" | **Referência**: o que um instalador configuraria |
| **Agente de RL** | O que o trabalho mede |
| Ótimo com previsão perfeita (otimização sabendo o futuro) | **Teto**: o melhor resultado possível |

O resultado final fica fácil de comunicar: *"o agente economiza R$ X por mês a mais que a regra simples
e chega a Y% do ótimo teórico"*.

O ótimo com previsão perfeita não é uma solução real, porque ninguém sabe o sol e o consumo de amanhã.
O agente decide **sem conhecer o futuro**, e é isso que justifica usar RL.

---

## Dados (todos públicos)

| Dado | Fonte | Situação |
|---|---|---|
| Geração solar horária | NASA POWER ou PVGIS (gratuitas, com API) ou estações do INMET; localização de Uberlândia | a confirmar |
| Tarifa branca | ANEEL / CEMIG: valores de cada horário e os horários da ponta | a confirmar |
| Consumo residencial horário | **Ponto fraco**: poucos dados públicos brasileiros. Alternativas: bases estrangeiras ou perfil sintético com ruído | a definir |

---

## Riscos

1. **A economia pode ser pequena no Brasil.** Pela Lei 14.300/2022 (geração distribuída), a energia
   que o painel injeta na rede vira crédito, e a rede funciona quase como uma "bateria grátis".
   **Como lidar:** tratar o valor da energia injetada como parâmetro e testar cenários (compensação
   integral, parcial e nenhuma). Isso pode virar uma conclusão do trabalho:
   *em quais condições a bateria vale a pena?*
2. **Dados de consumo:** ver a tabela acima.
3. **"Dá para resolver sem RL?"** Só sabendo o futuro. Por isso o ótimo com previsão perfeita entra
   como teto de comparação, e não como solução.

---

## Plano de algoritmos

1. **Q-learning tabular**: carga em 11 níveis × 24 horas × sol em 3 faixas ≈ 800 estados.
   Rápido de treinar e fácil de explicar.
2. **DQN ou PPO**: estado contínuo, incluindo a previsão de geração.
3. **Extensões, se sobrar tempo**: carro elétrico, várias casas, diferentes tamanhos de bateria.

---

