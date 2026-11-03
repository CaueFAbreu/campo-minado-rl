# Bateria solar com Aprendizado por Reforço

Agente de aprendizado por reforço que decide, a cada hora, quando carregar e descarregar
a bateria de uma casa com painel solar, para minimizar a conta de luz na tarifa branca.

## Problema

A energia é mais cara no horário de ponta, o sol só gera de dia e o consumo da casa tem pico
à noite. O agente aprende a guardar energia barata e usá-la quando ela é cara, sem conhecer
a geração e o consumo futuros.

## Modelagem (MDP)

| | |
|---|---|
| Estado | carga da bateria, hora, dia útil/fim de semana, geração solar, consumo |
| Ações | carregar 2 kW · carregar 1 kW · parar · descarregar 1 kW · descarregar 2 kW |
| Recompensa | −custo da energia comprada + valor da energia injetada − desgaste da bateria |

Detalhes em [docs/proposta.md](docs/proposta.md).

## Avaliação

O agente é comparado com: sem bateria (piso), regra simples de carga/descarga (referência)
e ótimo com previsão perfeita (teto).

## Situação

- [x] Definição do problema e modelagem
- [ ] Dados: geração solar, tarifa branca, consumo residencial
- [ ] Ambiente de simulação
- [ ] Políticas de comparação
- [ ] Q-learning tabular
- [ ] DQN / PPO

## Equipe

- Cauê Fernando Faria Abreu
- Paulo Fernando Azevedo de Carvalho

Disciplina: Inteligência Artificial
