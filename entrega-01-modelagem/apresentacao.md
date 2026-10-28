# Entrega 1 — Modelagem do Problema como MDP

**Problema:** dado um time de futebol em posse de bola, determinar a sequência ótima de ações
(passes, chutes, dribles, movimentações) que maximize a probabilidade de converter a posse em gol,
considerando as posições dos 22 jogadores, o contexto do jogo e as transições probabilísticas do ambiente.

---

## Slide 1 — Ambiente e Estado Markoviano

**Ambiente**
- Campo de futebol (105 × 68 m) com 22 jogadores e a bola, simulado em tempo discreto (passos de ~0,1 s ou por evento).
- **Episódio = uma posse de bola**: começa quando o time recupera a bola e termina em
  **gol**, **perda da posse** (interceptação, desarme, bola fora, falta) ou **limite de tempo**.
- O time adversário faz parte do ambiente: seu comportamento é a fonte da incerteza nas transições.

**Estado `s_t` — o que o agente observa**

| Grupo | Informação | Dimensão |
|---|---|---|
| Bola | posição (x, y, z), velocidade (vx, vy, vz) | 6 |
| Companheiros (11) | posição (x, y) e velocidade (vx, vy) | 44 |
| Adversários (11) | posição (x, y) e velocidade (vx, vy) | 44 |
| Posse | qual jogador está com a bola | 1 |
| Contexto | placar (saldo), tempo restante, modo de jogo (bola rolando, escanteio, falta, lateral) | ~3 |

Total: ≈ 100 variáveis contínuas.

**Por que é markoviano?**
- Posições **e velocidades** de todos os jogadores e da bola descrevem a dinâmica atual:
  o próximo estado depende apenas de `s_t` e da ação `a_t`, não do histórico da jogada.
- Sem as velocidades o estado *não* seria markoviano (não daria para saber para onde cada jogador está indo).
- Simplificação: observabilidade total (como em dados de *tracking*). Intenções do adversário
  não são observáveis e entram como estocasticidade da transição `P(s' | s, a)`.

---

## Slide 2 — Ações do Agente

**Agente:** o jogador com a bola (o controle passa para quem recebe o passe).
Os companheiros sem bola seguem uma política fixa de movimentação (ocupar espaços / oferecer linha de passe).

| Ação | Descrição | Qtd. |
|---|---|---|
| Passe curto(j) | passe rasteiro para o companheiro j | 10 |
| Passe longo / cruzamento(j) | bola alta para o companheiro j | 10 |
| Chute | finalização ao gol | 1 |
| Conduzir(dir) | avançar com a bola em uma de 8 direções | 8 |
| Driblar | tentar passar pelo marcador mais próximo | 1 |
| Segurar | proteger a bola / esperar movimentação dos companheiros | 1 |

Total: **31 ações discretas**.

**Transições probabilísticas (exemplos)**
- Passe: probabilidade de sucesso cai com a distância e com adversários próximos da linha do passe.
- Chute: vira gol com probabilidade ≈ xG (depende de distância, ângulo e pressão).
- Drible: sucesso depende da distância e do número de marcadores.
- Mesmo com sucesso, a posição final da bola e dos jogadores tem ruído.

---

## Slide 3 — Recompensas

**Recompensa principal (esparsa)**

| Evento | Recompensa |
|---|---|
| Gol marcado | **+1** (fim do episódio) |
| Perda de posse / fim do tempo | **0** (fim do episódio) |
| Qualquer outro passo | 0 |

- Com fator de desconto γ = 1, o valor do estado é exatamente
  **V(s) = P(gol | s)**: maximizar o retorno esperado = maximizar a probabilidade de converter a posse em gol.
- Usar γ < 1 (ex.: 0,99) favorece jogadas mais rápidas, antes que a defesa se organize.

**Recompensa auxiliar (*reward shaping*), para acelerar o aprendizado**
- Problema: o gol é raro, então o sinal +1 quase nunca aparece no início do treino.
- Shaping baseado em potencial: `F(s, s') = γ·Φ(s') − Φ(s)`, com Φ(s) = ameaça da posição da bola
  (ex.: *expected threat* — xT — ou distância ao gol).
- Esse formato **não altera a política ótima** (Ng, Harada & Russell, 1999): só guia a exploração,
  recompensando avançar a bola para zonas perigosas e penalizando recuar.
