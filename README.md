# Previsão do Preço das Ações do Google usando GRU RNN

*To access the english version [click here](README.eng.md)*

Este projeto utiliza uma Rede Neural Recorrente (RNN) baseada em Gated Recurrent Unit (GRU) para analisar e prever os preços das ações do Google. O modelo foi projetado para capturar as dependências temporais presentes em dados financeiros históricos, permitindo revelar tendências de mercado e padrões de volatilidade ao longo do tempo.

## Índice

1. [Contexto](#contexto)
2. [O que é uma RNN com GRU?](#o-que-é-uma-rnn-com-gru)
3. [Ferramentas Utilizadas](#ferramentas-utilizadas)
4. [O Processo](#o-processo)
5. [Visão Geral dos Dados e Análise](#visão-geral-dos-dados-e-análise)
6. [A Análise](#a-análise)
7. [O Que Aprendi](#o-que-aprendi)
8. [Habilidades Praticadas](#habilidades-praticadas)
9. [Conclusão](#conclusão)
10. [Contato](#contato)
11. [Contribuições](#contribuições)
12. [Estrutura do Repositório](#estrutura-do-repositório)

## Contexto

Este projeto foi criado como parte de uma análise exploratória de previsão de séries temporais financeiras. Ao aplicar um modelo GRU RNN aos dados históricos de ações do Google, o objetivo é prever tendências futuras do mercado e fornecer insights sobre a dinâmica do mercado entre 2006 e 2018.

## O que é uma RNN com GRU?

### Redes Neurais Recorrentes (RNNs)
Redes Neurais Recorrentes (RNNs) são uma classe de redes neurais especialmente adequadas para processar dados sequenciais. Diferentemente das redes neurais feedforward tradicionais, as RNNs possuem ciclos em sua arquitetura, permitindo que mantenham um estado oculto que persiste ao longo dos passos do tempo. Essa "memória" possibilita que as RNNs compreendam o contexto e as dinâmicas temporais, essenciais para tarefas como previsão de séries temporais, processamento de linguagem natural e reconhecimento de fala.

No entanto, as RNNs tradicionais frequentemente enfrentam dificuldades para aprender dependências de longo prazo devido a problemas como o desaparecimento do gradiente. Conforme a informação é propagada de volta por diversas camadas durante o treinamento, os gradientes podem se dissipar, dificultando o aprendizado a partir de entradas anteriores em uma sequência.

### Gated Recurrent Units (GRUs)
GRUs são uma variante avançada das RNNs, projetada para superar algumas das limitações das RNNs convencionais. Elas incorporam mecanismos de *gate* que regulam o fluxo de informação através da rede:

- **Porta de Atualização (Update Gate):**  
  Controla quanto do estado oculto anterior deve ser levado para o estado atual. Essa porta permite que o modelo retenha informações importantes de longo prazo, descartando detalhes menos relevantes.

- **Porta de Reset (Reset Gate):**  
  Determina como combinar a nova entrada com o estado oculto anterior. Ao "resetar" seletivamente o estado oculto, essa porta possibilita que a rede esqueça informações passadas irrelevantes e se adapte melhor às novas entradas.

A arquitetura GRU simplifica o design em comparação com outros modelos com *gate* (como os LSTMs), mantendo a capacidade de capturar efetivamente dependências de curto e longo prazo. Esse equilíbrio torna as GRUs particularmente eficazes para tarefas de previsão, como a previsão de preços de ações, onde tanto as tendências recentes quanto o contexto histórico são relevantes.

## Ferramentas Utilizadas

- **Linguagem de Programação:** Python 3.12.9  
- **Frameworks de Deep Learning:** TensorFlow/Keras (ou PyTorch)  
- **Manipulação de Dados:** Pandas, NumPy  
- **Visualização de Dados:** Matplotlib, Seaborn  
- **Pré-processamento & Métricas:** Scikit-learn  
- **IDE:** Visual Studio Code (VSCode)

## O Processo

1. **Pré-processamento dos Dados:**  
   - Limpeza e normalização dos dados  
   - Divisão do conjunto de dados em subconjuntos de treinamento e teste

2. **Construção do Modelo:**  
   - Projeto da arquitetura RNN baseada em GRU  
   - Compilação do modelo com otimizador e função de perda apropriados

3. **Treinamento:**  
   - Alimentação dos dados históricos de ações no modelo  
   - Ajuste dos hiperparâmetros para otimizar o desempenho

4. **Avaliação:**  
   - Avaliação do modelo utilizando métricas como o Erro Quadrático Médio (MSE)  
   - Visualização dos preços previstos versus os preços reais

5. **Predição:**  
   - Geração de previsões para analisar tendências futuras do mercado

## Visão Geral dos Dados e Análise

O conjunto de dados utilizado neste projeto contém dados históricos de preços das ações do Google (GOOGL), cobrindo o período de 1º de janeiro de 2006 a 1º de janeiro de 2018. A seguir, uma visão geral do conjunto de dados e os principais insights obtidos na Análise Exploratória dos Dados (EDA):

### Visão Geral do Conjunto de Dados

- **Data:** Data de negociação  
- **Open:** Preço de abertura da ação no dia  
- **High:** Maior preço registrado durante o dia  
- **Low:** Menor preço registrado durante o dia  
- **Close:** Preço de fechamento da ação  
- **Volume:** Número de ações negociadas durante o dia  
- **Name:** Símbolo da ação (GOOGL)

### Análise Exploratória dos Dados (EDA)

- **Sem Valores Faltantes:**  
  Todas as colunas estão completamente preenchidas, eliminando a necessidade de tratamento de dados faltantes.

- **Total de Dias de Negociação:**  
  O conjunto de dados contém 3.019 registros diários.

- **Intervalo de Preços das Ações:**  
  - *Preço de Fechamento Mínimo:* $128.85  
  - *Preço de Fechamento Máximo:* $1,085.09  
  - *Preço de Fechamento Médio:* $428.04

- **Volatilidade das Ações:**  
  O preço de fechamento apresenta um desvio padrão de $236.34, indicando flutuações significativas.

- **Volume de Negociação:**  
  Os volumes diários de negociação variam consideravelmente, com uma média de 3.55 milhões de ações e um pico de 41.18 milhões.

- **Análise de Tendência (2006-2018):**  
  A tendência geral mostra um crescimento consistente a longo prazo no preço das ações do Google, intercalado por períodos de alta volatilidade. Notavelmente, há um aumento acentuado de 2015 a 2018, refletindo forte confiança do mercado.

## A Análise

Esta seção descreve a análise experimental onde foram treinadas e comparadas quatro configurações diferentes de modelos GRU. Os modelos diferem pelo número de camadas GRU e pela quantidade de unidades em cada camada:

- **Única Camada GRU com 15 Unidades (single_15):**  
  Este modelo possui uma única camada GRU contendo 15 unidades. Obteve o desempenho mais próximo dos valores reais.

- **Única Camada GRU com 50 Unidades (single_50):**  
  Um modelo com uma única camada GRU de 50 unidades. Seu desempenho foi muito semelhante ao do modelo single_15, indicando que o aumento no número de unidades não alterou significativamente o resultado neste caso.

- **Duas Camadas GRU com 15 Unidades Cada (double_15):**  
  Esta configuração empilha duas camadas GRU com 15 unidades cada. Embora as previsões tenham sido um pouco menos precisas em comparação com os modelos de camada única, as linhas de previsão foram mais estáveis, sugerindo uma melhor suavização da série temporal.

- **Duas Camadas GRU com 50 Unidades Cada (double_50):**  
  Um modelo mais profundo, com duas camadas GRU, cada uma com 50 unidades. Este modelo apresentou o pior desempenho, com previsões consideravelmente afastadas dos valores reais.

### Curvas de Perda e Previsões dos Modelos

A seguir, seguem os espaços reservados para as imagens das curvas de perda de cada configuração de modelo (na seguinte ordem):

1. **Curva de Perda - Única Camada GRU com 15 Unidades (single_15):**  
   ![Curva de Perda - Single_15](<assets/single_layer_15_gru_loss.png>)

2. **Curva de Perda - Única Camada GRU com 50 Unidades (single_50):**  
   ![Curva de Perda - Single_50](<assets/single_layer_50_gru_loss.png>)

3. **Curva de Perda - Duas Camadas GRU com 15 Unidades (double_15):**  
   ![Curva de Perda - Double_15](<assets/double_layer_15_gru_loss.png>)

4. **Curva de Perda - Duas Camadas GRU com 50 Unidades (double_50):**  
   ![Curva de Perda - Double_50](<assets/double_layer_50_gru_loss.png>)

Por fim, a imagem a seguir mostra um gráfico comparativo com os valores reais e as previsões de cada modelo:

- **Valores Reais vs. Previsões:**  
  ![Comparação de Previsões](<assets/predictions-comparrison.png>)

### Resumo dos Resultados

- **Melhor Desempenho:**  
  O modelo **single_15** apresentou resultados mais próximos dos valores reais, com o modelo **single_50** também demonstrando desempenho muito semelhante.

- **Estabilidade vs. Precisão:**  
  O modelo **double_15**, apesar de ser um pouco menos preciso, produziu linhas de previsão mais estáveis.

- **Desempenho Inferior:**  
  O modelo **double_50** apresentou o pior desempenho, com previsões significativamente distantes em relação aos outros modelos.

- **Insight Geral:**  
  Apesar da experimentação com diversas arquiteturas, nenhum dos modelos conseguiu capturar com precisão os valores reais das ações, indicando que um ajuste mais fino ou o uso de arquiteturas mais complexas pode ser necessário para melhorar a acurácia das previsões.

## O Que Aprendi

- A eficácia das RNNs com GRU na modelagem de dados complexos de séries temporais.  
- O papel crítico do pré-processamento detalhado dos dados para aprimorar o desempenho do modelo.  
- Insights sobre a dinâmica das tendências e volatilidade do mercado de ações.  
- A importância do ajuste e validação dos modelos de deep learning em conjuntos de dados financeiros.

## Habilidades Praticadas

- Design de Deep Learning e Redes Neurais  
- Análise e Previsão de Séries Temporais  
- Visualização de Dados e Análise Estatística  
- Programação em Python e Implantação de Modelos  
- Pesquisa e Análise Crítica de Dados Financeiros

## Conclusão

Este projeto demonstra o potencial e os desafios de utilizar RNNs com GRU para a previsão dos preços das ações com base em dados históricos do Google. Embora os modelos de camada única (particularmente **single_15** e **single_50**) tenham obtido previsões mais próximas dos valores reais, nenhuma das configurações conseguiu capturar perfeitamente a dinâmica do mercado. Esses insights ressaltam a necessidade de explorar mais a fundo as arquiteturas de modelos e o ajuste dos hiperparâmetros para melhorar o desempenho.

## Contato

Se você tiver alguma dúvida ou feedback, sinta-se à vontade para entrar em contato:  
[GitHub](https://github.com/faduzin) | [LinkedIn](https://www.linkedin.com/in/ericfadul/) | [eric.fadul@gmail.com](mailto:eric.fadul@gmail.com)

## Contribuições

- Tayenne Euqueres
- William de Oliveira Silva

## Estrutura do Repositório

```bash
├── assets/           # Recursos suplementares (ex.: imagens, gráficos)
├── data/             # Arquivos de dados brutos e processados
├── notebooks/        # Notebooks Jupyter
├── src/              # Código fonte para treinamento e avaliação do modelo
├── .gitignore        # Arquivo para ignorar arquivos desnecessários
├── LICENSE           # Informações sobre a licença (MIT)
├── README.md         # Documentação e visão geral do projeto
└── requirements.txt  # Lista de dependências do Python
```
Esta estrutura mantém o projeto organizado e facilita a navegação pelo código, dados e recursos de análise.
