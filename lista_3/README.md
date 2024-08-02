# CPS769 - Introdução à Inteligência Artificial e Aprendizagem Generativa

Este repositório contém as listas de exercícios da disciplina CPS769 - Introdução à Inteligência Artificial e Aprendizagem Generativa, do Programa de Engenharia de Sistemas e Computação (PESC) do Instituto Alberto Luiz Coimbra de Pós-Graduação e Pesquisa de Engenharia (COPPE/UFRJ).

## Questão 1

O objetivo deste trabalho é treinar o que foi mostrado em classe nas últimas aulas, isto é, como usar chatbots para chamada a outras ferramentas ou banco de dados. Especificamente, você trabalhará com as técnicas de chamada de função aprendidas em aula utilizando uma base de dados que fornece informações sobre o tempo, disponibilizada no Kaggle pelo INMET.

Nesta tarefa, você deverá implementar (e rodar) os passos necessários para interpretar e responder perguntas em linguagem natural.

1. Implemente o que for necessário para responder perguntas como:

   (a) “Qual o dia mais quente do ano de 2001”?

   (b) “Qual o dia mais quente do ano de 2021”?

   (c) “Qual a média da temperatura no ano de 2021”?

2. Tente também responder perguntas complexas como:

   (a) “Qual a média do mês de janeiro de 2010”?

   (b) “Janeiro de 2021 fez frio?”?

   (c) “Compare a média de temperatura do ano de 2001 com a de 2021”?

Base de dados: [Kaggle - Brazil Weather Information by INMET](https://www.kaggle.com/datasets/gregoryoliveira/brazil-weather-information-by-inmet)

Aprenda a trabalhar com os dados em Python. A base de dados é apresentada em CSV, onde cada linha representa um dia e uma hora. A base possui outras informações, além da temperatura. Para trabalhar de maneira prática com um arquivo CSV em Python, podemos utilizar o pandasql.

A solução encontra-se neste [Notebook Jupyter](./lista_3/lista_3.ipynb)