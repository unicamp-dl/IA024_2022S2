# Verificação de factualidade

**Autor:** João Carlos Cerqueira (jc.cerqueira13@gmail.com)
**Data:** 01 de dezembro de 2022


## Introdução

O objetivo do projeto é  implementar um sistema capaz de automatizar a verificação  da veracidade de afirmações. Mais especificamente, o sistema será composto de múltiplos modelos de aprendizado de máquina, onde cada um deles será responsável por uma sub-tarefa dentro do fluxo de verificação de veracidade.


## Metodologia

A proposta de solução do presente trabalho é reproduzir o sistema proposto em Pradeep et al. [1]. Esse sistema, denominado pelos autores como VerT5erini, é composto por 3 modelos de NLP T5 [7], onde cada um deles é responsável por um dos estágios do fluxo de verificação de veracidade.Os modelos T5 são uma família de modelos de NLP do tipo text-to-text, que implementa a arquitetura Transformer completa, isto é, com encoder e decoder, e suas versões pré-treinadas foram disponibilizadas pelos autores em diversos tamanhos.

O fluxo de verificação de veracidade pode ser definido de diversas formas. No presente trabalho, decidimos separá-lo em três etapas distintas: (1) Ranking de documentos relevantes, (2) Seleção de sentenças importantes e (3) Classificação.

Mais detalhes se encontram no documento `Projeto final - Joao Cerqueira - Verificação de Factualidade.pdf`.


## Referências

* [1] Ronak Pradeep, Xueguang Ma, Rodrigo Nogueira, and Jimmy Lin. Scientific Claim Verification with VERT5ERINI. arXiv:2010.11930, October 2020. arXiv: 2010.11930. http://arxiv.org/abs/2010.11930
* [2] Homepage about the dataset SciFact, provided by Allen Institute for AI: https://leaderboard.allenai.org/scifact/submissions/get-started
* [3] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or Fiction: Verifying Scientific Claims. arXiv:2004.14974, October 2020, arXiv: 2004.14974. http://arxiv.org/abs/2004.14974
* [4] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang, MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. https://microsoft.github.io/msmarco/
* [5]  Wikipedia: Okapi BM25 (https://en.wikipedia.org/wiki/Okapi_BM25)
* [6] SciFact scoring function implementation details (with examples) https://github.com/allenai/scifact/blob/master/doc/evaluation.md
* [7] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv:1910.10683.
* [8] Pyserini, a Python toolkit for reproducible information retrieval research with sparse and dense representations. https://github.com/castorini/pyserini
* [9] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. In Findings of EMNLP.
* [10] Hugging Face, MonoT5, a T5-base reranker fine-tuned on the MS MARCO passage dataset for 100k steps (or 10 epochs). https://huggingface.co/castorini/monot5-base-msmarco.
* [11] MS MARCO, a collection of datasets focused on deep learning in search. https://microsoft.github.io/msmarco
* [12] PaLM: Scaling Language Modeling with Pathways https://arxiv.org/abs/2204.02311
