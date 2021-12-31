# court_of_law_datasets_and_text_classifiers
Analyze text classifiers for court of law text documents.

The construction of the weighting of the terms directly affects the accuracy of the text classification models. Well-known resource weighting schemes are widely researched and applied, such as term frequency (TF), inverse document frequency term (TFIDF) and Okapi BM25 (BM25). However, its efficiency depends on the area of knowledge of the documents in which they are applied, since results of terms with equal or close weights, in different categories of documents, will make it difficult to predict the text classification models, given the non-linearity of the documents terms. As this research is applied in 4 categories of documents in the area of knowledge of law, having as a characteristic the non-linearity of terms in the different categories of documents, this work proposes to apply learning by similarity to technical terms, combining with different coefficients, to smooth out non-linearity or to make terms linear for different categories of documents. Similarity learning techniques are compared in the generation of resources: cosine, jaccard, data and simple correspondence. Two feature vectors are generated, binary and frequency, to carry out the supervised training of five text classification technologies: decision trees, Naive Bayes, k-NN, MLP neural network and SVM. The precision metric is used for evaluation, with 240 results from the combination of these techniques being generated. The most accurate combination is used to implement the API for the Department of Justice. The contributions of this work are centered on the insertion of learning techniques by similarity in the construction of weights, together with TF, TFIDF and BM25, and evaluation of the accuracy in the combination of different methodologies in the construction of weights, training models and classification.

This repository provides the Python code used to classify the documents. It makes use of the Scikit-learn framework.

Training datasets and test datasets are also available. The training datasets contain more than 1700 court of law text documents. The test data sets have 200 court of law text documents, distributed across 50 documents per label.

Two categories of court of law documents are used: contract (label=1); ownership, family, and retirement (laber=2).

##Programas
-Para gerar dataset: gerar_dataset_regras_especialista.rb  (cron: gerar_dataset_ontologia)
-para gerar testes: gerar_baseteste_sentencas_tfidf_ontologia.rb (cron: gerar_baseteste_ontologia)

##Termos combinados métodologia do trabalho
-Arquivo com os termos combinados: termos_combinados_jaccard_0.5_tfidf
-Arquivo gerado após processamento: dataset_termos_combinados_jaccard_sometodoproposto_0.5_1200_tfidf_{binario ou frequencia}.txt
-Arquivo gerado após processamento: base_teste_termos_combinados_jaccard_sometodoproposto_0.5_200_tfidf

##Termos combinados híbrido
-Arquivo com os termos combinados: termos_combinados_jaccard_0.5_tfidf_e_especialista.txt
-Arquivo gerado após processamento: dataset_termos_combinados_jaccard_hibrido_0.5_1200_tfidf_{binario ou frequencia}.txt
-Arquivo gerado após processamento: base_teste_termos_combinados_jaccard_hibrido_0.5_200_tfidf
