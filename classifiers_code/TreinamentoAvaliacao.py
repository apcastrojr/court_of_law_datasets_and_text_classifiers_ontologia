class TreinamentoAvaliacao:
  import sys
  import numpy as np

  def ler_dataset(self, arquivo_dataset):
    f = open(arquivo_dataset, "r")
    dataset_array=[]
    label_array=[]
    for x in f:
        data = []
        label= 0
        for dado in x.split(';'):
            if "#" in dado:
                data.append(int(dado.split('#')[0]))
                label=int(dado.split('#')[1].split('\n')[0])
            else:
                data.append(int(dado))
        dataset_array.append(data)
        label_array.append(label)  
    f.close()
    #print(dataset_array)
    #print(label_array)
    return dataset_array,label_array

  def tree(self, dataset_array, label_array, data_teste):
    from sklearn import tree

    clf = tree.DecisionTreeClassifier(max_depth=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def naivebayes(self, dataset_array, label_array, data_teste):
    from sklearn.naive_bayes import GaussianNB

    #Create a Gaussian Classifier
    clf = GaussianNB()
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def bnaivebayes(self, dataset_array, label_array, data_teste):
    from sklearn.naive_bayes import BernoulliNB

    #Create a Gaussian Classifier
    clf = BernoulliNB()
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def knn(self, dataset_array, label_array, data_teste):
    from sklearn.neighbors import KNeighborsClassifier

    #analise 2 resultados
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def mlp(self, dataset_array, label_array, data_teste):
    from sklearn.neural_network import MLPClassifier

    #analise 2 resultados
    clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1, max_iter=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def svm(self, dataset_array, label_array, data_teste):
    from sklearn import svm

    #analise 2 resultados
    clf = svm.SVC(max_iter=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def rf(self, dataset_array, label_array, data_teste):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def gbc(self, dataset_array, label_array, data_teste):
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(n_estimators=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def abc(self, dataset_array, label_array, data_teste):
    from sklearn.ensemble import AdaBoostClassifier

    clf = AdaBoostClassifier(n_estimators=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def gpc(self, dataset_array, label_array, data_teste):
    from sklearn.gaussian_process import GaussianProcessClassifier

    clf = GaussianProcessClassifier(max_iter_predict=300)
    clf.fit(dataset_array, label_array)
    return clf.predict(data_teste)

  def avaliar_acuracia(self, array_predicao, array_label_corretos):
    acuracia=cont=0
    for i in array_predicao:
      if i == array_label_corretos[cont]:
        acuracia+=1
      cont+=1
    return (acuracia / cont) * 100

  def avaliar_fmeasure(self, array_predicao, array_label_corretos):
    from sklearn.metrics import f1_score

    macro = f1_score(array_label_corretos,array_predicao, average='macro')
    micro = f1_score(array_label_corretos,array_predicao, average='micro')
    weighted = f1_score(array_label_corretos,array_predicao, average='weighted')

    return weighted

  def avaliar_all(self, array_predicao, array_label_corretos):
    from sklearn.metrics import precision_recall_fscore_support

    return precision_recall_fscore_support(array_label_corretos, array_predicao, average='weighted')

  def accuracy(self,array_predicao, array_label_corretos):
    from sklearn.metrics import accuracy_score

    return accuracy_score(array_label_corretos, array_predicao)

  if __name__ == '__main__':
    if len(sys.argv) != 4:
       print("Erro na passagem dos argumentos")
       print("Correto:")
       print("        python3 <nome_arquivo_dataset> <nome_arquivo_teste> <nome_metodo_predicao [mlp, svm, rf, gbc, abc, gpc, tree, nb, bnb, knn]>")
    else:
       #constantes
       arquivo_dataset  =sys.argv[1]
       arquivo_datateste=sys.argv[2]
       metodo_predicao  =sys.argv[3]
       from TreinamentoAvaliacao import TreinamentoAvaliacao

       print("***************************************")
       print("Classifier: {}".format(metodo_predicao))	

       ta = TreinamentoAvaliacao()
       dataset_array, label_array = ta.ler_dataset(arquivo_dataset)
       data_teste, label_teste = ta.ler_dataset(arquivo_datateste)

       resultado_predicao=[]
       if metodo_predicao == "mlp":
          resultado_predicao = ta.mlp(dataset_array,label_array,data_teste)
       elif metodo_predicao == "svm":
          resultado_predicao = ta.svm(dataset_array,label_array,data_teste)
       elif metodo_predicao == "rf":
          resultado_predicao = ta.rf(dataset_array,label_array,data_teste)
       elif metodo_predicao == "gbc":
          resultado_predicao = ta.gbc(dataset_array,label_array,data_teste)
       elif metodo_predicao == "abc":
          resultado_predicao = ta.abc(dataset_array,label_array,data_teste)
       elif metodo_predicao == "gpc":
          resultado_predicao = ta.gpc(dataset_array,label_array,data_teste)
       elif metodo_predicao == "tree":
          resultado_predicao = ta.tree(dataset_array,label_array,data_teste)
       elif metodo_predicao == "nb":
          resultado_predicao = ta.naivebayes(dataset_array,label_array,data_teste)
       elif metodo_predicao == "bnb":
          resultado_predicao = ta.bnaivebayes(dataset_array,label_array,data_teste)
       elif metodo_predicao == "knn":
          resultado_predicao = ta.knn(dataset_array,label_array,data_teste)



       if resultado_predicao == []:
         print("***************************************")
         print("  Erro:")
         print("       Nome método de predição errado. Passar o tipo correto: [mlp, svm, rf, gbc, abc, gpc]")
         print("***************************************")
       else:
         acuracia = ta.avaliar_acuracia(resultado_predicao,label_teste)
         print("***************************************")
         print("Accuracy: {}".format(round(acuracia,2)))
         #print(resultado_predicao)
         #print(resultado_datateste)
         print("***************************************")
         fmeasure = ta.avaliar_fmeasure(resultado_predicao,label_teste)
         print("F-measure: {}".format(round(fmeasure,2)))
         print("***************************************")
         all = ta.avaliar_all(resultado_predicao,label_teste)
         print("Precision, recall, F-measure: {}".format(all))
         print("***************************************")
         accuracy = ta.accuracy(resultado_predicao,label_teste)
         print("Accuracy: {}".format(round(accuracy,6)))
         print("Accuracy: {}".format(round(ta.accuracy(resultado_predicao,label_teste),6)))
         print("***************************************")
