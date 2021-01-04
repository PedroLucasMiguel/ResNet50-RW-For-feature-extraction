class Arfflib(object):
    def __init__(self, file_name: str, n_features: int):
        self.__file_name = file_name
        self.__n_features = n_features
        self.f = open("Net_Output/Arff_Files/" + self.__file_name + ".arff", "w")
        #Cria o header e fecha o arquivo
        self.__header()
        self.f.close()
        self.f = open("Net_Output/Arff_Files/" + self.__file_name + ".arff", "a")

    def __header(self):
        self.f.write("@RELATION '" + self.__file_name + ".arff'\n\n")
        for i in range(1, self.__n_features+1):
            self.f.write("@ATTRIBUTE x" + str(i) + " REAL\n")
        self.f.write("@ATTRIBUTE class {0,1}\n\n")
        self.f.write("@DATA\n\n")

    def append(self, array: list, output):
        for i in range(len(array)):
            self.f.write(str(array[i])+",")
        self.f.write(str(output)+"\n")

    def close(self):
        self.f.close()