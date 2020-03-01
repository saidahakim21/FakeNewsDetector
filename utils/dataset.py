from csv import DictReader
import sys
import csv

class DataSet():
    def __init__(self, name="", path="fnc-1"):
        self.path = path

        print("Reading dataset")
       # bodies = name+"_bodies.csv" # BODY ID , Body
       # stances = name+"_stances.csv" # Headline,  BODY ID, STANCE
       # articles = self.read(bodies)
      #  self.articles = dict()

        self.trainData = self.read(name+".csv")

        #make the ID an integer value
        for s in self.trainData:
            s['id'] = int(s['id'])

        for s in self.trainData:
            s['label'] = int(s['label'])


        #copy all bodies into a dictionary




    def read(self,filename):
        rows = []
        csv.field_size_limit(sys.maxsize)

        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
