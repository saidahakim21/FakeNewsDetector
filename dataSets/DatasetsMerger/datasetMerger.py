import csv
from _csv import QUOTE_ALL
from csv import DictReader
import sys
import random
from tqdm import tqdm

#reads the dataset from csv files and load it into the memory in the shape of python array
def read(path,filename):
    rows = []
    csv.field_size_limit(sys.maxsize)

    with open(path + "/" + filename, "r", encoding='utf-8') as table:
        r = DictReader(table)

        for line in r:
            rows.append(line)
    return rows

#cleans the real articles lines (if either title or author or text) is null , the line will be dumped
def cleanRealArticles(realRaw):
    cleanedList  = []
    for e in realRaw:
        if ((not e['title']) or (not e['author']) or (not e['text'])):
            continue

        e['label'] = '1'
        e['title'] = e['title'].replace('\n','\r\n')
        e['text']  = e['text'].replace('\n','\r\n')
        e['author']  = e['author'].replace('\n','\r\n')

        cleanedList.append(e)
    return cleanedList

#cleans the real articles lines (if either text or title) is null , or the line isn't in english , then it will be dumped
def cleanFakeArticles(fakeRaw):
    cleanedFake = []
    for e in fakeRaw:
        if ((not e['title']) or (not e['text']) or (not e['language']) or (e['language'] != "english")):
            continue
        else:
            e['label'] = '0'
            e['title'] = e['title'].replace('\n','\r\n')
            e['text'] = e['text'].replace('\n','\r\n')
            cleanedFake.append(e)
    return cleanedFake

if __name__ == "__main__":


    realOne = read("./","articles1.csv")
    fake    = read("./","fake.csv")

    cleanReal = cleanRealArticles(realOne)
    cleanFake = cleanFakeArticles(fake)

    perfectDataset = cleanFake + cleanReal[:len(cleanFake)]

    random.shuffle(perfectDataset)

    with open('fake_gold_real_articles.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                lineterminator='\r\n',
                                quoting= QUOTE_ALL)
        filewriter.writerow(['id','title','author','text','label'])

        for i,e in tqdm(enumerate(perfectDataset)):
            if not e['author']:
                filewriter.writerow([str(i), e['title'], '',e['text'],e['label']])
            else:
                filewriter.writerow([str(i),e['title'], e['author'], e['text'],e['label']])