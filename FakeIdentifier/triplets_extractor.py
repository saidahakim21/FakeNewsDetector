

import requests
from SPARQLWrapper import SPARQLWrapper, JSON





def replaceWithDbpediaURI(res):
    fredDomain = res.find('#')
    if(fredDomain != -1):
        queryName = res.split('#',1)
        response = requests.get('http://lookup.dbpedia.org/api/search/KeywordSearch?QueryString='+queryName[1])
        respString = str(response.content)
        uri = respString.split('</URI>',1)[0]
        uri = uri.split('<URI>',1)[1]
        return uri
    return res


def cleanRelation(rel):
    cleanRelation = rel.split('#',1)[1]
    return cleanRelation

def get_dbpedia_relation(uri1,uri2):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    sparql.setQuery("SELECT distinct ?p WHERE { <"+uri1+ "> ?p <"+uri2+"> }")

    return sparql.query().convert()


def analyse_text(text):
    triplets = []

    headers = {
        'accept': 'application/rdf+json',
        'Authorization': 'Bearer  0109a783-0410-3dbe-90f4-ac5a73f24b9a',
    }

    print("text = "+text)
    sameAsProp = 'http://www.w3.org/2002/07/owl#sameAs'

    response = requests.get('http://wit.istc.cnr.it/stlab-tools/fred?textannotation=earmark&semantic-subgraph=true&text='+text, headers=headers)

    defienedProperties = ['http://www.ontologydesignpatterns.org/ont/fred/domain.owl#capitalOf','http://www.ontologydesignpatterns.org/ont/fred/domain.owl#founderOf']

    nodes = response.json()
    entry_nodes = list(nodes.keys())

    stack = dict()
    for n in entry_nodes :
        stack[n] = 0



    fred_to_dbpedia_property = dict()
    fred_to_dbpedia_property['capitalOf'] = ['dbpedia:ontology/capital']
    fred_to_dbpedia_property['founderOf'] = ['http://dbpedia.org/ontology/foundedBy','http://dbpedia.org/property/founders']

    for n in entry_nodes :
        if stack[n] == 0 :
            stack[n] = 1
            properties_list = nodes.get(n)
            intressted_properties = list(filter(lambda p: p in defienedProperties, properties_list))
            for p in intressted_properties:
                value = properties_list.get(p) # since graph is oriented
                triplets.append((n,p,str(value[0]['value'])))
            if sameAsProp in properties_list :
                equivalent_node = properties_list.get(sameAsProp)[0]['value']
                #refresh triplets
                for i, t in enumerate(triplets):
                    if n == t[0]:
                        triplets[i] = (equivalent_node, t[1], t[2])

    for (res,rel,val) in triplets:
        uri1 = replaceWithDbpediaURI(res)
        claimed_relation = cleanRelation(rel)
        uri2 = replaceWithDbpediaURI(val)


        dbpedia_result = get_dbpedia_relation(uri1,uri2)
        dbpedia_inversed_result = get_dbpedia_relation(uri2,uri1)



        if( (len(dbpedia_result['results']['bindings']) == 0 ) & (len(dbpedia_inversed_result['results']['bindings']) == 0)) :
            return {'uri_1': str(uri1), 'uri_2' : str(uri2), 'rel':claimed_relation}
        else:
            dbpedia_relation = dbpedia_result['results']['bindings'][1]['p']['value']
            dbpedia_relation2 = dbpedia_inversed_result['results']['bindings'][1]['p']['value']

            if ( (dbpedia_relation2 in fred_to_dbpedia_property[claimed_relation] ) | (dbpedia_relation in fred_to_dbpedia_property[claimed_relation])):
                return {'result': 'True'}
            else:
                return {'uri_1': str(uri1), 'uri_2': str(uri2), 'rel': claimed_relation}