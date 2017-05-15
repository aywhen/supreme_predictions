"""
BEFORE you run this, download bulk data from CourtListener.com.
Instructions here:
https://github.com/hinthornw/supreme_predictions/wiki/Text-Data

Then, make sure that you are running mongod (default settings).
"""

import requests
from BeautifulSoup import BeautifulSoup
import pandas as pd
from pymongo import MongoClient
import os, json
import csv

path = '../data/'
client = MongoClient()
db = client['scotus']
opinions = db['opinions']
people = db['people']
clusters = db['clusters']


# Old stuff using API
def check_authors():
    # helper function to check that we match the authors properly
    # not sure if this is good rn.

    df = pd.read_csv('../data/SCDB_2016_01_justiceCentered_Citation.csv')
    justice_names = df.justiceName.unique()
    justices = dict.fromkeys(justice_names)

    print 'justices', justices
    for person in people.find({'cl_id': {'$regex': 'fjc.*'}}):
        justice_name = get_justice_name(person)
        justices[justice_name] = person['resource_uri']
    return justices

def get_justice_name(author):
    """
    Gets the corresponding justiceName (for SCDB dataset) for every author in
    the CourtListener.com dataset.
    """
    first = author['name_first']
    middle = author['name_middle']
    last = author['name_last']
    firsti = first[0].upper() if len(first) > 0 else ''
    middlei = middle[0].upper() if len(middle) > 0 else ''
    justice_name = firsti + middlei + last

    # JHarlan2
    if justice_name == 'JHarlan':
        justice_name += author['name_suffix']

    return justice_name

def get_author(author_url):
    # deprecated
    r = requests.get(author_url)
    data = r.json()
    justice_name = get_justice_name(data)

def get_opinions(caseId):
    # by API. DEPRECATED
    # https://github.com/hinthornw/supreme_predictions/wiki/Text-Data
    opinions = []

    r = requests.get('https://www.courtlistener.com/api/rest/v3/clusters/?scdb_id=%s' %caseId)
    data = r.json()
    if data['count'] == 0:
        return opinions

    cluster = data['results'][0]
    for opinion_url in cluster['sub_opinions']:
        r = requests.get(opinion_url)
        data = r.json()
        author = get_author(data['author'])
        html = data['html']
        soup = BeautifulSoup(html)
        text = soup.text
        opinions.append(
            {'author': author,
             'text': text}
        )
    return opinions

# NEW SHIT
def get_text(scdb_id, justice_name):
    """
    Gets text for a given voteId & justiceName (from SCDB)
    """
    cluster = clusters.find_one({'scdb_id': scdb_id})
    if not cluster:
        return ''

    # improve to concatenate multiple opinions.
    cursor = opinions.find({'resource_uri':
                              {'$in': cluster['sub_opinions']},
                              'justiceName': justice_name})
    if cursor.count() == 0:
        return ''

    text = ''
    for opinion in cursor:
        html = opinion['html']
        soup = BeautifulSoup(html)
        text += soup.text

    return text

def match_authors():
    """
    assigns justiceName field from SCDB dataset to people in the
    CourtListener.com dataset.
    """

    cursor = opinions.find({'author': {'$exists': True, '$nin': [None, '']}})
    for i, o in enumerate(cursor):
        author = people.find_one({'resource_uri': o['author']})
        if 'name_first' not in author:
            print ('Could not find author', o['author'],
                   'of case', o['resource_uri'])
        justice_name = get_justice_name(author)
        opinions.update_one({'_id': o['_id']},
                            {'$set': {'justiceName': justice_name}})

def load(data_dir, collection):
    """
    Loads unzipped CourtListener.com Bulk Data API directory (containing .json
    files) into the MongoDB collection given. Returns the collection size.

    @param data_dir: the directory from CourtListener.com containing .json files
    @param collection: the MongoDB collection in which to load the data.
    """
    for fname in os.listdir(data_dir):
        f = open(os.path.join(data_dir, fname))
        data = json.load(f)
        collection.insert_one(data)
    return collection.count()

def load_all(sub_dir__collection_name=None):
    """
    Loads in data from CourtListener.com Bulk Data API.

    @param sub_dir__collection_name (optional): list of tuples
    specifying the subdirectory name and collection name on current client.
    """
    sub_dir__collection_name = sub_dir__collection_name or [
        ('opinions', 'opinions'),
        ('people', 'people'),
        ('clusters', 'clusters')
    ]
    for (subdir, collection_name) in sub_dir__collection_name:
        count = load(os.path.join(path, subdir), db[collection_name])
        print 'Loaded in', count, 'entries for', collection_name

def generate_corpus(outputf='courtlistener.csv'):
    """
    Writes to outputfile a csv with the columns:
    voteId, text, vote

    where:
    - voteId is the voteId from SCDB_2016_01_justiceCentered_Citation.csv
      (i.e. identifier for a justice's vote on a case),
    - text is all the text data that we can get for this vote (in plain text)
    - vote is the justice's vote (see codes here:
      http://scdb.wustl.edu/documentation.php?var=vote)

    @param outputf (optional): filename for output file.
    Defaults to courtlistener.csv
    """
    f = open(os.path.join(path, outputf), 'w')
    fieldnames = ['voteId', 'text', 'vote']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    df = pd.read_csv('../data/SCDB_2016_01_justiceCentered_Citation.csv')
    for index, row in df.iterrows():
        scdb_id = row.caseId
        vote_id = row.voteId
        justice_name = row.justiceName
        opinion = row.opinion
        vote = row.vote
        if opinion in [2, 3]:
            text = get_text(scdb_id, justice_name)
            writer.writerow({'voteId': vote_id,
                             'text': text,
                             'vote': vote})
    f.close()

def set_indices():
    # ensure uniqueness
    people.create_index('resource_uri', unique=True)
    opinions.create_index('resource_uri', unique=True)
    clusters.create_index('resource_uri', unique=True)

def setup_courtlistener():
    """
    This should only be run ONCE on any client.
    """
    print 'Time to pop the cherry.'
    load_all()
    set_indices()
    match_authors()

def is_first_time():
    """
    Tells if this is the first time that the client is running this script.
    """
    return opinions.count() == 0

def main():
    if is_first_time():
        setup_courtlistener()
        generate_corpus()
    else:
        print ("Your cherry has been popped. Are you sure you want to generate"
               "the corpus again?")
        answer = input('Yes or No?[Y/N]')
        if answer.lower().contains('y'):
            generate_corpus()

if __name__ == '__main__':
    main()
