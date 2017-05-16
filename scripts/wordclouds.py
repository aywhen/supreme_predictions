import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

# http://scdb.wustl.edu/documentation.php?var=direction
CONSERVATIVE = 1
LIBERAL = 2

def get_data():
    train_x = np.loadtxt('../data/out_courtlistener_bow5362_1000.csv',
                         delimiter=',')
    train_y = np.loadtxt('../data/out_courtlistener_classes5362_1000.txt')
    test_x = np.loadtxt('../data/out_courtlistener_test_bow5362_1000.csv',
                        delimiter=',')
    test_y = np.loadtxt('../data/out_courtlistener_testY5362_1000.txt')
    vocab = np.loadtxt('../data/out_courtlistener_vocab_5362_1000.txt',
                             dtype=str)
    voteIds = np.loadtxt('../data/out_courtlistener_all_samples.txt',
                         dtype=str)
    voteIds = voteIds.reshape(voteIds.shape[0], 1)
    bow = np.concatenate((train_x, test_x))
    bow = np.hstack((voteIds, bow))
    votes = np.concatenate((train_y, test_y))
    votes = votes.reshape(votes.shape[0], 1)
    votes = np.hstack((voteIds, votes))
    scdb = pd.read_csv('../data/SCDB_2016_01_justiceCentered_Citation.csv')
    scdb = scdb.fillna(value=-1)
    samples = pd.DataFrame(data=voteIds, columns=['voteId'])
    samples = samples.merge(scdb, how='left', on='voteId')
    return (bow, votes, samples, vocab)

def filter_bow(bow, samples, mask_f):
    bow_df = pd.DataFrame(
        data=bow,
        columns=['voteId'] + range(bow.shape[1] - 1))
    samples_bow = samples.merge(bow_df, how='left', on='voteId')
    filtered = samples_bow[mask_f(samples_bow)]
    filtered_bow = filtered[bow_df.columns]
    filtered_samples = filtered[samples.columns]
    return (filtered_bow.values, filtered_samples)

def transform_bow(bow, transform=None):
    transform_default = lambda x: x
    transform = transform or transform_default
    voteIds = bow[:,0]
    voteIds = voteIds.reshape(voteIds.shape[0], 1)
    bow = np.delete(bow, 0, axis=1)
    bow = np.array(bow, dtype=float)
    transformed_bow = transform(bow)
    return np.hstack((voteIds, transformed_bow))

def get_conservative(data):
    bow, votes, samples, vocab = data
    mask_f = lambda x: x.direction==CONSERVATIVE
    filtered_bow, filtered_samples = filter_bow(bow, samples,
                                                mask_f)
    votes = filtered_samples.vote.values
    return (filtered_bow, votes, filtered_samples, vocab)

def get_liberal(data):
    bow, votes, samples, vocab = data
    mask_f = lambda x: x.direction==LIBERAL
    filtered_bow, filtered_samples = filter_bow(bow, samples,
                                                mask_f)
    votes = filtered_samples.vote.values
    return (filtered_bow, votes, filtered_samples, vocab)

def get_tfidf(data):
    from sklearn.feature_extraction.text import TfidfTransformer

    bow, votes, samples, vocab = data
    transformer = TfidfTransformer(smooth_idf=False)
    transform_f = lambda x: transformer.fit_transform(x).toarray()
    transformed_bow = transform_bow(bow, transform=transform)
    return (transformed_bow, votes, filtered_samples, vocab)

def get_conservative_tfidf(data):
    bow_tfidf, votes, samples, vocab = get_conservative(data,
                                                        transform=transform_f)
    return (bow_tfidf, samples)

def get_top_words(x, y, vocab, k=100):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # scores are the sum across each column, i.e. the total count across all
    # instances of this word.
    scores = x.sum(axis=0)
    return get_top_scores(scores, vocab, k=k)

def get_top_scores(scores, vocab, mask=None, k=None):
    df = pd.DataFrame(data=scores, index=vocab, columns=['score'])
    if mask is not None:
        masked = df[mask]
        sorted_scores = masked
    else:
        sorted_scores = df
    sorted_scores = sorted_scores.sort_values(by='score', ascending=False)
    if k:
        sorted_scores = sorted_scores[:k]
    frequencies = dict(zip(sorted_scores.index,
                           [i[0] for i in sorted_scores.values]))
    return frequencies

def get_top_words_by_chi2(x, y, vocab, k=100):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(x, y)
    return get_top_scores(selector.scores_, vocab, k=k,
                          mask=selector.get_support())

def generate_wordcloud_chi2(data, label='direction', k=100):
    bow, votes, samples, vocab = data
    x = np.delete(bow, 0, axis=1)
    y = samples[label].values
    #return x, y, vocab
    top = get_top_words_by_chi2(x, y, vocab)
    # return top
    generate_wordcloud(top)

def generate_wordclouds_chi2_conservative_liberal(data):
    conservative = get_conservative(data)
    liberal = get_liberal(data)
    labels = ['vote', 'majority']
    for name, dataset in [('conservative', conservative), ('liberal', liberal)]:
        print name
        for label in labels:
            print label
            generate_wordcloud_chi2(liberal, label='vote')

def generate_wordclouds_raw_frequency(data):
    # c_bow, c_votes, c_samples, c_vocab = get_conservative(data)
    # l_bow, l_votes, l_samples, l_vocab = get_liberal(data)
    i = 1
    for bow, votes, samples, vocab in [get_conservative(data),
                                       get_liberal(data)]:
        if i == 1:
            print 'conservative'
        else:
            print 'liberal'
        _generate_wordclouds_raw_freq(bow, samples, vocab)
        i += 1
    pass

def _generate_wordclouds_raw_freq(bow, samples, vocab):
    labels = ['vote', 'majority']
    x = np.delete(bow, 0, axis=1)
    for label in labels:
        print label
        y = samples[label].values
        top = get_top_words(x, y, vocab)
        generate_wordcloud(top)


def generate_wordcloud_tfidf(data):
    # TFIDF DOESN'T MAKE SENSE
    from sklearn.feature_extraction.text import TfidfTransformer

    transformer = TfidfTransformer(smooth_idf=False)
    transform_f = lambda x: transformer.fit_transform(x).toarray()

    bow, votes, conservative_samples, vocab = get_conservative(data)
    conservative_tfid = transform_bow(bow, transform=transform_f)
    bow, votes, lib_samples, vocab = get_liberal(data)
    liberal_tfid = transform_bow(bow, transform=transform_f)

    # conservative_tfidf, conservative_samples = get_conservative_tfidf(data)
    labels = ['vote', 'majority']
    x = conservative_tfidf.values
    x = np.delete(x, 0, axis=1)
    for label in labels:
        y = conservative_samples[label].values
        top = get_top_words(x, y, vocab)
        generate_wordcloud(top)
    pass

def generate_wordcloud(frequencies, k=100):
    """
    @param frequencies: dict mapping strings to numerical counts
    """
    # adjust options on WordCloud
    # top = get_top_words(x, y, vocab, k=k)
    wc = WordCloud()
    wc.generate_from_frequencies(frequencies)
    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def generate_all():
    # word clouds for most predictive of liberal, conservative,
    # yes vote, no vote.
    data = get_data()
    #generate_wordcloud_chi2(data, 'direction')
    #generate_wordcloud_chi2(data, 'vote')
    #generate_wordcloud_chi2(data, 'majority')
