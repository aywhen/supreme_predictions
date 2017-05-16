import pandas as pd
import numpy as np
from datetime import datetime
#import preprocess_text

# column names by category
id_variables = [
    u'caseId', u'docketId', u'caseIssuesId', u'voteId',
    u'usCite', u'sctCite', u'ledCite', u'lexisCite',
    u'docket']
bg_variables = [
    u'caseName', u'petitioner', u'petitionerState',
    u'respondent', u'respondentState', u'jurisdiction',
    u'adminAction', u'adminActionState', u'threeJudgeFdc',
    u'caseOrigin', u'caseOriginState', u'caseSource',
    u'caseSourceState', u'lcDisagreement', u'certReason',
    u'lcDisposition', u'lcDispositionDirection',
]
chrono_include = [u'naturalCourt', u'chief']
chrono_donotinclude = [u'dateDecision', u'decisionType', u'term',
                       u'dateArgument', u'dateRearg']
chrono_variables = chrono_include + chrono_donotinclude
substantive_variables = [
    u'issue', u'issueArea', u'decisionDirection',
    u'decisionDirectionDissent', u'authorityDecision1',
    u'authorityDecision2', u'lawType', u'lawSupp', u'lawMinor']
outcome_variables = [
    u'declarationUncon', u'caseDisposition',
    u'caseDispositionUnusual', u'partyWinning', u'precedentAlteration']
voting_variables = [u'voteUnclear', u'majOpinWriter', u'majOpinAssigner',
                    u'splitVote', u'majVotes', u'minVotes']

# column names for inclusion in train/test
feature_cols = substantive_variables + bg_variables + chrono_include
label_cols = ['partyWinning'] #outcome_variables
id_cols = ['voteId']

# other vars
cutoff_date = datetime(2015, 1, 1) # anything after this date is in test set

# --- MAIN ---
def main():
    # read the file in and construct train/test sets
    df = pd.read_csv('../data/SCDB_2016_01_caseCentered_Citation.csv',
                     parse_dates=['dateDecision'])
    df = df[pd.notnull(df.partyWinning)] # only include rows with valid partyWinning
    train = df[df.dateDecision < cutoff_date]
    test = df[df.dateDecision >= cutoff_date]

    # split text and numeric
    trainX = train[id_cols + feature_cols]
    trainX_text = train[feature_cols].select_dtypes(['object'])
    trainX_text = trainX_text.assign(voteId=train.voteId)
    trainX_numeric = train[feature_cols].select_dtypes(['number'])
    trainX_numeric.assign(voteId=train.voteId)
    trainY = train[id_cols + label_cols]
    testX = test[id_cols + feature_cols]
    testX_text = test[feature_cols].select_dtypes(['object'])
    testX_text = testX_text.assign(voteId=test.voteId)
    testX_numeric = test[feature_cols].select_dtypes(['number'])
    testX_numeric.assign(voteId=test.voteId)
    testY = test[id_cols + label_cols]

    # write to files
    trainX.to_csv('../data/trainX.csv', index=False)
    trainX_text.to_csv('../data/trainX_text.csv', index=False)
    trainX_numeric.to_csv('../data/trainX_num.csv', index=False)
    trainY.to_csv('../data/trainY.csv', index=False)
    testX.to_csv('../data/testX.csv', index=False)
    testX_text.to_csv('../data/testX_text.csv', index=False)
    testX_numeric.to_csv('../data/testX_num.csv', index=False)
    testY.to_csv('../data/testY.csv', index=False)

if __name__=='__main__':
    main()
