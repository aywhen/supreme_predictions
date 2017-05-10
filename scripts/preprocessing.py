import pandas as pd
import numpy as np
from datetime import datetime

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
chrono_variables = [
    u'dateDecision', u'decisionType', u'term',
    u'naturalCourt', u'chief', u'dateArgument', u'dateRearg']
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
feature_cols = substantive_variables + bg_variables
label_cols = ['partyWinning'] #outcome_variables
id_cols = ['voteId']

# other vars
cutoff_date = datetime(2015, 1, 1) # anything after this date is in test set

# read the file in and construct train/test sets
df = pd.read_csv('../data/SCDB_2016_01_caseCentered_Citation.csv',
                 parse_dates=['dateDecision'])
df = df[pd.notnull(df.partyWinning)]
train = df[df.dateDecision < cutoff_date]
test = df[df.dateDecision >= cutoff_date]

trainX = train[id_cols + feature_cols]
trainY = train[id_cols + label_cols]
testX = test[id_cols + feature_cols]
testY = test[id_cols + label_cols]

# write to files
trainX.to_csv('../data/trainX.csv', index=False)
trainY.to_csv('../data/trainY.csv', index=False)
testX.to_csv('../data/testX.csv', index=False)
testY.to_csv('../data/testY.csv', index=False)
