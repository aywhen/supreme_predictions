import pandas as pd
import numpy as np
from datetime import datetime





# column names by category
id_variables = [
    u'justice', u'justiceName', #Name and unique ID
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
    u'caseDispositionUnusual', u'partyWinning', u'precedentAlteration',  
    u'firstAgreement', u'secondAgreement']

voting_variables = [u'voteUnclear', u'majOpinWriter', u'majOpinAssigner',
                    u'splitVote', u'majVotes', u'minVotes',  u'vote', u'opinion',
                    u'direction', u'majority']

# column names for inclusion in train/test
feature_cols = substantive_variables + bg_variables + chrono_include
label_cols = outcome_variables + voting_variables
id_cols = ['voteId']

# other vars
cutoff_date = datetime(2015, 1, 1) # anything after this date is in test set

# read the file in and construct train/test sets
df = pd.read_csv('../data/SCDB_2016_01_justiceCentered_Citation.csv',
                 parse_dates=['dateDecision'])
df = df[pd.notnull(df.partyWinning)] # only include rows with valid partyWinning
train = df[df.dateDecision < cutoff_date]
test = df[df.dateDecision >= cutoff_date]

trainX = train[id_cols + feature_cols]
trainY = train[id_cols + label_cols]
testX = test[id_cols + feature_cols]
testY = test[id_cols + label_cols]

# write to files
trainX.to_csv('../data/trainX_justice.csv', index=False)
trainY.to_csv('../data/trainY_justice.csv', index=False)
testX.to_csv('../data/testX_justice.csv', index=False)
testY.to_csv('../data/testY_justice.csv', index=False)
