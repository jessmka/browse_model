import argparse
import numpy as np
import pandas as pd
import pickle

from datetime import datetime
from sklearn.preprocessing import LabelEncoder  #, OneHotEncoder, StandardScaler, MinMaxScaler
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.interactions import Interactions
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.representations import CNNNet
from svr.utils.s3helpers import s3Helpers
from svr.dbhelpers import DBHelpers

class SequenceModelTraining():

    MAX_SEQ_LENGTH = 10
    SEQ_STEP_SIZE = 10

    def __init__(self):
        self.now = int(datetime.timestamp(datetime.now()))
        self.s3Helpers = s3Helpers()
        self.dbHelpers = DBHelpers()
        self.type_iters = dict(pid=9, cgid=9)   #how many epochs each model should run, if none then best_params value
    
    def load_params(self):
        best_params = self.s3Helpers.download_from_s3_io(
            'jess_models/best_params.pkl', 
            pickle.load
            )
        return best_params


    views_query = """
        select eu.email, cpv.session_id,
        to_timestamp(timestamp/1000) as activity_dt, 
        date_trunc('day',to_timestamp(timestamp/1000)) as day,
        case when split_part(split_part(query_string,'cgid=',2),'&',1) <> '' then split_part(split_part(query_string,'cgid=',2),'&',1) 
        else SPLIT_PART(SPLIT_PART(SPLIT_PART(cpv.query_string,'pid=',2),'&',1),'?',1)
        end as activity, 
        SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(SPLIT_PART(url,'/color/',2),'/',1),'?',1),'&',1),'http',1) as color_code,
        case when split_part(split_part(query_string,'cgid=',2),'&',1) <> '' then 'cgid' else 'pid' end as activity_type
        from commersive.clickstream_page_views cpv
        join datascience.email_uuids eu on cpv.user_uuid = eu.uuid
        where (SPLIT_PART(SPLIT_PART(SPLIT_PART(cpv.query_string,'pid=',2),'&',1),'?',1) <> ''
         or split_part(split_part(query_string,'cgid=',2),'&',1) <> '') 
         and cpv.timestamp > 1000*EXTRACT(epoch FROM '{}'::TIMESTAMP)::BIGINT
         AND cpv.host in ('mgemi.com', 'www.mgemi.com', 'production-veloce-rcw.demandware.net')
        order by 1,2

        ;"""

    def seq_train_test_split(self, df, le_user, le_cat_pid):
        """ 
        Samples user/sessions from dataframe
        and builds train and test dataframe and full dataframe
        """
        user_days = df[['email', 'session_id']].drop_duplicates()

        fg = user_days.groupby('email')
        nmax=4   # smaller than this just sample one user/date, otherwise sample 25% 

        df_test = fg.apply(lambda x: x.sample(frac=.25) if len(x) >= nmax else x.sample(n=1))
        df_test.reset_index(level=0, inplace=True, drop=True)

        pid_df_test = df[df['session_id'].isin(df_test.session_id)]
        pid_df_train = df[~(df['session_id'].isin(pid_df_test.session_id)) & df.activity.isin(pid_df_test.activity)]
        pid_df_test = pid_df_test[pid_df_test.activity.isin(pid_df_train.activity)]

        train_idx_user = le_user.transform(pid_df_train['email'])
        train_idx_pid = le_cat_pid.transform(pid_df_train['activity'])
        test_idx_user = le_user.transform(pid_df_test['email'])
        test_idx_pid = le_cat_pid.transform(pid_df_test['activity'])

        interactions_train = Interactions(train_idx_user, train_idx_pid+1,
                                                 timestamps = pid_df_train['activity_dt'])
        sequential_interaction_train = interactions_train.to_sequence(
            max_sequence_length=self.MAX_SEQ_LENGTH, step_size=self.SEQ_STEP_SIZE
            )

        interactions_test = Interactions(test_idx_user, test_idx_pid+1, timestamps = pid_df_test['activity_dt'])
        sequential_interaction_test = interactions_test.to_sequence(
            max_sequence_length=self.MAX_SEQ_LENGTH, step_size=self.SEQ_STEP_SIZE
            )


        full_idx_user = le_user.transform(df['email'])
        full_idx_activity = le_cat_pid.transform(df['activity'])
        interactions_full = Interactions(full_idx_user, full_idx_activity+1,
                                             timestamps = df['activity_dt'])
        sequential_interaction_full = interactions_full.to_sequence(
            max_sequence_length=self.MAX_SEQ_LENGTH, step_size=self.SEQ_STEP_SIZE
            )

        return sequential_interaction_train, sequential_interaction_test, sequential_interaction_full


    def get_data_and_encoder(self, start_date):
        """Query data and return it and fitted label encoder"""
        le_user = LabelEncoder()
        le_cat_pid = LabelEncoder()
        df0 = self.dbHelpers.query_postgres(self.views_query.format(start_date))
        le_user.fit(df0['email'])
        le_cat_pid.fit(df0['activity'])
        return df0, le_user, le_cat_pid

    def model_eval(self, model, test, k):
        """Return model evaluation scores"""
        # pr = sequence_precision_recall_score(model, test, k, exclude_preceding=False)
        mrr = sequence_mrr_score(model, test)
        return dict(
                mrr=np.mean(mrr)
                # ,precision=np.mean(pr[0]),
                # recall=np.mean(pr[1])
                )


    def make_model_dict(self, model, model_type, evals):
        """ Make dict with name of model and object"""
        model_file = 'seq_'+model_type+'_model'+str(self.now)
        model_dict = dict(
            name=model_file,
            results=evals,
            obj=model
            )
        return model_dict



    def train(self, seq, iter_=None):
        rb = self.best_params
        if iter_ is None:
            iter_ = rb['n_iter']

        net = CNNNet(seq.num_items,
                     embedding_dim=rb['embedding_dim'],
                     kernel_width=rb['kernel_width'],
                     dilation=rb['dilation'],
                     num_layers=rb['num_layers'],
                     nonlinearity=rb['nonlinearity'],
                     residual_connections=rb['residual'])

        smodel = ImplicitSequenceModel(loss=rb['loss'],
                                      representation=net,
                                      batch_size=rb['batch_size'],
                                      learning_rate=rb['learning_rate'],
                                      l2=rb['l2'],
                                      n_iter= iter_, 
                                      use_cuda=False,
                                      random_state=np.random.RandomState(100))

        smodel.fit(seq, verbose=True)
        return smodel

    def do_training(self, start_date):
        self.best_params = self.load_params()
        df0, le_user, le_activity = self.get_data_and_encoder(start_date)
        model_info = dict(sqmodel_le_cat_pid=le_activity)

        for model_type in ('pid', 'cgid'):
            seq_train, seq_test, seq_full = self.seq_train_test_split(
                df0[df0['activity_type']==model_type], 
                le_user, 
                le_activity)
            n_iter = self.type_iters[model_type]
            model = self.train(seq_full, n_iter)
            eval_results = self.model_eval(model, seq_test, 3)
            model_info['seq_model_'+model_type] = self.make_model_dict(
            model, model_type, eval_results
            )

        self.s3Helpers.pickle_load_to_s3_io(model_info, 'jess_models/model_info.pkl')













