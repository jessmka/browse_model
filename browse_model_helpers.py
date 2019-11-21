import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import requests
import scipy as sp

from sklearn.preprocessing import LabelEncoder, StandardScaler
from spotlight.interactions import Interactions

# from svr.abandon_browse_cache_tables import BrowseHelpers
from svr.dbhelpers import DBHelpers
from svr.utils.s3helpers import s3Helpers
from svr.browse_model.models.sequence_model import SequenceModel
from svr.browse_model.models.send_model import SendModel
from svr.utils.mgemi import core_headers, ft_master_orders_name


from svr.utils.mgemi import inv_cache_from_db
from werkzeug.contrib.cache import SimpleCache

class ModelHelpers():

    SEND_THRESHOLD = 0.5
    SIZE_THRESHOLD = 0.7
    FRACTION_SEND = 0.45    # proportion of people under threshold to send
    # lookback period, fraction send should be function
    INVENTORY_MIN = 1
    CORE_API = os.environ['CORE_API']
    CORE_TOKEN = os.environ['CORE_TOKEN']

    """
    Everyone goes through send model. Those who meet send threshold and
    random sample of those below threshold are passed through sequence model
    """

    def __init__(self):
        self.dbHelpers = DBHelpers()
        self.s3Helpers = s3Helpers()
        self.cache = SimpleCache()

        # self.BrowseHelpers = BrowseHelpers()

    def model_files(self):
        return dict(
        seq_model_pid = dict(
            s3loc = 'jess_models/seq_model_pid.pkl',
            loc = 'svr/browse_model/data/seq_model_pid.pkl'
            ),
        seq_model_cgid = dict(
            s3loc = 'jess_models/seq_model_cgid.pkl',
            loc = 'svr/browse_model/data/seq_model_cgid.pkl'
            ),
        seq_model_le_cat_pid = dict(
            s3loc = 'jess_models/sqmodel_le_cat_pid.pkl',
            loc = 'svr/browse_model/data/sqmodel_le_cat_pid.pkl'
            ),
        send_scaler = dict(
            s3loc = 'jess_models/send_scaler.pkl'
            ),
        send_model_simple = dict(
            s3loc = 'jess_models/send_simple_model.pkl'
            ),
        model_info = dict(
            s3loc = 'jess_models/model_info.pkl'
            )
        )
    size_query = """
        select distinct on (billing_email, category)
        billing_email as email, category as dept, identifier
        from {} ftm
        join dim_product_skus sk on ftm.sales_order_sku=sk.upc
        join dim_product_variants pv on sk.product_variant_id = pv.id
        where billing_email in ('{}')
        order by billing_email, category, order_date desc
        """

    sku_size_query = """
        select product_style_code||'_'||group_code as activity,
          color_code as sksz_color_code,
          pv.id as pvid, identifier as identifier, sk.upc
          , ap.active_flag
          from dim_product_variants pv
          join dim_product_skus sk on pv.id=sk.product_variant_id
          join datascience.active_products ap on sk.upc=ap.upc
        where product_style_code||'_'||group_code
        in ('{}')
        and sk.damage_level_id=1
        """

    get_category_query = """
        select distinct group_code, product_style_code,
        product_style_code||'_'||group_code as activity,
        category as dept
        from public.dim_product_variants
        where category like '%%Shoes'

        """

    get_variant_fields_query = """
        select
           pv.variant_primary_image,
           pv.product_id,
           pv.product_name,
           pv.color,
           pv.material,
           pv.color_code as pv_color_code,
           pv.group_code,
           product_style_code,
           pv.id AS product_variant_id,
           pv.retail_usd,
           pv.factory_region,
           pv.product_style_code|| '_' ||pv.group_code AS activity
        from dim_product_variants pv
        where pv.product_style_code|| '_' ||pv.group_code in ('{}')
        """
    check_if_sent = """select email,
            case when bool_or(send_flag) then 1 else 0 end as sent
             from datascience.send_model_output
             where timestamp >= '{}'::TIMESTAMP - interval '{} hours'
             group by 1
            ;"""

    # def product_dict(self, row):
    #     return dict(product_group_id=row.pid,
    #                    name=row.product_name, product_variant_id=str(row.product_variant_id),
    #                    color_code=row.color_code, image_url=row.variant_primary_image,
    #                    url='http://www.mgemi.com/'+row.pid+'.html#!/color/'+row.color_code,
    #                     url_prefix='http://www.mgemi.com/'+row.pid+'.html', url_suffix='#!/color/'+row.color_code,
    #                    manufacturerRegion=row.factory_region, price = format(row.retail_usd, '.0f'))

    def send_threshold_and_random(self, df, threshold, fraction, update_time):
        """
        Takes a dataframe and selects all rows over threshold,
        then randomly samples a fraction of rows under threshold,
        who have not been sent an email in the last time windown
        """
        was_sent = self.dbHelpers.query_postgres(self.check_if_sent.format(update_time, 8))
        was_sent = was_sent[was_sent.sent==0]
        over = df[(df['send_pred']>= threshold) & (df.email.isin(was_sent.email))]
        under = df[(df['send_pred'] < threshold) & (df.email.isin(was_sent.email))]
        under_rand = under.sample(frac=fraction)
        return pd.concat([over['email'], under_rand['email']])


    def color_codes_to_list_df(self, pid_df):
        """
        From long dataframe of pids and colors, make a
        dataframe with 1 row per user and pid, and list of color codes
        """
        out_df = pd.DataFrame(pid_df[~(pid_df.color_code=='')].groupby(['email','activity'])['color_code'].apply(set)).reset_index() #
        return out_df

    def make_query_string_from_df(self, df, var):
        """
        Create string of unique values from a dataframe column
        separated by commas to be passed in to a query
        """
        lm = list(df[var].unique())
        str1 = "','".join((str(n) for n in lm))
        return str1

    def send_model_output_to_db(self, send_users, send_pred, update_time):
        send_pred = np.reshape(send_pred, -1)
        df = pd.DataFrame(
            np.transpose(np.vstack((send_users,send_pred))),
            columns = ['email','send_pred']
            )
        df['timestamp'] = update_time
        df['send_flag'] = False
        df2 = df.copy()
        df2.columns = ['email', 'prob', 'timestamp', 'send_flag']
        self.dbHelpers.dataframe_to_postgres(df2,
            'datascience.send_model_output')
        return df

    def seq_model_output_to_db(self, pid_pred_df, cat_pred_df, update_time):
        pred_df = pd.concat([pid_pred_df, cat_pred_df], axis = 0)
        dict_list = []
        pred_df.loc[(~pd.isnull(pred_df.color_code)) & (pred_df.activity_type=='pid'),'color_code'] = \
            pred_df.loc[(~pd.isnull(pred_df.color_code)) & (pred_df.activity_type=='pid'),'color_code'].apply(list)
        for name, group in pred_df.groupby(['email']):
            outer_dict = dict(email=name)
            pid_pred_dict = group[group['activity_type']=='pid'][['activity','pred']].to_dict('records')
            cat_pred_dict = group[group['activity_type']=='cgid'][['activity','pred']].to_dict('records')
            outer_dict['pid_pred'] = pid_pred_dict
            outer_dict['cat_pred'] = cat_pred_dict
            color_dict = group[(group['activity_type']=='pid') & (~pd.isnull(group.color_code))][['activity','color_code']].to_dict('records')
            outer_dict['color'] = color_dict
            dict_list.append(outer_dict)
        db_df = pd.DataFrame(dict_list)[['email','pid_pred','cat_pred','color']]
        db_df[['pid_pred', 'cat_pred','color']] = db_df[['pid_pred', 'cat_pred','color']].applymap(lambda x: str(x).replace("'",'"'))
        db_df['timestamp'] = update_time
        db_df.columns = ['email', 'product_pred', 'category_pred', 'color_codes', 'timestamp']
        self.dbHelpers.dataframe_to_postgres(db_df, 'datascience.seq_model_output')


    def sku_inventory_get(self, sku_list):
        fast_inventory = self.cache.get('inventory')
        if fast_inventory is None:
            fast_inventory = inv_cache_from_db(self.dbHelpers)
            self.cache.set('inventory', fast_inventory, timeout=60 * 60)
        sku_set = set()
        for sku in sku_list:
            if fast_inventory.get(str(sku),0) >= self.INVENTORY_MIN:
                sku_set.add(sku)
        return sku_set


    def sku_inventory_get_api(self, sku_list):
        sku_set = set()
        try:
            r = requests.post(self.CORE_API+'/vendors/demandware/lazy/inventoryForSkus',
                  json=dict(skus=sku_list), headers=core_headers())
            for i,k in r.json().items():
                if k.get('stock_level') >= self.INVENTORY_MIN:
                    sku_set.add(i)
        except Exception as e:
            logging.exception(e)
        return sku_set



    def user_sizes(self, df):
        """
        Takes a dataframe of users and pids and returns the most
        recent sizes bought by each user on the matching category
        """
        ft_master_orders_table_name = ft_master_orders_name(self.dbHelpers) #'ft_master_orders'
        email_string = self.make_query_string_from_df(df, 'email')
        size_df = self.dbHelpers.query_postgres(
            self.size_query.format(ft_master_orders_table_name, email_string)
            )
        product_df = self.dbHelpers.query_postgres(self.get_category_query)
        df = df.merge(product_df, on='activity', how='left') #get category

        df = df.merge(size_df, on=['email','dept'], how='left') #get most recent size for that category
        return df

    def product_inventory_sizes_threshold(self, prod_df, skus_w_inventory):
        """
        Takes a dataframe of pids and color codes and an array of skus that have inventory,
        returns pids with inventory as a fraction of all sizes over a given threshold
        """

        prod_df['has_inventory'] = np.where(prod_df['upc'].isin(skus_w_inventory),1,0)
        prod_agg = pd.DataFrame(
            prod_df.groupby(
                ['activity','sksz_color_code'])['has_inventory'].agg(
                    {'sku':'count','has_inventory':np.sum}
                )
            ).reset_index()
        prod_agg['frac_of_color'] = prod_agg['has_inventory']/prod_agg['sku']
        prod_agg = prod_agg[prod_agg['frac_of_color'] >= self.SIZE_THRESHOLD]
        return prod_agg

    def row_color_check(self, row):
        try:
            if pd.isna(row.color_code) and row.active_flag=='active': #
                return True
            elif row['sksz_color_code'] in row['color_code']:
                return True
            else:
                return False
        except TypeError:
            return False


    def check_inventory_on_pred_df(self, pid_df):
        """
        Takes dataframe of users, pids and predictions, gets category of pids,
        and most recent size they have bought in that category
        and checks product inventory for buyers and leads,
        filters only rows with inventory
        """

        pid_df = self.user_sizes(pid_df)
        pid_string = self.make_query_string_from_df(pid_df, 'activity') #get distinct pids
        sku_size_df = self.dbHelpers.query_postgres(self.sku_size_query.format(pid_string))

        prods = pid_df[['activity','identifier']].drop_duplicates()
        prods = prods.merge(sku_size_df, on='activity')
        prods = prods[pd.isnull(prods.identifier_x)|(prods.identifier_x==prods.identifier_y)]

        skus_to_check = list(prods.upc.unique())
        skus_w_inventory = self.sku_inventory_get(skus_to_check)
        prods = self.product_inventory_sizes_threshold(prods, skus_w_inventory)


        # row_color_check = lambda row: ((row.color_code=={''})|(row.sksz_color_code in row.color_code))
        buyer_df = pid_df[~pd.isnull(pid_df.identifier)].merge(sku_size_df, on=['activity','identifier'])
        if not buyer_df.empty:
            buyer_df = buyer_df[buyer_df.apply(self.row_color_check, axis=1)]
            buyer_df = buyer_df[buyer_df.upc.isin(skus_w_inventory)]

        lead_df = pid_df[pd.isnull(pid_df.identifier)].merge(sku_size_df, on = 'activity')
        if not lead_df.empty:
            lead_df = lead_df[lead_df.apply(self.row_color_check, axis=1)]
            # lead_prods = self.product_inventory_sizes_threshold(prods, skus_w_inventory)
            lead_df = lead_df.merge(prods, on=['activity', 'sksz_color_code'],
                how='inner') #this is wrong right_on = ['pid','sksz_color_code'],
         # buyer_df = self.buyer_inv_check(pid_df,sku_size_df)
        # lead_df = self.lead_inv_check(pid_df, sku_size_df)
        out_df = pd.concat([buyer_df,lead_df])
        # out_df.rename(columns={'sksz_color_code':'color_code'}, inplace=True)
        return out_df



    def get_variant_fields(self, rec_df):
        """
        Given a dataframe of products, returns product variant
        fields merged on to original dataframe
        """
        rec_df = rec_df[rec_df['activity_type']=='pid']
        pid_list = self.make_query_string_from_df(rec_df, 'activity')
        prod_df = self.dbHelpers.query_postgres(
            self.get_variant_fields_query.format(pid_list)
            )
        return prod_df

    def get_max_row_dedupe(self, pdf, user_field, max_field):
        """
        Finds rows with the max prediction for a user, then
        randomly picks a single one
        """
        idx = pdf.sort_values(max_field, ascending=False).drop_duplicates(user_field)
        return idx

    def category_map_to_product(self, cat_df):
        """
        Add special landing category pages in to data as pdps - replace them with a pid

        """
        categories_to_send = self.dbHelpers.query_postgres(
            'select category, pid from datascience.category_page_product_map'
            )
        # cat_df_filtered = cat_df[cat_df['activity'].isin(categories_to_send['category'])]
        cat_df_filtered = cat_df.merge(
            categories_to_send,
            left_on='activity', right_on='category', how='inner')
        # cat_df_filtered['activity'] = new_activity_col
        cat_df_filtered.drop(['activity','category'], axis=1, inplace=True)
        cat_df_filtered.rename(columns={'pid': 'activity'}, inplace=True)
        cat_df_filtered['color_code'] = np.nan
        cat_df_filtered['activity_type'] = 'pid'
        return cat_df_filtered


def main():
    startdt = '2018-07-20'
    enddt = '2018-07-21'
    bucket = 'datasciencemodels.mgemi.com'

    # model_files = mh.ModelHelpers.model_files()
    # for key, value in model_files.items():
    #     print(key, value)
    #     download_from_s3(bucket, value['s3loc'], value['loc'])

    # placeholder to break out 2 groups, 1st group does not go through send model
    # user_idx, Xs = mh.ModelPipeline.send_model_pipeline(startdt, enddt)
    # mh.download_from_s3('jessmodels/seq_model.pkl', 'Models/dl_delete.pkl')
    # mh.download_from_s3('jessmodels/send_log_model.yaml','Models/dl_delete2.pkl')
    # mh.download_from_s3('jessmodels/send_log_model_weights.h5','Models/dl_delete3.pkl')

    # placeholder to break out 2 groups, 1st group does not go through send model
    # user_idx, Xs = mh.model_pipeline.send_model_pipeline(startdt, enddt)
    # send_pred = mh.send_model.predict_model(model, user_idx, X_scaled)


    # # choose some threshold from them to go through sequence model
    # # the people going through need to be an arg of the pipeline
    # pid_seq_data = mh.model_pipeline.sequence_model_pipeline('pid')
    # cat_seq_data = mh.model_pipeline.sequence_model_pipeline('cgid')
    # pid_pred = mh.sequencemodel.predict_model(pid_seq_data)
    # cat_pred = mh.sequencemodel.predict_model(cat_seq_data)
    # placeholder to combine cats and pids, each user gets payload object
    # placeholder to send to iterable




if __name__ == "__main__":
    main()













