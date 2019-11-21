import numpy as np
import os
import requests


def core_headers():
    return {
        'authorization': 'Bearer ' + os.environ['CORE_TOKEN'],
        'cache-control': 'no-cache',
        'content-type': 'application/json',
        'x-organization': '1'
    }


def prod_db_query():
    return """
                select id, replace(product_name,'  ', ' ') as product_name, material, color,
                product_style_code || '_' || group_code as product_group_code,
                color_code, edition_code, product_style_code, retail_usd, factory_region,
                filter_color,
                variant_primary_image, group_code,
                coalesce(group_creative->>'shortDescription','') as short_desc,
                extract(year from plan_date) as launch_year,
                id as pvid
                from dim_product_variants
                where category like 'Shoes';
        """

def inv_cache_from_db(dbHelper, upper_lim=np.inf):
    """
        Uses the 'demandware.inventory.publish' event from DW
        to build a dict of in-stock products
    """
    inv_query = """
        select
            event
        from
            commersive.events
        where
            type = 'demandware.inventory.publish'
        order by id desc limit 1;
    """

    dt = dbHelper.query_postgres(inv_query)

    inv_dict = dict()

    for i in dt.event.iloc[0]:
        if (i.get('qty',0) > 0) and (i.get('qty',0) <= upper_lim):
            inv_dict[i.get('sku')] = i.get('qty')

    return inv_dict


def prodgroup_cache_waitlist_from_api(app_cache, product_group, conf):
    #Check for recent response from cache, if nothing then go to API
    CACHE_TIMEOUT = 300
    inventory_set = app_cache.get(product_group)
    if inventory_set is None:
        ep = conf['ep']
        headers = conf['headers']
        bh = conf['bh']
        inventory_info = requests.get(ep+product_group, headers=headers).json()
        inventory_set = bh.inventory_data(inventory_info)
        app_cache.set(product_group, inventory_set, CACHE_TIMEOUT)

    return inventory_set


def proddata_cache_waitlist_from_api(app_cache, product_code, conf):
    #Check for recent response from cache, if nothing then go to API
    CACHE_TIMEOUT = 300
    prod_data = app_cache.get('d'+product_code)
    if prod_data is None:
        ep = conf['ep']
        headers = conf['headers']
        bh = conf['bh']
        prod_data = requests.get(ep+product_code, headers=headers).json()
        app_cache.set('d'+product_code, prod_data, CACHE_TIMEOUT)

    return prod_data

#SOME UTILS for fetching data from MG analytics DB

class DB(object):
    def __init__(self, dbhelper):
        self.dbhelper = dbhelper

class DBSalesData(DB):

    sales_query = """
        select
            lower(billing_email) as email, identifier, product_variant_id, category,
            li.order_date
        from
            {} li
        join
            dim_product_skus sk on sk.upc = li.sales_order_sku
        join
            dim_product_variants vk on vk.id = sk.product_variant_id
    """

    MIN_UNITS = 3

    def get_data(self):
        ftm_name = ft_master_orders_name(self.dbhelper)
        dt = self.dbhelper.query_postgres(self.sales_query.format(ftm_name))
        idx = dt.query("category=='Shoes'").groupby('email').product_variant_id.count() > self.MIN_UNITS
        valid_emails = idx.index[idx]
        return dt[dt.email.isin(valid_emails) & (dt.category == 'Shoes')]
class DBPageviewsData(DB):

    pageview_query = """
    select 
      pv.id as product_variant_id,
      substring(substring(cpv.query_string from 'pid=([0-9\_]+).*'), 1, 7) as product_style_code, 
      substring(cpv.query_string from 'pid=([0-9]+).*') pv2, 
      substring(cpv.query_string from 'pid=([0-9\_]+){7}') as product_style_code, 
      substring(substring(cpv.query_string from 'pid=([0-9\_]+).*'), 9, 10) as group_code, 
      substring(url from 'color/([0-9]+)') as color_code,
      query_string, *
    from 
        commersive.clickstream_page_views cpv
    join 
        dim_product_variants pv on 
        substring(substring(cpv.query_string from 'pid=([0-9\_]+).*'), 1, 7) = pv.product_style_code and
        substring(substring(cpv.query_string from 'pid=([0-9\_]+).*'), 9, 10)= pv.group_code and
        substring(url from 'color/([0-9]+)')=pv.color_code    
    """

    MIN_UNITS = 3

    def get_data(self):
        ftm_name = ft_master_orders_name(self.dbhelper)
        dt = self.dbhelper.query_postgres(self.sales_query.format(ftm_name))
        idx = dt.query("category=='Shoes'").groupby('email').product_variant_id.count() > self.MIN_UNITS
        valid_emails = idx.index[idx]
        return dt[dt.email.isin(valid_emails) & (dt.category == 'Shoes')]

class DBActiveStyleData(DB):

    active_query = """select distinct pvid::int from datascience.active_products where active_flag = 'active'"""

    def get_data(self):
        dt = self.dbhelper.query_postgres(self.active_query)
        return set(dt.pvid)


class DBVariantSKU(DB):

    query = """
            select identifier, product_variant_id, upc from dim_product_skus
        """

    def get_data(self):
        return self.dbhelper.query_postgres(self.query)


class DBProdDB(DB):

    def get_data(self):
        dt = self.dbhelper.query_postgres(prod_db_query())
        return dt.set_index('id').to_dict(orient='index')

def ft_master_orders_name(dbhelper):
        """
        Gets current name of looker_scratch ft_master_orders table
        """
        query_table = """
            select concat('looker_scratch.', table_name) as table_name
            from information_schema.tables
            where table_schema = 'looker_scratch' and table_name like '%ft_master_orders' limit 1;
        """
        table_name_df = dbhelper.query_postgres(query_table)
        table_name = table_name_df.iat[0,0]

        return table_name
