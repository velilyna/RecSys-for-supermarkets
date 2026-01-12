import os
import pandas as pd
import pickle
from collections import defaultdict



def prepare_tafeng(input_path="data/ta_feng_all_months_merged.csv",
                   out_path="data/processed/tafeng_preprocessed.pkl"):
    if not os.path.exists(input_path):
        print(f"Файл {input_path} не найден")
        return

    df = pd.read_csv(input_path)
    df.columns = [c.upper() for c in df.columns] 

    if 'TRANSACTION_DT' not in df.columns or 'CUSTOMER_ID' not in df.columns or 'PRODUCT_ID' not in df.columns:
        raise KeyError(f"Не найдены нужные колонки")

    df['TRANSACTION_DT'] = pd.to_datetime(df['TRANSACTION_DT'], errors='coerce')

    df = df.dropna(subset=['CUSTOMER_ID', 'PRODUCT_ID'])
    df = df.sort_values(['CUSTOMER_ID', 'TRANSACTION_DT'])

    grouped = df.groupby(['CUSTOMER_ID', 'TRANSACTION_DT'])['PRODUCT_ID'].apply(list).reset_index()

    user_hist = defaultdict(list)
    for _, row in grouped.iterrows():
        user = int(row['CUSTOMER_ID'])
        ts = row['TRANSACTION_DT']
        items = [int(i) for i in row['PRODUCT_ID']]
        user_hist[user].append((ts, items))

    for u in user_hist:
        user_hist[u].sort(key=lambda x: x[0])
    user_hist = {u: seq for u, seq in user_hist.items() if len(seq) >= 5}

    all_users = sorted(user_hist.keys())
    all_items = sorted({i for seq in user_hist.values() for _, items in seq for i in items})
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {i: j for j, i in enumerate(all_items)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"user_hist": user_hist, "user2idx": user2idx, "item2idx": item2idx}, f)

    print(f" {len(user2idx)} пользователей, {len(item2idx)} товаров.")



def prepare_dunnhumby(data_dir="data/",
                      out_path="data/processed/dunnhumby_preprocessed.pkl"):
    trans_path = os.path.join(data_dir, "transaction_data.csv")
    prod_path = os.path.join(data_dir, "product.csv")

    if not os.path.exists(trans_path):
        print(f"Не найден {trans_path}")
        return

    trans = pd.read_csv(trans_path)
    prods = pd.read_csv(prod_path) if os.path.exists(prod_path) else None

    trans.columns = [c.lower() for c in trans.columns]
    if prods is not None:
        prods.columns = [c.lower() for c in prods.columns]
        if 'product_id' in prods.columns:
            trans = trans.merge(prods[['product_id']], on='product_id', how='left')


    if 'day' in trans.columns:
        trans['date'] = pd.to_datetime(trans['day'], errors='coerce', origin='1900-01-01', unit='D')
    elif 'date' in trans.columns:
        trans['date'] = pd.to_datetime(trans['date'], errors='coerce')

    trans = trans.dropna(subset=['household_key', 'product_id'])
    trans = trans.sort_values(['household_key', 'date'])

    grouped = trans.groupby(['household_key', 'date'])['product_id'].apply(list).reset_index()

    user_hist = defaultdict(list)
    for _, row in grouped.iterrows():
        user = int(row['household_key'])
        ts = row['date']
        items = [int(i) for i in row['product_id']]
        user_hist[user].append((ts, items))

    for u in user_hist:
        user_hist[u].sort(key=lambda x: x[0])
    user_hist = {u: seq for u, seq in user_hist.items() if len(seq) >= 5}

    all_users = sorted(user_hist.keys())
    all_items = sorted({i for seq in user_hist.values() for _, items in seq for i in items})
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {i: j for j, i in enumerate(all_items)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"user_hist": user_hist, "user2idx": user2idx, "item2idx": item2idx}, f)

    print(f" {len(user2idx)} пользователей, {len(item2idx)} товаров")

if __name__ == "__main__":
    prepare_tafeng()
    prepare_dunnhumby()