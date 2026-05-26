import os
import pickle
from collections import defaultdict

import pandas as pd


def iterative_k_core(
    df,
    user_col,
    item_col,
    basket_col,
    min_user_baskets=5,
    min_item_interactions=5,
    max_iter=20,
):
    df = df.copy()

    for it in range(max_iter):
        old_len = len(df)
        old_users = df[user_col].nunique()
        old_items = df[item_col].nunique()

        # 1. Фильтруем товары по числу interaction rows
        item_counts = df[item_col].value_counts()
        good_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df[item_col].isin(good_items)].copy()

        # 2. Удаляем дубликаты товара внутри одной корзины
        df = df.drop_duplicates(subset=[user_col, basket_col, item_col])

        # 3. Удаляем пустые корзины автоматически через оставшиеся rows
        # 4. Фильтруем пользователей по числу непустых корзин
        user_basket_counts = df.groupby(user_col)[basket_col].nunique()
        good_users = user_basket_counts[
            user_basket_counts >= min_user_baskets
        ].index

        df = df[df[user_col].isin(good_users)].copy()

        new_len = len(df)
        new_users = df[user_col].nunique()
        new_items = df[item_col].nunique()

        print(
            f"  k-core iter {it + 1}: "
            f"rows {old_len}->{new_len}, "
            f"users {old_users}->{new_users}, "
            f"items {old_items}->{new_items}"
        )

        if new_len == old_len:
            break

    return df


def build_user_hist_from_df(
    df,
    user_col,
    item_col,
    time_col,
    basket_col,
):
    """
    Собирает user_hist:
        user -> [(timestamp, [item1, item2, ...]), ...]
    """
    df = df.sort_values([user_col, time_col])

    grouped = (
        df.groupby([user_col, basket_col, time_col])[item_col]
        .apply(lambda x: sorted(set(int(i) for i in x)))
        .reset_index()
    )

    user_hist = defaultdict(list)

    for _, row in grouped.iterrows():
        user = int(row[user_col])
        ts = row[time_col]
        items = row[item_col]

        if len(items) > 0:
            user_hist[user].append((ts, items))

    for u in user_hist:
        user_hist[u].sort(key=lambda x: x[0])

    return dict(user_hist)


def save_processed(user_hist, out_path):
    all_users = sorted(user_hist.keys())

    all_items = sorted(
        {
            item
            for seq in user_hist.values()
            for _, basket in seq
            for item in basket
        }
    )

    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {item: j for j, item in enumerate(all_items)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "user_hist": user_hist,
                "user2idx": user2idx,
                "item2idx": item2idx,
            },
            f,
        )

    n_baskets = sum(len(seq) for seq in user_hist.values())
    n_interactions = sum(
        len(basket)
        for seq in user_hist.values()
        for _, basket in seq
    )

    print("\nSaved:", out_path)
    print(f"  users: {len(user2idx)}")
    print(f"  items: {len(item2idx)}")
    print(f"  baskets: {n_baskets}")
    print(f"  interactions: {n_interactions}")

    if len(user2idx) > 0:
        print(f"  avg baskets/user: {n_baskets / len(user2idx):.2f}")

    if n_baskets > 0:
        print(f"  avg items/basket: {n_interactions / n_baskets:.2f}")


def prepare_tafeng(
    input_path="data/ta_feng_all_months_merged.csv",
    out_path="data/processed/tafeng_preprocessed.pkl",
    min_user_baskets=5,
    min_item_interactions=5,
):
    print("\n=== Preparing TaFeng ===")

    if not os.path.exists(input_path):
        print(f"Файл {input_path} не найден")
        return

    df = pd.read_csv(input_path)
    df.columns = [c.upper() for c in df.columns]

    required = ["TRANSACTION_DT", "CUSTOMER_ID", "PRODUCT_ID"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(f"Не найдены нужные колонки: {missing}")

    df["TRANSACTION_DT"] = pd.to_datetime(
        df["TRANSACTION_DT"],
        errors="coerce",
    )

    df = df.dropna(subset=["TRANSACTION_DT", "CUSTOMER_ID", "PRODUCT_ID"])

    df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype(int)
    df["PRODUCT_ID"] = df["PRODUCT_ID"].astype(int)

    # Для TaFeng считаем одну дату одной корзиной.
    df["BASKET_ID"] = (
        df["CUSTOMER_ID"].astype(str)
        + "_"
        + df["TRANSACTION_DT"].dt.strftime("%Y-%m-%d")
    )

    print(
        f"Raw: rows={len(df)}, "
        f"users={df['CUSTOMER_ID'].nunique()}, "
        f"items={df['PRODUCT_ID'].nunique()}, "
        f"baskets={df['BASKET_ID'].nunique()}"
    )

    df = iterative_k_core(
        df=df,
        user_col="CUSTOMER_ID",
        item_col="PRODUCT_ID",
        basket_col="BASKET_ID",
        min_user_baskets=min_user_baskets,
        min_item_interactions=min_item_interactions,
    )

    user_hist = build_user_hist_from_df(
        df=df,
        user_col="CUSTOMER_ID",
        item_col="PRODUCT_ID",
        time_col="TRANSACTION_DT",
        basket_col="BASKET_ID",
    )

    save_processed(user_hist, out_path)


def prepare_dunnhumby(
    data_dir="data/",
    out_path="data/processed/dunnhumby_preprocessed.pkl",
    min_user_baskets=5,
    min_item_interactions=5,
):
    print("\n=== Preparing Dunnhumby ===")

    trans_path = os.path.join(data_dir, "transaction_data.csv")

    if not os.path.exists(trans_path):
        print(f"Не найден {trans_path}")
        return

    trans = pd.read_csv(trans_path)
    trans.columns = [c.lower() for c in trans.columns]

    required = ["household_key", "product_id"]
    missing = [c for c in required if c not in trans.columns]

    if missing:
        raise KeyError(f"Не найдены нужные колонки: {missing}")

    if "day" in trans.columns:
        trans["date"] = pd.to_datetime(
            trans["day"],
            errors="coerce",
            origin="1900-01-01",
            unit="D",
        )
    elif "date" in trans.columns:
        trans["date"] = pd.to_datetime(trans["date"], errors="coerce")
    else:
        raise KeyError("В Dunnhumby нет колонки day или date")

    trans = trans.dropna(subset=["household_key", "product_id", "date"])

    trans["household_key"] = trans["household_key"].astype(int)
    trans["product_id"] = trans["product_id"].astype(int)

    if "basket_id" in trans.columns:
        trans["BASKET_ID"] = trans["basket_id"].astype(str)
    elif "transaction_id" in trans.columns:
        trans["BASKET_ID"] = trans["transaction_id"].astype(str)
    else:
        trans["BASKET_ID"] = (
            trans["household_key"].astype(str)
            + "_"
            + trans["date"].dt.strftime("%Y-%m-%d")
        )

    print(
        f"Raw: rows={len(trans)}, "
        f"users={trans['household_key'].nunique()}, "
        f"items={trans['product_id'].nunique()}, "
        f"baskets={trans['BASKET_ID'].nunique()}"
    )

    trans = iterative_k_core(
        df=trans,
        user_col="household_key",
        item_col="product_id",
        basket_col="BASKET_ID",
        min_user_baskets=min_user_baskets,
        min_item_interactions=min_item_interactions,
    )

    user_hist = build_user_hist_from_df(
        df=trans,
        user_col="household_key",
        item_col="product_id",
        time_col="date",
        basket_col="BASKET_ID",
    )

    save_processed(user_hist, out_path)


if __name__ == "__main__":
    prepare_tafeng(
        min_user_baskets=5,
        min_item_interactions=5,
    )

    prepare_dunnhumby(
        min_user_baskets=5,
        min_item_interactions=5,
    )