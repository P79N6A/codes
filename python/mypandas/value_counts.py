# coding: utf-8
import pandas as pd


def value_cnt2str(df, field):
    df_field = pd.DataFrame(df[field].value_counts()).reset_index()
    df_field.columns = [field, 'count']
    df_field_str = df_field.to_string(index=False)
    return df_field_str



if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 1, 2, 2, 2]})
    print (value_cnt2str(df, 'a'))
