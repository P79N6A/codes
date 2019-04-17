# coding: utf-8
import pandas as pd


def value_cnt2str(df, field):
    df_field = pd.DataFrame(df[field].value_counts()).reset_index()
    df_field.columns = [field, 'count']
    df_field_str = df_field.to_string(index=False)
    return df_field_str


def get_dist(df, field, prefix='count|ratio'):
    df = pd.DataFrame(df[field].value_counts())
    df.columns = [field]
    count = sum(df[field])
    df[prefix] = df[field].map(lambda x: str(x) + '|' + str(round((x * 1.0 / count), 4)))
    df[field] = df.index
    df_sum = pd.DataFrame([['总计', count]])
    df_sum.columns = df.columns
    df.sort_values(field, inplace=True)
    df_sum.index = ['总计']
    df = df.append(df_sum)
    #     print df
    return df


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 1, 2, 2, 2]})
    print(value_cnt2str(df, 'a'))
