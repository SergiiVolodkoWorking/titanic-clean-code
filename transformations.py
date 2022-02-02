def add_title_from_name(df):
    # The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature.
    # The `expand=False` flag returns a DataFrame.
    title_regex = r' ([A-Za-z]+)\.'
    df['Title'] = df.Name.str.extract(title_regex, expand=False)
    return df
