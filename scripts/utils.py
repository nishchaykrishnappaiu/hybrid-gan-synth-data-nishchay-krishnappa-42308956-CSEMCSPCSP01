import pandas as pd



def detect_column_types(df:pd.DataFrame,categorical_thershold:int=20):

    categorical,continuous = [],[]
    for col in df.columns:
        s = df[col]
        if s.dtype == "O" or str(s.dtype).startswith("category"):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            nunique = s.nunique(dropna=True)
            if nunique <= categorical_thershold:
                categorical.append(col)
            else:
                continuous.append(col)
        else:
            categorical.append(col)
    return categorical, continuous