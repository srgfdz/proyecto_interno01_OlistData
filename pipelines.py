import pandas as pd
from pandas.api import types as pdt
import operator

#Funciones para limpiar y transformar df de pandas indicando por parámetros lo que queremos hacer...

def clean_field_names_pandas(df: pd.DataFrame, case: str = "lower") -> pd.DataFrame:
    """
    Renombra columnas:
      - strip + collapse espacios
      - espacios → "_"
      - quita tildes
      - pasa a lower/upper según case
    """
    cols = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
          .str.replace(" ", "_")
          .str.normalize("NFKD")
          .str.encode("ascii", "ignore").str.decode("ascii")
    )
    cols = cols.str.upper() if case.lower() == "upper" else cols.str.lower()
    df = df.copy()
    df.columns = cols
    return df

def _clean_str_series(ser: pd.Series, case: str) -> pd.Series:
    """
    Limpia una Serie de texto:
      - strip + collapse espacios
      - quita tildes
      - pasa a lower/upper según case
    Usa StringDtype para preservar nulos.
    """
    s = ser.astype("string")
    s = (
        s.str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.normalize("NFKD")
         .str.encode("ascii", "ignore").str.decode("ascii")
    )
    s = s.str.upper() if case.lower() == "upper" else s.str.lower()
    return s

def clean_text_values_pandas(df: pd.DataFrame, case: str = "lower") -> pd.DataFrame:
    """
    Normaliza valores de texto en columnas object, string o category.
    """
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype

        if isinstance(dtype, pd.CategoricalDtype):
            # limpia categorías y valores
            ser = df[col]
            cats = pd.Series(ser.cat.categories, dtype="string")
            cats_clean = _clean_str_series(cats, case)
            ser_new = ser.cat.set_categories(cats_clean).astype("string")
            ser_new = _clean_str_series(ser_new, case).astype("category")

        elif pdt.is_object_dtype(dtype) or pdt.is_string_dtype(dtype):
            ser_new = _clean_str_series(df[col], case)

        else:
            continue

        df[col] = ser_new
    return df

def clean_datetime_columns_pandas(
    df: pd.DataFrame,
    datetime_params: dict = None
) -> pd.DataFrame:
    """
    Convierte a datetime SOLO las columnas listadas en datetime_params:
      datetime_params = { col_name: format_str, ... }
    No hay inferencia automática.
    """
    df = df.copy()
    datetime_params = datetime_params or {}

    for col, fmt in datetime_params.items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
            print(f"*** Columna {col!r}: convertida con formato {fmt} ***")
    return df

def fill_nulls(df: pd.DataFrame, fill_map: dict) -> pd.DataFrame:
    """
    Rellena nulos según fill_map = { col_name: fill_value, ... }
    """
    df = df.copy()
    for col, val in fill_map.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].fillna(val)
            after = df[col].isna().sum()
            print(f"*** Columna {col!r} con {before} nulls → rellenado con {val!r} ({after} nulls actualmente) *** ")
    return df



def replace_invalid_numeric_values(df: pd.DataFrame, numeric_check_conf: dict) -> pd.DataFrame:
    df = df.copy()

    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }

    fields = numeric_check_conf.get("fields_name", [])
    op_str = numeric_check_conf.get("op", "<=")
    threshold = numeric_check_conf.get("value", 0)
    method = numeric_check_conf.get("method", "mean").lower()

    if op_str not in ops:
        print(f"***  Operador {op_str!r} no soportado. ***")
        return df

    op_func = ops[op_str]

    for col in fields:
        if col not in df.columns:
            print(f"*** Columna {col!r} no existe en el DataFrame. ***")
            continue

        mask = op_func(df[col], threshold)
        count = mask.sum()

        if count == 0:
            print(f"*** Columna {col!r}: ningún valor cumple la condición '{col} {op_str} {threshold}' — no se reemplaza nada. ***")
            continue

        if method == "mean":
            replacement = df.loc[~mask, col].mean()
        elif method == "median":
            replacement = df.loc[~mask, col].median()
        elif method == "mode":
            mode = df.loc[~mask, col].mode()
            replacement = mode[0] if not mode.empty else None
        else:
            print(f"*** Método {method!r} no soportado. Se omite columna {col!r}. ***")
            continue

        print(f"*** Columna {col!r}: {count} valores reemplazados con {method} ({replacement:.2f}) usando condición '{col} {op_str} {threshold}'***")
        df.loc[mask, col] = replacement

    return df


def clean_pandas_df(
    df: pd.DataFrame,
    fields_name_params: dict,
    values_params: dict,
) -> pd.DataFrame:
    """
    Pipeline:
      1) Renombra columnas (fields_name_params['case'])
      2) Convierte fechas (fields_name_params['datetime_fields'])
      3) Rellena nulos (values_params['change_nulls_maps'])
      4) Normaliza texto (values_params['case'])
    """
    # 1) Renombrar columnas
    case_names = fields_name_params.get("case", "lower")
    df2 = clean_field_names_pandas(df, case_names)

    # 2) Convertir datetime
    datetime_fields = fields_name_params.get("datetime_fields", {})
    if datetime_fields:
        df2 = clean_datetime_columns_pandas(df2, datetime_fields)

    # 3) Rellenar nulos
    null_fill_map = values_params.get("change_nulls_maps", {})
    if null_fill_map:
        df2 = fill_nulls(df2, null_fill_map)

    # 4) Normalizar texto
    case_values = values_params.get("case", "lower")
    df2 = clean_text_values_pandas(df2, case_values)


    #5) Validación datos numéricos
    numeric_check_params = values_params.get("numeric_checks", {})
    if numeric_check_params:
        df2 = replace_invalid_numeric_values(df2, numeric_check_params)

    return df2


