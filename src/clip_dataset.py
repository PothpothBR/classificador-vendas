import ctypes
from typing import Any, Callable
import pandas

def __base_groupBy(column: str, data: dict):
    data_g = {}
    for _Name in data[column]:
        if _Name not in data_g.keys():
            data_g.update({_Name: 1})
        else:
            data_g[_Name] += 1
    return data_g

def groupBy(column: str, data: pandas.DataFrame | list | tuple) -> dict:
    """agrupa e conta a coluna <column> da tabela <data>

    Args:
        column (str): nome da coluna
        data (pandas.DataFrame | list | tuple): a tabela a ser usada (a mesma não é alterada)

    Returns:
        dict: um dicionario com o nome das colunas como chave e seu respectivo numero de incidências
    """
    if type(data) in (list, tuple):
        data_g = {}
        for pack in data:
            data_s = __base_groupBy(column, pack)
            for val in data_s:
                if data_g.get(val, -1) != -1:
                    data_s[val] += data_g[val]
            data_g.update(data_s)
        return data_g
    return __base_groupBy(column, data)
    
def limitTo(size: int, data: dict) -> dict:
    """corta um dicionario com os primeiros <size> valores

    Args:
        size (int): quantidade de valores a serem retornados
        data (dict): dicionario a ser usado

    Returns:
        dict: dicionario de valores cortado
    """
    data_l = {}
    i = 0
    for key, value in data.items():
        if i >= size: break
        data_l.update({key: value})
        i += 1
    return data_l

def orderBy(ac_dc: str, data: dict) -> dict:
    """ordena um dicionario pelo valor

    Args:
        ac_dc (str["asc" |  "desc"]): se a ordenação sera ascendente (asc) ou decrescente (desc)
        data (dict): dicionaro de valores

    Returns:
        dict: dicionario ordenado
    """
    return dict(sorted(
        data.items(),
        key=lambda a: a[1],
        reverse = True if ac_dc.lower() == "desc" else False
    ))
    
def __base_saveThe(save_names, ref_c, data):
    for i, name in zip(data.index, data[ref_c]):
        if name not in save_names:
            data.drop(i, axis=0, inplace=True)
    return data

def getOnly(save_names: str | list | tuple, ref_c: str, data: pandas.DataFrame | list | tuple) -> pandas.DataFrame:
    """filtra os valores de uma tabela, se o valor da coluna <ref_c> NÃO existir em <save_names>, essa linha(valor) sera removida

    Args:
        save_names (str | list | tuple): nomes a serem mantidos
        ref_c (str): coluna para realizar a busca
        data (pandas.DataFrame | list | tuple): tabela para realizar a busca

    Returns:
        pandas.DataFrame: a tabela alterada
    """
    if type(save_names) not in (list, tuple, type(dict().keys())): save_names = [save_names]
    if type(data) in (list, tuple):
        data_c = __base_saveThe(save_names, ref_c, data[0])
        for _Data in data[1:]:
            data_c = pandas.concat([data_c, __base_saveThe(save_names, ref_c, _Data)], ignore_index=True,)
        return data_c
    else:
        return __base_saveThe(save_names, ref_c, data)
    

def dropThese(columns: str | list | tuple, data: pandas.DataFrame) -> pandas.DataFrame:
    """remove as colunas <columns> da tabela <data>

    Args:
        columns (str | list | tuple): colunas a serem removidas
        data (pandas.DataFrame): tabela a ser alterada

    Returns:
        pandas.DataFrame: a tabela alterada
    """
    if type(columns) not in (list, tuple): columns = [columns]
    data.drop(columns, axis=1, inplace=True)
    return data

def changeTo(lamb: Callable, column: str, data: pandas.DataFrame) -> pandas.DataFrame:
    """recebe uma função <lamb> a mesma é aplicada para cada valor da linha, unicamente para a coluna <column> alterando o mesmo.

    Args:
        lamb (Callable): função para alterar o dado tem o protótipo: func(valor) -> valor_alterado
        column (str): coluna a ser realizado a alteração
        data (pandas.DataFrame): tabela a ser alterada 

    Returns:
        pandas.DataFrame: a tabela alterada
    """
    data[column] = data[column].map(lamb)
    return data

def shuffle(data: pandas.DataFrame):
    """embaralha os dados da tabela

    Args:
        data (pandas.DataFrame): tabela a ser alterada

    Returns:
        _type_: a tabela alterada
    """
    for _ in range(7):
        data = data.sample(frac=1).reset_index(drop=True)
    return data
    
def clipDataset() -> pandas.DataFrame:
    """padroniza o dataset para utilizar

    Args:
        data (pandas.DataFrame): data frame

    Returns:
        pandas.DataFrame: data frame clipado
    """
    data1 = pandas.read_csv("./res/PS4_gamesSales.csv")
    data2 = pandas.read_csv("./res/Video_Games_Sales_as_at_22_Dec_2016.csv")
    data3 = pandas.read_csv("./res/XboxOne_GameSales.csv")
    clss = {'Action': 0, 'Sports': 1}
    
    data = \
    shuffle(
        changeTo(lambda a: clss[a], "Genre",
            changeTo(lambda a: a/2022, "Year",
                dropThese(["Index", "Name", "Platform", "Game", "Publisher", "Critic_Score","Critic_Count","User_Score","User_Count","Developer","Rating"],
                    getOnly(
                        limitTo(2, 
                            orderBy("desc",
                                groupBy("Genre",
                                        (data1, data2, data3)
                                )
                            )
                        ).keys(), "Genre", (data1, data2, data3)
                    )
                )
            )
        )
    )

    # salva o novo dataset
    data.fillna(0).to_csv("./res/clipped.csv")
    return data

def clipDataset2() -> pandas.DataFrame:
    """padroniza o dataset para utilizar

    Args:
        data (pandas.DataFrame): data frame

    Returns:
        pandas.DataFrame: data frame clipado
    """
    data1 = pandas.read_csv("./res/Jan_2019_ontime.csv")
    
    clss = {'ATL': 0, 'ORD': 1}
    
    data = \
    shuffle(
        changeTo(lambda a: clss[a], "ORIGIN",
                dropThese(["OP_CARRIER","TAIL_NUM"],
                    getOnly(
                        limitTo(2, 
                            orderBy("desc",
                                groupBy("ORIGIN",
                                        data1
                                )
                            )
                        ).keys(), "ORIGIN", data1
                    )
                )
        )
    )

    # salva o novo dataset
    data.fillna(0).to_csv("./res/clipped2.csv")

if __name__ == "__main__":
    clipDataset2()