import json
from json import JSONEncoder
from typing import Any

from calculus_path_mod.term_engine import Var, Term, Atom, Application as App, Abstraction as Lambda


def term_to_dict(term: Term, unique_vars: dict = None) -> dict:
    """
    :param term: tree representation of term via Term class
    :param unique_vars: dict, where key is _data value from Vars in terms and value is a unique index
    :return: dict representation of a term
    """

    if unique_vars is None:
        unique_vars = set(term._get_list_variables())
        unique_vars = {key: val for val, key in enumerate(unique_vars)}
    if term.kind == "atom":
        return {"kind": "atom", "var": unique_vars[term._data._data]}
    if term.kind == "application":
        return {"kind": "app",
                "subj": term_to_dict(term._data[0], unique_vars),
                "obj": term_to_dict(term._data[1], unique_vars)}
    # term.kind == "abstraction"
    return {"kind": "lambda",
            "var": unique_vars[term._data[0]._data],
            "body": term_to_dict(term._data[1], unique_vars)}


def dict_to_term(term_dict: dict, fresh_vars: dict = dict()) -> Term:
    """
    :param term_dict: dict representation of a term
    :param fresh_vars: dict, where key is a unique index, and value is a new Var object
    :return: a tree representation of term in a Term object
    """

    if term_dict["kind"] == "atom":
        if term_dict["var"] not in fresh_vars.keys():
            fresh_vars[term_dict["var"]] = Var()
        return Atom(fresh_vars[term_dict["var"]])
    if term_dict["kind"] == "app":
        return App(dict_to_term(term_dict["subj"]), dict_to_term(term_dict["obj"]))
    if term_dict["kind"] == "lambda":
        if term_dict["var"] not in fresh_vars.keys():
            fresh_vars[term_dict["var"]] = Var()
        return Lambda(fresh_vars[term_dict["var"]], dict_to_term(term_dict["body"]))
    return None


def save_terms(file_name: str, list_terms: list, is_overwrite=True):
    """
    :param file_name: path to file for saving terms
    :param list_terms: list of Term objects
    :param is_overwrite: if it's 'False' - add records to the file, and rewrite file otherwise
    """

    with open(file_name, "w" if is_overwrite else "a") as storage_file:
        for term in list_terms:
            storage_file.write(json.dumps(term_to_dict(term)) + "\n")


def load_terms(file_name: str) -> list:
    """
    :param file_name: path to file for loading terms
    :return: list of Term objects
    """

    list_terms = []
    with open(file_name, "r") as storage_file:
        for term_line in storage_file:
            list_terms.append(dict_to_term(json.loads(term_line)))
    return list_terms
