import sys
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import Counter

sys.path.append("../../.")
from calculus_path_mod.term_engine import *
from calculus_path_mod.reduction_strategy import *
from calculus_path_mod.terms import num_comparison, nat_numbers, arithm_ops, combinators, pairs, logic
from calculus_path_mod.terms.pseudonym import *

from contextlib import contextmanager
import threading
import _thread


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

from calculus_path_mod.json_serialization import load_terms

# from_inx, to_inx = 0, 3
# from_inx, to_inx = 3, 6
# from_inx, to_inx = 6, 9
# from_inx, to_inx = 9, 12
# from_inx, to_inx = 12, 15
# from_inx, to_inx = 15, 18
from_inx, to_inx = 18, 20

print(from_inx, "-", to_inx)

lists_terms_LO = [load_terms(f"../../tests_11_retests/collected_terms/terms_210_filtered_LO_{inx_}.dat") for inx_ in range(from_inx, to_inx)]

def gen_norm_data(terms_list, strategy):
    normalized_terms_dict = dict()
    for inx_, term in tqdm(list(enumerate(terms_list))):
        try:
            with time_limit(120, f"can't normalize this {inx_} term"):
                term_name = term.simple_str()
                normalized_terms_dict[term_name] = []
                term_red_steps = 0
                (step_term, _, _), norm_term = term.one_step_normalize_visual(strategy)
                normalized_terms_dict[term_name].append(step_term.simple_str())

                while norm_term:
                    normalized_terms_dict[term_name].append(norm_term.simple_str())
                    (step_term, _, _), norm_term = norm_term.one_step_normalize_visual(strategy)

                    # computation limitation
                    if (step_term.vertices_number > 3_000) or (term_red_steps > 400):
                    # if (step_term.vertices_number > 1_500) or (term_red_steps > 200):
                        norm_term = None
        except TimeoutException as te_:
            print(te_.msg)
    return normalized_terms_dict

list_res_OO = [gen_norm_data(terms_LO, RIStrategy()) for terms_LO in lists_terms_LO]

steps_lo = []
simple_terms = []

for res_ in list_res_OO:
    for key_ in res_.keys():
        list_red_steps = res_[key_]
        total_steps = len(list_red_steps) - 1
        for inx_ in range(total_steps + 1):
            if list_red_steps[inx_] not in simple_terms:
                simple_terms.append(list_red_steps[inx_])
                steps_lo.append(total_steps - inx_)

df = pd.DataFrame({"steps_num": steps_lo, "simple_terms": simple_terms})
print(len(df))
df = df.drop_duplicates(subset="simple_terms")
print(len(df))

len(set(df["simple_terms"]))

df.to_csv(f"./data_RI/steps_simple_term_str_extended_v1_{from_inx}_{to_inx - 1}.csv", index=False)
