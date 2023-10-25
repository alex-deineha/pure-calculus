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
        # if the action ends in specified time, the timer is canceled
        timer.cancel()

from calculus_path_mod.json_serialization import load_terms

# from_inx, to_inx = 0, 3
# from_inx, to_inx = 3, 6
# from_inx, to_inx = 6, 9
# from_inx, to_inx = 9, 12
# from_inx, to_inx = 12, 15
from_inx, to_inx = 15, 18
# from_inx, to_inx = 18, 20
#
print(from_inx, "-", to_inx)
lists_terms_LO = [load_terms(f"../../tests_11_retests/collected_terms/terms_210_filtered_LO_{inx_}.dat") for inx_ in range(from_inx, to_inx)]

# lists_terms_LO = [
#     load_terms(f"../../tests_11_retests/terms_210_filtered_LO.dat"),
#     load_terms(f"../../tests_11_retests/terms_210_filtered_RI.dat"),
# ]

def gen_norm_data(terms_list, strategies_list):
    steps_strategies_ = []
    simple_terms_ = []

    for term in tqdm(terms_list):
        norm_term = term
        term_red_step = 0
        while norm_term:
            list_steps_strategy = []
            term_red_step += 1
            simple_terms_.append(norm_term.simple_str())
            for strategy in strategies_list:
                try:
                    with time_limit(10, f"can't normalize this {term} term"):
                        _, steps = norm_term.normalize(strategy)
                        list_steps_strategy.append(steps if steps < 1_000 else 1_000)
                except TimeoutException:
                    list_steps_strategy.append(1_000)
            steps_strategies_.append(list_steps_strategy)

            (step_term, _, _), norm_term = norm_term.one_step_normalize_visual(LOStrategy())
            if (step_term.vertices_number > 3_000) or (term_red_step > 400):
                norm_term = None

    return simple_terms_, steps_strategies_

list_res_OO = [gen_norm_data(terms_LO, [LOStrategy(), RIStrategy()]) for terms_LO in lists_terms_LO]

steps_lo = []
steps_ri = []
simple_terms = []

for res_ in list_res_OO:
    st, ss = res_[0], res_[1]
    for inx_ in range(len(st)):
        simple_terms.append(st[inx_])
        steps_lo.append(ss[inx_][0])
        steps_ri.append(ss[inx_][1])

df = pd.DataFrame({"LO_steps_num": steps_lo, "RI_steps_num": steps_ri, "simple_terms": simple_terms})
print(len(df))
df = df.drop_duplicates(subset="simple_terms")
print(len(df))

# df.to_csv(f"./data_steps/steps_simple_term_str.csv", index=False)
df.to_csv(f"./data_steps/steps_simple_term_str_extended_v1_{from_inx}_{to_inx - 1}.csv", index=False)
