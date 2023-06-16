import sys
import re

sys.path.append("../")
from calculus_path_mod.term_engine import *
from calculus_path_mod.reduction_strategy import *
from calculus_path_mod.terms import num_comparison, nat_numbers, arithm_ops, combinators, pairs, logic

from calculus_path_mod.terms.pseudonym import *

LAMBDA_COMMANDS_DICT = {
    # logic
    "TRUE": "logic.true_term()",
    "FALSE": "logic.false_term()",
    "IF": "logic.ite_term()",
    "ITE": "logic.ite_term()",
    "NOT": "logic.not_term()",
    "AND": "logic.and_term()",
    "OR": "logic.or_term()",

    # combinators
    "K": "combinators.k_term()",
    "K_STAR": "combinators.k_star_term()",
    "S": "combinators.s_term()",
    "I": "combinators.i_term()",
    "Y": "combinators.y_term()",
    "Z": "combinators.z_term()",
    "ETTA_V": "combinators.etta_v_term()",

    # nat numbers
    "NUM": "nat_numbers.num_term(",

    # num comparison
    "ISZERO": "num_comparison.iszero_term()",
    "LEQ": "num_comparison.leq_term()",
    "EQ": "num_comparison.eq_term()",
    "LT": "num_comparison.lt_term()",
    "NEQ": "num_comparison.neq_term()",
    "GEQ": "num_comparison.geq_term()",
    "GT": "num_comparison.gt_term()",

    # pairs
    "PAIR": "pairs.pair_term()",
    "FIRST": "pairs.first_term()",
    "SECOND": "pairs.second_term()",

    # arithm ops
    "SUCC": "arithm_ops.succ_term()",
    "SINC": "arithm_ops.sinc_term()",
    "PRED": "arithm_ops.pred_term()",
    "SUBTRACT": "arithm_ops.subtract_term()",
    "MINUS": "arithm_ops.subtract_term()",
    "DIV": "arithm_ops.div_term()",
    "MOD": "arithm_ops.mod_term()",
    "IDIV": "arithm_ops.idiv_term()",
    "PLUS": "arithm_ops.plus_term()",
    "SUM": "arithm_ops.plus_term()",
    "MULT": "arithm_ops.mult_term()",
}

LAMBDA_COMMANDS_DESCRIPT_DICT = {
    "NOT": "(λa. (ITE a FALSE TRUE))",
    "AND": "(λa. (λb. (ITE a b a)))",
    "OR": "(λa. (λb. (ITE a a b)))",

    "ISZERO": "(λn. (n (λx. FALSE) TRUE))",
    "LEQ": "(λn. (λm. (ISZERO (SUBTRACT n m))))",
    "EQ": "(λn. (λm. (AND (LE n m) (LE m n)) ))",
    "LT": "(λa. (λb. (NOT (LEQ b a)) ))",
    "NEQ": "(λa. (λb. (OR (NOT (LEQ a b)) (NOT (LEQ b a))) ))",
    "GEQ": "(λa. (λb. (LEQ b a) ))",
    "GT": "(λa. (λb. (NOT (LEQ a b)) ))",

    "PAIR": "(𝜆x. (𝜆y. (𝜆p. (p x y)) ))",

    "SUCC": "(𝜆n. (𝜆x. (𝜆y. (x (n x y)) )))",
    "SINC": "(𝜆p. (PAIR (SECOND p) (SUCC (SECOND p))) )",
    "PRED": "(𝜆n. (FIRST (NUM[n] SINC (PAIR NUM[0] NUM[0]))) )",
    "SUBTRACT": "(𝜆n. (𝜆m. (m PRED n) ))",
    "MINUS": "(𝜆n. (𝜆m. (m PRED n) ))",
    "DIV": "(Y (λgqab. (LT a b (PAIR q a) (g (SUCC q) (SUB a b) b) ) NUM[0]) )",
    "MOD": "(λab. (SECOND (DIV a b)) )",
    "IDIV": "(λab. (FIRST (DIV a b)) )",
    "PLUS": "(λmn. (n SUCC m) )",
    "SUM": "(λmn. (n SUCC m) )",
    "MULT": "(λmn. (m (PLUS n) NUM[0]) )",
}

REDUCTION_STRATEGIES_DICT = {
    "LO": [LOStrategy(), "Left Most Outer Most Strategy"],
    "LI": [LIStrategy(), "Left Most Inner Most Strategy"],
}

HELP_STR = """#help -- to call this menu

#show-syntax -- show syntax used for typing lambda terms
#show-strategies -- show available strategies
#show-lib -- show all available terms in the terms library
#show-all -- show all terms defined by user
#show TERM_NAME -- to get description by the term
#show term_name -- to show a defined term by term name
#show-full TERM_NAME -- to get a full term definition by the term nem
#show-full term_name -- to show a full term definition of a defined term by term name

#import /path_to_lib/lib_file.lmd -- for including terms from other file. 
                It must have a name in which will be defined a term
#define term_name = term_definition # -- for defining term in the memory,
                MUST ENDS on '#' symbol for allowing multiline input 
#reduce term_name -- reduce the term by term_name by LeftOuter str, 
                reduced term will appear in term_name_red
#reduce term_name STRATEGY_NAME -- reduce term by term_name with defined strategy
                reduced term will appear in term_name_red_strategy_name
#reduce-no-lim term_name -- reduce the term by term_name by LeftOuter str, 
                reduced term will appear in term_name_red
#reduce-no-lim term_name STRATEGY_NAME -- reduce term by term_name with defined strategy
                reduced term will appear in term_name_red_strategy_name"""

SYNTAX_HELP_STR = """Lamda Calculus syntax rules:

* 'x' -- means atom term
* '(@var_name. body_term)' -- abstract term, between @ and . symbols defined variable
* (object_term subject_term) -- application term
* (term_1 term_2 term_3 term_n) -- short version of (((term_1 term_2) term_3) term_n)
* instead term you can use predefined terms in library or your own
"""

SHOW_LIB_STR = f"""** LOGIC TERMS **
TRUE == {logic.true_term().funky_str()}
FALSE == {logic.false_term().funky_str()}
IF == {logic.ite_term().funky_str()}
ITE == {logic.ite_term().funky_str()}
NOT == {LAMBDA_COMMANDS_DESCRIPT_DICT["NOT"]}
AND == {LAMBDA_COMMANDS_DESCRIPT_DICT["AND"]}
OR == {LAMBDA_COMMANDS_DESCRIPT_DICT["OR"]}

** COMBINATOR TERMS **
K == {combinators.k_term().funky_str()}
K_STAR == {combinators.k_star_term().funky_str()}
S == {combinators.s_term().funky_str()}
I == {combinators.i_term().funky_str()}
Y == {combinators.y_term().funky_str()}
Z == {combinators.z_term().funky_str()}
ETTA_V == {combinators.etta_v_term().funky_str()}

** NUMBERS **
NUM[0] == {nat_numbers.num_term(0).funky_str()}
NUM[1] == {nat_numbers.num_term(1).funky_str()}
NUM[2] == {nat_numbers.num_term(2).funky_str()}
NUM[3] == {nat_numbers.num_term(3).funky_str()}

** NUMBER COMPARISONS **
ISZERO == {LAMBDA_COMMANDS_DESCRIPT_DICT["ISZERO"]}
LEQ == {LAMBDA_COMMANDS_DESCRIPT_DICT["LEQ"]}
EQ == {LAMBDA_COMMANDS_DESCRIPT_DICT["EQ"]}
LT == {LAMBDA_COMMANDS_DESCRIPT_DICT["LT"]}
NEQ == {LAMBDA_COMMANDS_DESCRIPT_DICT["NEQ"]}
GEQ == {LAMBDA_COMMANDS_DESCRIPT_DICT["GEQ"]}
GT == {LAMBDA_COMMANDS_DESCRIPT_DICT["GT"]}

** PAIRS **
PAIR == {LAMBDA_COMMANDS_DESCRIPT_DICT["PAIR"]}
FIRST == {pairs.first_term().funky_str()}
SECOND == {pairs.second_term().funky_str()}

** ARITHMETIC OPERATIONS **
SUCC == {LAMBDA_COMMANDS_DESCRIPT_DICT["SUCC"]}
SINC == {LAMBDA_COMMANDS_DESCRIPT_DICT["SINC"]}
PRED == {LAMBDA_COMMANDS_DESCRIPT_DICT["PRED"]}
SUBTRACT == {LAMBDA_COMMANDS_DESCRIPT_DICT["SUBTRACT"]}
MINUS == {LAMBDA_COMMANDS_DESCRIPT_DICT["MINUS"]}
DIV == {LAMBDA_COMMANDS_DESCRIPT_DICT["DIV"]}
MOD == {LAMBDA_COMMANDS_DESCRIPT_DICT["MOD"]}
IDIV == {LAMBDA_COMMANDS_DESCRIPT_DICT["IDIV"]}
PLUS == {LAMBDA_COMMANDS_DESCRIPT_DICT["PLUS"]}
SUM == {LAMBDA_COMMANDS_DESCRIPT_DICT["SUM"]}
MULT == {LAMBDA_COMMANDS_DESCRIPT_DICT["MULT"]}
"""


def tokenize_term(lambda_code) -> list:
    brackets_counter = 0
    is_not_tokenized = True

    is_not_space_delimited = True

    lambda_code_tokenized = ""
    for symbol in lambda_code:
        if symbol == "(":
            brackets_counter += 1
        if symbol == ")":
            brackets_counter -= 1
        if brackets_counter == 0:
            is_not_tokenized = True
        if brackets_counter == 1:
            if is_not_tokenized:
                lambda_code_tokenized += "><"
                is_not_tokenized = False
        if brackets_counter == 0:
            if is_not_space_delimited:
                if symbol == " ":
                    lambda_code_tokenized += "><"
                    is_not_space_delimited = False
            else:
                if symbol != " ":
                    is_not_space_delimited = True

        lambda_code_tokenized += symbol

    lambda_code_tokenized = [token.strip() for token in lambda_code_tokenized.split("><")]
    lambda_code_tokenized = [token for token in lambda_code_tokenized if token != ""]

    return lambda_code_tokenized


def process_tokens_to_pt(lambda_code: str, vars_list: list, local_term_defs_list: list = None) -> str:
    # is abstraction
    if re.match(r"\s*\(\s*@\s*[a-zA-Z0-9_-].\s*", lambda_code):
        inx_open = 0
        inx_close = -1

        # remove outer brackets
        while lambda_code[inx_open] != "(":
            inx_open += 1
        while lambda_code[inx_close] != ")":
            inx_close -= 1
        lambda_code = lambda_code[inx_open + 1: inx_close]

        inx_open = 0
        while lambda_code[inx_open] != ".":
            inx_open += 1

        var_name = lambda_code[:inx_open]
        lambda_code = lambda_code[inx_open + 1:]

        for var in vars_list:
            if var in var_name:
                var_name = var
                break

        return f"Lambda({var_name}, {process_tokens_to_pt(lambda_code, vars_list, local_term_defs_list)})"
    else:  # it is an app or a single term
        if ("(" in lambda_code) and (")" in lambda_code):  # it is an app
            inx_open = 0
            inx_close = -1

            # remove outer brackets
            while lambda_code[inx_open] != "(":
                inx_open += 1
            while lambda_code[inx_close] != ")":
                inx_close -= 1
            lambda_code = lambda_code[inx_open + 1: inx_close]

            tokens = tokenize_term(lambda_code)
            if len(tokens) == 0:
                raise Exception("Something went wrong")
            elif len(tokens) == 1:
                return process_tokens_to_pt(tokens[0], vars_list, local_term_defs_list)
            elif len(tokens) == 2:
                return f"App({process_tokens_to_pt(tokens[0], vars_list, local_term_defs_list)}, {process_tokens_to_pt(tokens[1], vars_list, local_term_defs_list)})"
            else:
                result_line = "multi_app_term("
                for token in tokens:
                    result_line += str(process_tokens_to_pt(token, vars_list, local_term_defs_list)) + ", "
                result_line += ")"
                return result_line
        else:  # it is a single term
            lambda_code = lambda_code.strip()
            if lambda_code in LAMBDA_COMMANDS_DICT.keys():
                return LAMBDA_COMMANDS_DICT[lambda_code]
            elif local_term_defs_list and (lambda_code in local_term_defs_list):
                return f">>{lambda_code}<<"
            else:
                if ("." in lambda_code) \
                        or ("@" in lambda_code):
                    raise Exception("Not allowed symbol in Lambda term")
                if ("NUM" in lambda_code) and ("[" in lambda_code) and ("]" in lambda_code):
                    try:
                        num = int(lambda_code.strip().split("[")[1][:-1])
                        return LAMBDA_COMMANDS_DICT["NUM"] + str(num) + ")"
                    except:
                        raise Exception("Can't parse number")
                return lambda_code + "_"
            pass


def convert_to_python_code(lambda_code: str, local_term_defs_list: list = None) -> str:
    # remove "\n", "\t", and outer spaces symbol
    lambda_code = lambda_code.replace("\n", "").replace("\t", " ").strip()
    if lambda_code == "":
        raise Exception("Can't convert to lambda term an empty value")

    # check brackets
    count_open_brackets = lambda_code.count("(")
    count_close_brackets = lambda_code.count(")")
    if count_open_brackets != count_close_brackets:
        raise Exception(
            f"Wrong count brackets, can't interpreter this cause '(' = {count_open_brackets}, ')' = {count_close_brackets}")

    # remove redundant spaces
    lambda_code = re.sub(r"\s+", " ", lambda_code)

    # find variables & atom terms
    vars_atoms_list = lambda_code.replace("(", " ").replace(")", " ").replace(".", " ").replace("@", " ")
    vars_atoms_list = re.sub(r"\s+", " ", vars_atoms_list)
    vars_atoms_list = vars_atoms_list.split()
    vars_atoms_list = [va_name for va_name in vars_atoms_list if va_name not in LAMBDA_COMMANDS_DICT.keys()]
    vars_atoms_list = [va_name for va_name in vars_atoms_list if not re.match(r"NUM\[[0-9]*]", va_name)]
    if local_term_defs_list:
        vars_atoms_list = [va_name for va_name in vars_atoms_list if va_name not in local_term_defs_list]
    vars_atoms_list = list(set(vars_atoms_list))

    result_line = "def gen_term():\n"
    for var_name in vars_atoms_list:
        result_line += "\t" + var_name + " = Var()\n"
        result_line += "\t" + var_name + f"_ = Atom({var_name})\n"
    if vars_atoms_list:
        result_line += "\n"
    result_line += "\tresult_term = " + process_tokens_to_pt(lambda_code, vars_atoms_list, local_term_defs_list)
    result_line += "\n\treturn result_term"
    return result_line


def conv_to_term(lambda_code: str, local_term_defs_dict: dict = None) -> Term:
    # define a function for generating a term
    # using exec() and call this function to get result
    term_def_str = convert_to_python_code(
        lambda_code,
        list(local_term_defs_dict.keys()) if local_term_defs_dict else None
    )
    params_dict = None
    params_str = ""
    if ">>" in term_def_str:
        required_terms = re.findall(r">>(.*?)<<", term_def_str)
        required_terms = list(set(required_terms))
        params_dict = dict()
        for r_term in required_terms:
            params_dict[r_term] = local_term_defs_dict[r_term]._update_bound_vars()
            term_def_str = term_def_str.replace(f">>{r_term}<<", r_term)
        params_str = ", ".join(required_terms)
        term_def_str = term_def_str[:13] + params_str + term_def_str[13:]
        term_def_str = term_def_str[:]
    if params_dict:
        exec(term_def_str, globals(), params_dict)
        return eval(f"gen_term({params_str})", globals(), params_dict)
    else:
        exec(term_def_str)
        return eval("gen_term()", params_dict)


class LambdaCalculusInterpreter:
    def __init__(self):
        self.commands_buffer = ""
        self.terms_container = dict()
        self.defs_container = dict()

    def console_engine_show(self, command_str: str = None):
        result = ""
        command_str = command_str.replace("\n", " ").strip()
        if command_str == "#help":
            result = HELP_STR
        elif command_str == "#show-strategies":
            result = "Strategies: \n"
            for key_, val_ in REDUCTION_STRATEGIES_DICT.items():
                result += f"* {key_} is {val_[1]}\n"
        elif command_str == "#show-syntax":
            result = SYNTAX_HELP_STR
        elif command_str == "#show-lib":
            result = SHOW_LIB_STR
        elif command_str == "#show-all":
            if len(self.terms_container) > 0:
                for key_, val_ in self.terms_container:
                    result += f"{key_} = {val_.funky_str()}\n"
            else:
                result = "No terms in own library"
        elif "#show-full" in command_str:
            commands_list = command_str.split()
            commands_list = [comm_ for comm_ in commands_list if comm_ != ""]
            if len(commands_list) < 2:
                result = "command 'show-full' should contains after it names of terms"
            else:
                commands_list = commands_list[1:]
                result = "Full terms description:\n"
                for comm_ in commands_list:
                    if comm_ in LAMBDA_COMMANDS_DICT.keys():
                        result += f"* {comm_} == {eval(LAMBDA_COMMANDS_DICT[comm_]).funky_str()}\n"
                    elif comm_ in self.terms_container.keys():
                        result += f"* {comm_} == {self.terms_container[comm_].funky_str()}\n"
                    else:
                        result += f"{comm_} doesn't have a term definition"
        elif "#show" in command_str:
            commands_list = command_str.split()
            commands_list = [comm_ for comm_ in commands_list if comm_ != ""]
            if len(commands_list) < 2:
                result = "command 'show' should contains after it names of terms"
            else:
                commands_list = commands_list[1:]
                result = "Terms description:\n"
                for comm_ in commands_list:
                    if comm_ in LAMBDA_COMMANDS_DESCRIPT_DICT.keys():
                        result += f"* {comm_} == {LAMBDA_COMMANDS_DESCRIPT_DICT[comm_]}\n"
                    elif comm_ in LAMBDA_COMMANDS_DICT.keys():
                        result += f"* {comm_} == {LAMBDA_COMMANDS_DICT[comm_]}\n"
                    elif comm_ in self.defs_container.keys():
                        result += f"* {comm_} == {self.defs_container[comm_]}\n"
                    elif comm_ in self.terms_container.keys():
                        result += f"* {comm_} == {self.terms_container[comm_].funky_str()}\n"
                    else:
                        result += f"{comm_} doesn't have a term definition"
        else:
            result = "No command found consider '#help' for getting help how to use the lambda interpreter"
        return result

    def console_engine_import(self, command_str: str = None):
        command_list = command_str.strip().split()
        command_list = [comm_ for comm_ in command_list if comm_ != ""]
        if len(command_list) != 2:
            return "Wrong command definition"

        def_term_str = ""
        with open(command_list[1], "r") as lmd_file:
            for lmd_str in lmd_file.readlines():
                def_term_str += lmd_str

        def_term_arr = def_term_str.strip().split("=")
        def_term_arr = [dt.strip() for dt in def_term_arr]
        if len(def_term_arr) != 2:
            return "The lmd file has WRONG format. It should contains one term definition!"

        def_term_arr[0] = def_term_arr[0].strip()

        try:
            term_obj = conv_to_term(def_term_arr[1], self.terms_container)
            self.defs_container[def_term_arr[0]] = def_term_arr[1]
            self.terms_container[def_term_arr[0]] = term_obj
            return f"Imported term could be fined & reduced by name: '{def_term_arr[0]}'"
        except Exception as eee:
            return f"Can't parse lmd-code, it has wrong format, please reformat it!\nError: {eee}"

    def console_engine_define(self, command_str: str = None):
        command_str = command_str.strip().replace("#define", "").rstrip("#").strip()
        def_term_arr = command_str.strip().split("=")
        def_term_arr = [dt.strip() for dt in def_term_arr]
        if len(def_term_arr) != 2:
            return "ERROR: For defining term you should have a term name, and a term definition"

        try:
            term_obj = conv_to_term(def_term_arr[1], self.terms_container)
            self.defs_container[def_term_arr[0]] = def_term_arr[1]
            self.terms_container[def_term_arr[0]] = term_obj
            return f"Defined term could be found & reduced by name: '{def_term_arr[0]}'"
        except Exception as eee:
            return f"ERROR: Can't parse lmd-code, it has wrong format, please reformat it!\nError: {eee}"

    def console_engine_reduce(self, command_std: str = None, is_lim_reduction=True):
        commands_list = command_std.strip().split()
        commands_list = [comm_ for comm_ in commands_list if comm_ != ""]
        if len(commands_list) == 2:
            term_name = commands_list[1].strip()
            if term_name in self.terms_container.keys():
                norm_term_obj, steps = self.terms_container[term_name].normalize(
                    REDUCTION_STRATEGIES_DICT["LO"][0],
                    is_lim_reduction
                )
                self.terms_container[f"{term_name}_red"] = norm_term_obj
                return f"{term_name}_red - saved as result of reduction\n" \
                    + f"{steps} steps to normalize\n" \
                    + f"LO-norm({term_name}) = {norm_term_obj.funky_str()}\n"
            else:
                return f"Term with name {term_name} not available"
        elif len(commands_list) == 3:
            term_name = commands_list[1].strip()
            strategy_name = commands_list[2].strip()
            if (term_name in self.terms_container.keys()) \
                    and (strategy_name in REDUCTION_STRATEGIES_DICT.keys()):
                norm_term_obj, steps = self.terms_container[term_name].normalize(
                    REDUCTION_STRATEGIES_DICT[strategy_name][0],
                    is_lim_reduction
                )
                self.terms_container[f"{term_name}_red_{strategy_name}"] = norm_term_obj
                return f"{term_name}_red_{strategy_name} - saved as result of reduction\n" \
                    + f"{steps} steps to normalize\n" \
                    + f"LO-norm({term_name}) = {norm_term_obj.funky_str()}\n"
            else:
                return f"Term with name {term_name} or strategy {strategy_name} not available"
        else:
            return "Wrong reduction command format"

    def process_commands(self, command_str: str = None):
        if not command_str or command_str == "":
            return None

        if self.commands_buffer == "":
            if "#import" in command_str:
                self.commands_buffer = ""
                return self.console_engine_import(command_str)
            elif "#reduce-no-lim" in command_str:
                self.commands_buffer = ""
                return self.console_engine_reduce(command_str, False)
            elif "#reduce" in command_str:
                self.commands_buffer = ""
                return self.console_engine_reduce(command_str, False)
            elif "#define" in command_str:
                self.commands_buffer = ""
                if command_str.strip().endswith("#"):
                    return self.console_engine_define(command_str)
                else:
                    self.commands_buffer = command_str
                    return None
            else:
                return self.console_engine_show(command_str)

        if "#define" in self.commands_buffer:
            self.commands_buffer += "\n" + command_str
            if self.commands_buffer.strip().endswith("#"):
                command_str = self.commands_buffer
                self.commands_buffer = ""
                return self.console_engine_define(command_str)


if __name__ == "__main__":
    inter_obj = LambdaCalculusInterpreter()

    inter_obj.process_commands("#define sum_term = ( PLUS ")
    inter_obj.process_commands(" NUM[2] ")
    inter_obj.process_commands(" NUM[3] ")
    inter_obj.process_commands("")
    res = inter_obj.process_commands(")#")
    print(res)
    print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(inter_obj.process_commands("#reduce sum_term"))
    print(" >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(inter_obj.process_commands("""
            #define new_term = (IF (ISZERO sum_term_red) TRUE FALSE)
            #""")
          )
