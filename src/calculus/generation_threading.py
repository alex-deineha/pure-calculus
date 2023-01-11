from threading import Thread
from calculus.generation import *


class GenTermsThread(Thread):
    def __init__(self, count_terms=100, down_vertices_limit=50, up_vertices_limit=60,
                 random_average_count=20, thread_name="", mode="all"):
        super().__init__()
        self.unfiltered_terms = gen_lambda_terms(count=int(count_terms * 1.3), down_vertices_limit=down_vertices_limit,
                                                 up_vertices_limit=up_vertices_limit)
        self.gen_terms = []
        self.gen_stepsLO = []
        self.gen_stepsRI = []
        self.gen_stepsRand = []
        self.count_terms = count_terms
        self.down_vertices_limit = down_vertices_limit
        self.up_vertices_limit = up_vertices_limit
        self.random_average_count = random_average_count
        self.thread_name = thread_name
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def run(self):
        print(f"Running thread: {self.thread_name}")
        if self.mode in ["all", "gen"]:
            self.gen_terms, self.gen_stepsLO = \
                gen_filtered_lambda_terms(count_terms=self.count_terms,
                                          down_vertices_limit=self.down_vertices_limit,
                                          up_vertices_limit=self.up_vertices_limit,
                                          terms=self.unfiltered_terms)
            print(f"Thread {self.thread_name} is generated and filtered terms")

        if self.mode in ["all", "RI"]:
            print(f"Thread {self.thread_name} is doing RI norm")
            self.gen_stepsRI = [term.normalize(RightmostInnermostStrategy())[1] for term in tqdm(self.gen_terms)]
            print(f"Thread {self.thread_name} is DONE RI norm")

        if self.mode in ["all", "Rand"]:
            print(f"Thread {self.thread_name} is doing Random norm")
            self.gen_stepsRand = [
                sum([term.normalize(RandomStrategy())[1] for i in range(self.random_average_count)])
                / self.random_average_count
                for term in tqdm(self.gen_terms)
            ]
            print(f"Thread {self.thread_name} is DONE Random norm")
        print(f"Thread {self.thread_name} is DONE")

    def get_terms(self):
        return self.gen_terms

    def get_stepsLO(self):
        return self.gen_stepsLO

    def get_stepsRI(self):
        return self.gen_stepsRI

    def get_stepsRand(self):
        return self.gen_stepsRand


class GenTermsThreadV2(Thread):
    def __init__(self, count_terms=100,
                 random_average_count=20, thread_name="", mode="all"):
        super().__init__()
        self.gen_terms, self.gen_stepsLO = gen_filtered_lambda_terms_v2(count_terms)
        self.gen_stepsRI = []
        self.gen_stepsRand = []
        self.random_average_count = random_average_count
        self.thread_name = thread_name
        print(f"Th_{self.thread_name}: generated {len(self.gen_terms)} terms")
        self.mode = mode

    def run(self):
        print(f"Running thread: {self.thread_name}")

        if self.mode in ["all", "RI"]:
            print(f"Thread {self.thread_name} is doing RI norm")
            self.gen_stepsRI = [term.normalize(RightmostInnermostStrategy())[1] for term in tqdm(self.gen_terms)]
            print(f"Thread {self.thread_name} is DONE RI norm")

        if self.mode in ["all", "Rand"]:
            print(f"Thread {self.thread_name} is doing Random norm")
            self.gen_stepsRand = [
                sum([term.normalize(RandomStrategy())[1] for i in range(self.random_average_count)])
                / self.random_average_count
                for term in tqdm(self.gen_terms)
            ]
            print(f"Thread {self.thread_name} is DONE Random norm")

        print(f"Thread {self.thread_name} is DONE")

    def get_terms(self):
        return self.gen_terms

    def get_stepsLO(self):
        return self.gen_stepsLO

    def get_stepsRI(self):
        return self.gen_stepsRI

    def get_stepsRand(self):
        return self.gen_stepsRand
