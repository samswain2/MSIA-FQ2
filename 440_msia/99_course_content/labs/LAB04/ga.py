import config as conf
import random

city_dist_mat = None
config = conf.get_config()
gene_len = config.city_num
individual_num = config.individual_num
gen_num = config.gen_num
mutate_prob = config.mutate_prob


def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


class Individual:
    def __init__(self, genes=None):
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        fitness = 0.0
        for i in range(gene_len - 1):
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness


class Ga:
    def __init__(self, input_):
        global city_dist_mat
        city_dist_mat = input_
        self.best = None
        self.individual_list = []
        self.result_list = []
        self.fitness_list = []

    def cross(self):
        new_gen = []
        random.shuffle(self.individual_list)
        for i in range(0, individual_num - 1, 2):
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, gene_len - 2)
            index2 = random.randint(index1, gene_len - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(genes1))
            new_gen.append(Individual(genes2))
        return new_gen

    def mutate(self, new_gen):
        for individual in new_gen:
            if random.random() < mutate_prob:
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)
                index2 = random.randint(index1, gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.individual_list += new_gen

    def select(self):
        group_num = 10 
        group_size = 10 
        group_winner = individual_num // group_num
        winners = []
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            winners += group[:group_winner]
        self.individual_list = winners

    @staticmethod
    def rank(group):
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        new_gen = self.cross()
        self.mutate(new_gen)
        self.select()
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        self.individual_list = [Individual() for _ in range(individual_num)]
        self.best = self.individual_list[0]
        for i in range(gen_num):
            self.next_gen()
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list