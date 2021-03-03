import random
import numpy as np
import sys
from time import time
import cPickle
import subprocess as sub
from glob import glob

from cppn.networks import CPPN
from cppn.softbot import Genotype, Phenotype, Population
from cppn.tools.algorithms import Optimizer
from cppn.tools.utilities import make_material_tree, natural_sort, make_one_shape_only
from cppn.objectives import ObjectiveDict
from cppn.tools.evaluation import evaluate_population, is_body_valid, body_func
from cppn.tools.mutation import create_new_children_through_mutation
from cppn.tools.selection import pareto_selection

# actuation +/- 50%
# (1.5**(1/3)-1)/0.01 = 14.4714

DEBUG = False
reloadEx = False 

BLOCK_LEN = 3

EXP_NUM = 2
NON_FRACTAL = True

NUM_RECURSIONS = 0  # int(sys.argv[1])

BLOCK_SIZE = (BLOCK_LEN,)*3
BODY_SIZE = (BLOCK_LEN**(NUM_RECURSIONS+1),)*3

SEED = 100*EXP_NUM + int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)

if DEBUG:
    print "DEBUG MODE"
    sub.call("rm a{}_id0_fit-1000000000.hist".format(SEED), shell=True)
    sub.call("rm -r pickledPops{0} && rm -r data{0}".format(SEED), shell=True)

if reloadEx:
    sub.call("rm voxcraft-sim && rm vx3_node_worker", shell=True)
    sub.call("cp /users/s/k/skriegma/sim/build/voxcraft-sim .", shell=True)
    sub.call("cp /users/s/k/skriegma/sim/build/vx3_node_worker .", shell=True)

sub.call("mkdir pickledPops{}".format(SEED), shell=True)
sub.call("mkdir data{}".format(SEED), shell=True)
sub.call("cp base.vxa data{}/".format(SEED), shell=True)

GENS = 1001
POPSIZE = 8*2-1 # +1 for the randomly generated robot that is added each gen

SAVE_HIST_EVERY = 100  # gens

CHECKPOINT_EVERY = 1  # gens
MAX_TIME = 47  # [hours] evolution does not stop; after MAX_TIME checkpointing occurs at every generation.

DIRECTORY = "."
start_time = time()


def one_muscle(data):
    # block = np.array([[[0, 1, 1], [1, 1, 1], [0, 1, 1]],
    #                   [[0, 1, 1], [1, 0, 1], [0, 1, 1]],
    #                   [[0, 1, 1], [0, 0, 0], [0, 1, 1]],
    #                  ])
    # return np.rot90(block.T, k=1, axes=(1, 2))
    block = make_one_shape_only(np.greater(data, 0))
    return block.astype(np.int8)


class MyGenotype(Genotype):

    def __init__(self):

        Genotype.__init__(self, orig_size_xyz=BLOCK_SIZE, world_size=BODY_SIZE)

        if EXP_NUM == 3:
            self.add_network(CPPN(output_node_names=["Data"]))
            self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", output_type=int, func=one_muscle)

        if EXP_NUM == 2:
            self.add_network(CPPN(output_node_names=["shape", "muscleOrTissue"]))

            self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", func=make_material_tree, output_type=int,
                                            dependency_order=["shape", "muscleOrTissue"])

            self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
                                                            material_if_true=None, material_if_false="0")

            self.to_phenotype_mapping.add_output_dependency(name="muscleOrTissue", dependency_name="shape",
                                                            requirement=True, material_if_true="2", material_if_false="1")   

        if NON_FRACTAL:
            self.add_network(CPPN(output_node_names=["aggregation"]))
            self.to_phenotype_mapping.add_map(name="aggregation", tag=None, output_type=int, func=one_muscle) 

            self.add_network(CPPN(output_node_names=["aggregationOfAggregations"]))
            self.to_phenotype_mapping.add_map(name="aggregationOfAggregations", tag=None, output_type=int, func=one_muscle) 


class MyPhenotype(Phenotype):

    def is_valid(self):
        agg = None
        agg_of_aggs = None
        
        for name, details in self.genotype.to_phenotype_mapping.items():
            state = details["state"]
            if name == "Data":
                block = state
            elif name == "aggregation":
                agg = state
            elif name == "aggregationOfAggregations":
                agg_of_aggs = state

        body = body_func(block, 2, agg, agg_of_aggs)

        if not is_body_valid(body):
            return False

        return True


# Now specify the objectives for the optimization.
# Creating an objectives dictionary
my_objective_dict = ObjectiveDict()

# Adding an objective named "fitness", which we want to maximize.
# This information is returned by Voxelyze in a fitness .xml file, with a tag named "distance"
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<fitness_score>")

# Add an objective to minimize the age of solutions: promotes diversity
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

if DEBUG:
    # quick test to make sure evaluation is working properly:
    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)
    my_pop.seed = SEED
    evaluate_population(my_pop, record_history=True)
    exit()

if len(glob("pickledPops{}/Gen_*.pickle".format(SEED))) == 0:
    # Initializing a population of SoftBots
    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)
    my_pop.seed = SEED

    # Setting up our optimization
    my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

else:
    successful_restart = False
    pickle_idx = 0
    while not successful_restart:
        try:
            pickled_pops = glob("pickledPops{}/*".format(SEED))
            last_gen = natural_sort(pickled_pops, reverse=True)[pickle_idx]
            with open(last_gen, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = cPickle.load(handle)
            successful_restart = True

            my_pop = optimizer.pop
            my_optimization = optimizer
            my_optimization.continued_from_checkpoint = True
            my_optimization.start_time = time()

            random.setstate(random_state)
            np.random.set_state(numpy_random_state)

            print "Starting from pickled checkpoint: generation {}".format(my_pop.gen)

        except EOFError:
            # something went wrong writing the checkpoint : use previous checkpoint and redo last generation
            sub.call("touch IO_ERROR_$(date +%F_%R)", shell=True)
            pickle_idx += 1
            pass


my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=GENS, 
                    checkpoint_every=CHECKPOINT_EVERY, save_hist_every=SAVE_HIST_EVERY, 
                    directory=DIRECTORY)


# print "That took a total of {} minutes".format((time()-start_time)/60.)
# # finally, record the history of best robot at end of evolution so we can play it back in VoxCad
# my_pop.individuals = [my_pop.individuals[0]]
# evaluate_population(my_pop, record_history=True)

