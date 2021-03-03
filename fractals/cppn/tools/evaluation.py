import hashlib
from lxml import etree
import subprocess as sub
import numpy as np

from cppn.tools.utilities import make_one_shape_only


def zoom(a, mag):
    mag = int(mag)
    new = np.zeros((a.shape[0]*mag, a.shape[1]*mag, a.shape[2]*mag), dtype=np.int8)
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            for z in range(a.shape[2]):
                new[x*mag:(x+1)*mag, y*mag:(y+1)*mag, z*mag:(z+1)*mag] = a[x, y, z]
    return new


# def double(block):
#     workspace = np.tile(block, (block.shape[0],)*3)
#     mask = zoom(block.astype(np.int8), workspace.shape[0] / block.shape[0])
#     workspace *= mask > 0
#     return workspace


def blockify(structure, block):
    workspace = np.tile(block, (structure.shape[0],)*3)
    mask = zoom(structure.astype(np.int8), workspace.shape[0] / structure.shape[0])
    workspace *= mask > 0
    return workspace


def is_block_valid(block, n_recursions):

    if np.sum(block > 0) < block.shape[0]:
        return False

    bots = [block]
    for n in range(n_recursions):
        # bots += [double(bots[-1])]
        bots += [blockify(bots[-1], bots[0])]

    if np.sum(make_one_shape_only(bots[-1])) != np.sum(bots[-1] > 0):
        return False

    return True


def is_body_valid(body):

    if np.sum(body > 0) < body.shape[0]:
        return False

    if np.sum(make_one_shape_only(body)) != np.sum(body > 0):
        return False

    return True


def body_func(data, n_recursions):
    # print data
    block = make_one_shape_only(np.greater(data, 0))
    block = block.astype(np.int8)
    block[block>0] = data[block>0]

    bots = [block]
    for n in range(n_recursions):
        bots += [blockify(bots[-1], bots[0])]
    
    return bots[-1]


def evaluate_population(pop, record_history=False):

    seed = pop.seed

    N = len(pop)
    if record_history:
        N = 1  # only evaluate the best ind in the pop

    # clear old .vxd robot files from the data directory
    sub.call("rm data{}/*.vxd".format(seed), shell=True)

    # remove old sim output.xml if we are saving new stats
    if not record_history:
        sub.call("rm output{}.xml".format(seed), shell=True)

    num_evaluated_this_gen = 0

    # hash all inds in the pop
    if not record_history:

        for n, ind in enumerate(pop):

            ind.teammate_ids = []
            ind.duplicate = False
            data_string = ""
            for name, details in ind.genotype.to_phenotype_mapping.items():
                data_string += details["state"].tostring()
                m = hashlib.md5()
                m.update(data_string)
                ind.md5 = m.hexdigest()

            if (ind.md5 in pop.already_evaluated) and len(ind.fit_hist) == 0:  # line 141 mutations.py clears fit_hist for new designs
                # print "dupe: ", ind.id
                ind.duplicate = True
            
            # It's still possible to get duplicates in generation 0.
            # Then there's two inds with the same md5, age, and fitness (because one will overwrite the other).
            # We can adjust mutations so this is impossible
            # or just don't evaluate th new yet duplicate design.
    
    # evaluate new designs
    for r_num, r_label in enumerate(['a', 'b', 'c']):

        for n, ind in enumerate(pop[:N]):

            world_size = (ind.genotype.orig_size_xyz[0]**(r_num+1),)*3

            # don't evaluate if invalid
            if not ind.phenotype.is_valid():
                for rank, goal in pop.objective_dict.items():
                    if goal["name"] != "age":
                        setattr(ind, goal["name"], goal["worst_value"])

                print "Skipping invalid individual"

            # if it's a new valid design, or if we are recording history, create a vxd
            # new designs are evaluated with teammates from the entire population (new and old).
            elif (ind.md5 not in pop.already_evaluated) or record_history:

                num_evaluated_this_gen += 1
                pop.total_evaluations += 1

                (bx, by, bz) = ind.genotype.orig_size_xyz
                (wx, wy, wz) = world_size

                root = etree.Element("VXD")  # new vxd root

                if record_history:
                    # sub.call("rm a{0}_gen{1}.hist".format(seed, pop.gen), shell=True)
                    history = etree.SubElement(root, "RecordHistory")
                    history.set('replace', 'VXA.Simulator.RecordHistory')
                    etree.SubElement(history, "RecordStepSize").text = '50'
                    etree.SubElement(history, "RecordVoxel").text = '1'
                    etree.SubElement(history, "RecordLink").text = '0'
                    etree.SubElement(history, "RecordFixedVoxels").text = '0'  # too expensive to draw the walls of the dish
                    etree.SubElement(history, "RecordCoMTraceOfEachVoxelGroupfOfThisMaterial").text = '1'  # record CoM


                structure = etree.SubElement(root, "Structure")
                structure.set('replace', 'VXA.VXC.Structure')
                structure.set('Compression', 'ASCII_READABLE')
                etree.SubElement(structure, "X_Voxels").text = str(wx)
                etree.SubElement(structure, "Y_Voxels").text = str(wy)
                etree.SubElement(structure, "Z_Voxels").text = str(wz)

                for name, details in ind.genotype.to_phenotype_mapping.items():
                    # print name
                    state = details["state"]
                    if name == "Data":
                        body = body_func(state, r_num)
                        flattened_state = body.reshape(wz, wx*wy)
                    else:
                        controller = body_func(state, r_num)
                        flattened_state = controller.reshape(wz, wx*wy)

                    data = etree.SubElement(structure, name)
                    for i in range(flattened_state.shape[0]):
                        layer = etree.SubElement(data, "Layer")
                        if name == "Data":
                            str_layer = "".join([str(c) for c in flattened_state[i]])
                        else:
                            str_layer = "".join([str(c)+", " for c in flattened_state[i]])
                        layer.text = etree.CDATA(str_layer)


                # # morphology
                # if "Data" in ind.genotype.to_phenotype_mapping:
                #     for name, details in ind.genotype.to_phenotype_mapping.items():
                #         state = details["state"]
                #         body = body_func(state, r_num)
                #         flattened_state = body.reshape(wz, wx*wy)

                #         data = etree.SubElement(structure, name)
                #         for i in range(flattened_state.shape[0]):
                #             if name == "Data":
                #                 layer = etree.SubElement(data, "Layer")
                #                 str_layer = "".join([str(c) for c in flattened_state[i]])
                #                 layer.text = etree.CDATA(str_layer)
                
                # fixed phase offset controller
                controller = np.zeros(world_size)
                # for y in range(world_size[1]):
                #     controller[:, y, :] = 0.5*y/float(world_size[0]-1)
                controller[body==1] = 0.5
                
                flattened_controller = controller.reshape(controller.shape[2], controller.shape[0] * controller.shape[1])
                phase_offsets = etree.SubElement(structure, "PhaseOffset")
                for i in range(flattened_controller.shape[0]):
                    layer = etree.SubElement(phase_offsets, "Layer")
                    str_layer = "".join([str(c)+", " for c in flattened_controller[i]])
                    layer.text = etree.CDATA(str_layer)

                # save the vxd to data folder
                with open('data'+str(seed)+'/bot_{:04d}'.format(ind.id) + r_label + '.vxd', 'wb') as vxd:
                    vxd.write(etree.tostring(root))

    # ok let's finally evaluate all the robots in the data directory

    if record_history:  # just save history, don't assign fitness
        print "Recording the history of the run champ"
        for r_num, r_label in enumerate(['a', 'b', 'c']):
            sub.call("mkdir data{}".format(str(seed)+str(r_label)), shell=True)
            sub.call("cp base.vxa data{}".format(str(seed)+str(r_label)), shell=True)
            sub.call('cp data'+str(seed)+'/bot_{:04d}'.format(ind.id) + '{}.vxd'.format(r_label) +  ' data{}'.format(str(seed)+str(r_label)), shell=True)
            sub.call("./voxcraft-sim -i data{0} > {0}_id{1}_fit{2}.hist".format(str(seed)+str(r_label), pop[0].id, int(100*pop[0].fitness)), shell=True)
            sub.call("rm -r data{}".format(str(seed)+str(r_label)), shell=True)

    else:  # normally, we will just want to update fitness and not save the trajectory of every voxel

        print "GENERATION {}".format(pop.gen)

        print "Launching {0} voxelyze calls, out of {1} individuals".format(num_evaluated_this_gen, len(pop))


        while True:
            try:
                sub.call("./voxcraft-sim -i data{0} -o output{1}.xml".format(seed, seed), shell=True)
                # sub.call waits for the process to return
                # after it does, we collect the results output by the simulator
                root = etree.parse("output{}.xml".format(seed)).getroot()
                break

            except IOError:
                print "Dang it! There was an IOError. I'll re-simulate this batch again..."
                pass

            except IndexError:
                print "Shoot! There was an IndexError. I'll re-simulate this batch again..."
                pass
        

        for ind in pop:

            if ind.phenotype.is_valid() and ind.md5 not in pop.already_evaluated:

                for r_num, r_label in enumerate(['a', 'b', 'c']):
                    body_len = float(ind.genotype.orig_size_xyz[0]**(r_num+1))
                    ind.fit_hist += [float(root.findall("detail/bot_{:04d}".format(ind.id) + r_label + "/fitness_score")[0].text) / body_len]

                ind.fitness = np.min(ind.fit_hist)
                print "Assigning ind {0} fitness {1}".format(ind.id, ind.fitness)

                pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                                  for rank, details in
                                                  pop.objective_dict.items()]

