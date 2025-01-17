import argparse
import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../../'))


from fusion.scheduling import batch_size
from fusion.scheduling import Resource
from fusion.scheduling import LoopLowerBound
from fusion.scheduling import ScheduleGenerator
from fusion.scheduling import extract_info
from fusion.scheduling import CostModel
from fusion.scheduling import res_parse

from fusion.nn_models import all_networks
from fusion.nn_models import import_network

def do_scheduling_timeloop(net, batch, arch , buffer_size, dataflow, DRAM_BW, path, timeloop_cost_model=False, z_fusion=False, d_fusion=False, womincost=False, wofusion=False, wofusion_optimus=False, is_shiDianNao=False, simple_cost_model=True) :
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    print("Layer Fusion Optimization Start")

    # Resource.
    arch_info, dataflow_info = extract_info(arch, dataflow, buffer_size)
    resource = Resource.arch(arch_info)
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)
    
    # Network.
    batch_size.init(batch)
    network = import_network(net)
    print(network)

    cost_model = CostModel(network, resource)

    # optimal schedule
    sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound, dataflow_info, DRAM_BW, path, timeloop_cost_model, z_fusion, d_fusion, womincost, wofusion, wofusion_optimus, is_shiDianNao, simple_cost_model)
    schedule_info_list, _ = sg.schedule_search()

    print(schedule_info_list ,'\n')\
        
    others = False
    is_access = True
    if ((z_fusion == True) and (womincost == True)) or (wofusion == True) :
         others = True
    
    # result parsing
    cost, access = res_parse(schedule_info_list,
                        resource, cost_model,
                        sg, network,
                        loop_lower_bound,
                        path, arch_info, others, is_access)

    return schedule_info_list, cost, access

def do_scheduling(args):
    """
    Get optimal scheduling for given problem. Return a result schedule.
    """

    # Resource.
    arch_info, dataflow_info = extract_info(args.arch, args.dataflow)
    resource = Resource.arch(arch_info)

    # Unroll loop lower bound
    loop_lower_bound = LoopLowerBound.dataflow(dataflow_info)

    # Network.
    batch_size.init(args.batch)
    network = import_network(args.net)
    print(network)

    cost_model = CostModel(network, resource)

    # optimal schedule
    sg = ScheduleGenerator(network, resource, cost_model, loop_lower_bound)
    schedule_info_list, _ = sg.schedule_search()
    cost, access = res_parse(schedule_info_list,
                        resource, cost_model,
                        sg, network,
                        loop_lower_bound,
                        args.path, arch_info)

    return schedule_info_list, cost


def argparser():
    """
    Argument parser.
    """

    ap = argparse.ArgumentParser(description="AutoMorpher")
    # ===================================================================================
    # accelerator architecture
    # ===================================================================================
    ap.add_argument("arch", help="architecture specification")
    # ===================================================================================
    # model and batch size
    # ===================================================================================
    ap.add_argument('net',
                    help='network name, should be a .py file under "nns". '
                         'Choices: {}.'.format(', '.join(all_networks())))
    ap.add_argument("-d", "--dataflow", help="restriction of the dataflow space")
    ap.add_argument('--batch', type=int, default=1,
                    help='batch size')
    # ===================================================================================
    # reuse optimization selection
    # ===================================================================================
    ap.add_argument('--path', default='.',
                    help='path to store results')
    # ===================================================================================
    # verbose
    # ===================================================================================
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='show progress and details.')

    args = ap.parse_args()

    return args


def main():
    """
    Main function.
    """
    args = argparser()
    schedule_info_list, cost = do_scheduling(args)

    return 0 if cost < float("inf") else 2


if __name__ == '__main__':
    sys.exit(main())
