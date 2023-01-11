
import math
import numpy as np

from operator import add
from functools import reduce
from scipy.optimize import minimize
from queue import Queue

from .interlayer import InterLayerReuse
from .layer import LocalRegionLayer, ConvLayer
from .network import Network
from .resource import Resource
from .mapping_point import MappingPoint

from . import loop_enum as le
from .cost_model import CostModel

import os
import subprocess
import yaml
from yaml.loader import SafeLoader
import shutil
from . import parse_timeloop_output

class ScheduleGenerator(object):
    """
    Search optimal scheduling for neural networks.
    """

    def __init__(self, network, resource, cost_model, loop_lower_bound, dataflow_info, DRAM_BW, path,
                 timeloop_cost_model=False, z_fusion=False, d_fusion=False, womincost=True, wofusion=False, wofusion_optimus=False, is_shiDianNao=False, simple_cost_model=True):
        if not isinstance(network, Network):
            raise TypeError("ScheduleGenerator: network must be a Network instance.")
        if not isinstance(resource, Resource):
            raise TypeError("ScheduleGenerator: resource must be a Resource instance.")
        if not isinstance(cost_model, CostModel):
            raise TypeError("ScheduleGenerator: cost_model must be a CostModel instance.")

        self.network = network
        self.resource = resource
        # self.args = args
        self.loop_lower_bound = loop_lower_bound
        self.cost_model = cost_model
        self.z_fusion = z_fusion
        self.d_fusion = d_fusion
        self.womincost = womincost
        self.wofusion = wofusion
        self.is_shiDianNao = is_shiDianNao
        
        self.path = path
        self.DRAM_BW = DRAM_BW
        self.dataflow = dataflow_info["dataflow"]
        
        self.wofusion_optimus = wofusion_optimus
        self.timeloop_cost_model = timeloop_cost_model
        self.simple_cost_model = simple_cost_model
        self.debug_msg = True
        
    def schedule_search(self):

        if self.z_fusion or self.d_fusion or self.wofusion:
            dptable = dict()
            res_map_0, res_0 = self.others(dptable, 0)
            dptable = dict()
            res_map_1, res_1 = self.others(dptable, 1)
            res_map = [res_map_1, res_map_0]
            res = [res_1, res_0]
        else:
            dptable = dict()
            if self.wofusion_optimus :
                schedule_single_layer = []
                cost_single_layer = 0
                for layer in self.network :
                    cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule([layer])
                    schedule_single_layer += [[vertex_list, loop_block, loop_order, cost, sfil_fit]]
                    cost_single_layer += cost
                res_map, res = schedule_single_layer , cost_single_layer
            else :
                if len(self.network.firsts()) > 1:
                    nx = sorted(list(self.network.firsts()))
                    dsv = self._dsv(nx)
                    res_map, res = self.hafs_with_timeloop_cost_model([dsv], dptable)
                else:
                    res_map, res = self.hafs_with_timeloop_cost_model(list(self.network.firsts()), dptable)
        print()
        return res_map, res

    def hafs_with_timeloop_cost_model(self, fusion_group, dptable):

        if not isinstance(fusion_group, list):
            raise TypeError('HaFS: fusion_group must be a list.')
        
        g = []
        for layer in fusion_group:
            if layer in self.network:
                g.append(layer)
        if len(g) > 0:
            print("Evaluation Target:",g)
        dpkey = tuple(sorted(fusion_group))
        if dpkey in dptable:
            return dptable[dpkey]
        
        nx = self._next(fusion_group)
        if len(g) > 0:
            if dpkey in dptable:
                print(dptable[dpkey])
                schedule, g_cost = dptable[dpkey]
                vertex_list, loop_block, loop_order, cost, sfil_fit = self._find_schedule(g)
                if g_cost == float("inf"):
                    print(g ,"impossible to fusion\n")
                    return dptable[dpkey]
            else :
                g_cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(g)
                if g_cost == float("inf"):
                    dptable[dpkey] = [[]], float("inf")
                    print(g ,"impossible to fusion\n")
                    return dptable[dpkey]
                else :
                    dptable[dpkey] = [[vertex_list, loop_block, loop_order, g_cost, sfil_fit]], g_cost
                

        if len(nx) == 0:
            if dpkey in dptable:
                return dptable[dpkey]
            else :
                cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(g)
                dptable[dpkey] = [[vertex_list, loop_block, loop_order, cost, sfil_fit]], cost
                return dptable[dpkey]

        optimal_s, min_cost = [[]], float("inf")

        masked = self._reachable(nx)
        for c in sorted(nx):
            if masked.get(c, False):
                continue
            fuse_node = fusion_group + [c]
            s, cost1 = self.hafs_with_timeloop_cost_model(fuse_node, dptable)
            if cost1 < min_cost:
                min_cost = cost1
                optimal_s = s

        if not self._is_dsv(fusion_group):
            if len(nx) == 1:
                dsv = nx[0]
            else:
                dsv = self._dsv(nx)
            s, cost2 = self.hafs_with_timeloop_cost_model([dsv], dptable)
            if cost2 + g_cost < min_cost:
                min_cost = cost2 + g_cost
                optimal_s = [[vertex_list, loop_block, loop_order, g_cost, sfil_fit]] + s

        dptable[dpkey] = optimal_s, min_cost
        return optimal_s, min_cost

    def others(self, dptable, mode):
        q = Queue(maxsize=0)
        for layer in self.network.firsts():
            q.put(layer)
        L = []
        visit = dict()
        while not q.empty():
            layer = q.get()
            L.append(layer)

            for nx in reversed(self.network.nexts(layer)):
                Network_Input_Counter = 0
                if nx is None:
                    continue
                if nx not in visit:
                    visit[nx] = 1
                else:
                    visit[nx] += 1
                if None in self.network.prevs(nx) :
                    Network_Input_Counter = 1 
                if len(self.network.prevs(nx)) - Network_Input_Counter == visit[nx]:
                    q.put(nx)
        res_map, res = self.others_m([L[0]], dptable, L, 1, mode)
        return res_map, res

    def others_m(self, fusion_group, dptable, L, idx, mode):
        if not isinstance(fusion_group, list):
            raise TypeError('l78z_m: fusion_group must be a list.')
        dpkey = tuple(fusion_group)
        if dpkey in dptable:
            return dptable[dpkey]

        if idx == len(L):
            cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(fusion_group, mode)
            dptable[dpkey] = [[vertex_list, loop_block, loop_order, cost, sfil_fit]], cost
            return dptable[dpkey]

        optimal_s, min_cost = [[]], float("inf")

        fuse_node = fusion_group + [L[idx]]
        s, cost = self.others_m(fuse_node, dptable, L, idx+1, mode)
        if cost < min_cost:
            min_cost = cost
            optimal_s = s

        s, cost = self.others_m([L[idx]], dptable, L, idx+1, mode)
        g_cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_schedule(fusion_group, mode)
        if cost + g_cost < min_cost:
            min_cost = cost + g_cost
            optimal_s = [[vertex_list, loop_block, loop_order, g_cost, sfil_fit]] + s

        dptable[dpkey] = optimal_s, min_cost
        return optimal_s, min_cost

    def _next(self, fusion_group):

        nexts = set()
        for layer in fusion_group:
            if layer not in self.network:
                for nx in layer.split("|")[1:]:
                    if nx not in fusion_group and nx:
                        nexts.add(nx)
            else:
                for nx in self.network.nexts(layer):
                    if nx not in fusion_group and nx:
                        nexts.add(nx)

        return list(nexts)

    def _is_dsv(self, fusion_group):

        return len(fusion_group) == 1 and fusion_group[0] not in self.network

    @ staticmethod
    def _dsv(nx):
        dsv = 'dsv'
        for n in sorted(nx):
            dsv += ('|' + n)

        return dsv

    def _reachable(self, child):
        masked = dict()

        def dfs(v):
            masked[v] = True
            for vc in self.network.nexts(v):
                if not masked.get(vc, False):
                    dfs(vc)

        for rc in child:
            if not masked.get(rc, False):
                for nn in self.network.nexts(rc):
                    dfs(nn)

        return masked

    def _find_uni_layer_schedule_timeloop(self, g):
        
        layer = self.network[g]
        cost, cost_initial, cost_optimus, cost_timeloop, loop_block, loop_order, self.sfil_fit , mapping = float('inf'), float('inf'), float('inf') , float('inf') , [None], [None], False , None

        if isinstance(layer, LocalRegionLayer):
            cost = 0
            return cost, loop_block, loop_order, self.sfil_fit

        loop_lower_bound = self.loop_lower_bound(layer)
        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count
        if s > layer.total_filter_size:
            self.sfil_fit = True

        

        if self.dataflow == 'OS' :
            if layer.wofm < self.resource.array_width :
                padded_hofm = int(math.ceil(layer.wofm*layer.wofm/self.resource.array_width))
                padded_wofm = self.resource.array_width 
                padded_layer = ConvLayer(layer.nifm, layer.nofm, [padded_hofm, padded_wofm] ,sfil=[layer.hfil,layer.wfil],strd=[layer.hstd, layer.wstd], nimg=layer.nimg)
                layer = padded_layer
                loop_lower_bound = self.loop_lower_bound(layer)

        count = 0
        if self.dataflow == 'WS' :
            schedulist_v2 = _unilayer_schedule_list_v2_ws
        elif self.dataflow == 'OS' :
            schedulist_v2 = _unilayer_schedule_list_v2_os
                        
        for scheduling in schedulist_v2:
            if count == 0 :
                self.mapping_t = "INPUT_REUSE"
            elif count == 1 :
                self.mapping_t = "PSUM_REUSE"
            elif count == 2 :
                self.mapping_t = "FILTER_REUSE"
            count += 1
            cost_t, cost_optimus_t, cost_timeloop_t = float('inf') ,float('inf') , float('inf')
            cost_t, loop_block_t, loop_order_t, glb_access_cost_t, dram_access_cost_t = scheduling(layer, self.resource, loop_lower_bound)
            cost_t = int(cost_t) if cost_t != float('inf') else cost_t

            if cost_t != float('inf') :
                if self.debug_msg : 
                    print(" DRAM-Buffer Mapping:", loop_block_t, loop_order_t, self.mapping_t, "Optimus Simple:", cost_t)
                
                # if self.simple_cost_model :
                cost_inner_list, point_list = self.mapping_timeloop([g],[loop_block_t],[loop_order_t])
                DRAM_access = self.cost_model.get_level_access_unilayer_and_innerlevel_timeloop_simple_cost(point_list, [g], 2)
                
                R,S,C,Q,P,M,B = loop_block_t[0],loop_block_t[1],loop_block_t[2],loop_block_t[3],loop_block_t[4],loop_block_t[5],loop_block_t[6]
                Input_Size = layer.total_ifmap_size_timeloop_simple_cost*DRAM_access[0]
                Weight_Size = layer.total_filter_size*DRAM_access[2]
                Output_Read_Size = layer.total_ofmap_size*(DRAM_access[1]-1)/2
                Output_Write_Size = layer.total_ofmap_size*(DRAM_access[1]+1)/2
                
                if self.dataflow == 'WS' :
                    Execution_Cycle = (B*P*Q+32) * ((layer.nimg*layer.wofm*layer.hofm)//(B*P*Q)) * (layer.hfil*layer.wfil*layer.nifm//32) * (layer.nofm//32)
                elif self.dataflow == 'OS' :
                    Execution_Cycle = (R*S*C+32) * ((layer.hfil*layer.wfil*layer.nifm)//(R*S*C)) * (layer.nimg*layer.wofm*layer.hofm//32) * (layer.nofm//32)
                    
                READ_CYCLE = int((Input_Size + Weight_Size + Output_Read_Size)//self.DRAM_BW)
                WRITE_CYCLE = int((Output_Write_Size)//self.DRAM_BW)
                Result_Cycle = max(Execution_Cycle,READ_CYCLE,WRITE_CYCLE)
                
                if self.debug_msg : 
                    print("     ", DRAM_access, Result_Cycle, Execution_Cycle, READ_CYCLE, WRITE_CYCLE)
                
                if not self.simple_cost_model :
                    
                    loop_block, loop_order = loop_block_t, loop_order_t
                    sublayer = ConvLayer(loop_block[le.C], loop_block[le.K], [loop_block[le.H], loop_block[le.W]], sfil=[loop_block[le.R], loop_block[le.D]], strd=[layer.hstd, layer.wstd], nimg=loop_block[le.B])
                    loop_lower_bound = self.loop_lower_bound(sublayer)
                    s = self.resource.buffer(0).capacity
                    actual_s = s * self.resource.paras[0].count
                    point_t, blocking_t, ordering_t, partition_t = None, None, None, None
                    
                    if self.dataflow == 'WS' :
                        schedulist = _unilayer_schedule_list_v1_ws
                        mapping_t = "WS"
                    elif self.dataflow == 'OS' :
                        schedulist = _unilayer_schedule_list_v1_os
                        mapping_t = "OS"
                    for scheduling in schedulist:
                        cost_t, loop_block_t, loop_order_t = scheduling(sublayer, actual_s, loop_lower_bound)
                        if cost_t < float('inf'):
                            
                            blocking_t = [loop_block_t, loop_block, layer.dimension]
                            loop_order_innermost = [le.NUM - 1] * le.NUM
                            non_max_block = [i for i, e in enumerate(loop_block_t) if (e != 1)]
                            order = 0
                            for i in non_max_block:
                                loop_order_innermost[i] = order
                                order += 1
                            blocking_t_timeloop = [loop_block_t, loop_block, layer.dimension]
                            ordering_t_timeloop = [loop_order_innermost, loop_order_t, loop_order]
                            
                            # TODO support unroll in other levels
                            innermost_partition = [1] * le.NUM
                            innermost_para_dim = []
                            for i, lp in enumerate(self.loop_lower_bound.unroll_loop):
                                innermost_para_dim.append([lp])
                                partition_size = self.loop_lower_bound.loop_lower_bound_init[lp]
                                if loop_block_t[lp] >= partition_size:
                                    innermost_partition[lp] = partition_size
                                else:
                                    innermost_partition[lp] = loop_block_t[lp]
                                    if self.resource.replication and self.loop_lower_bound.replication_loop is not None:
                                        para = partition_size // loop_block_t[lp]
                                        replp = self.loop_lower_bound.replication_loop[i]
                                        if para > 1 and loop_block_t[replp] > 1:
                                            innermost_para_dim[i].append(replp)
                                            innermost_partition[replp] = para if loop_block_t[replp] > para else loop_block_t[replp]
                    
                        Total_Cycle_Timeloop, Total_Ideal_Cycle_Timeloop, Total_Slowdown, Total_Energy, EnergyperMAC = self.timeloop_model_uni_layer(g, layer, mapping_t, blocking_t_timeloop, ordering_t_timeloop, innermost_partition) 
                        Result_Cycle = Total_Cycle_Timeloop
                                        
                cost_timeloop_t = Result_Cycle 
                if cost_timeloop_t < cost:
                    mapping, cost_initial, cost_optimus, cost_timeloop, loop_block, loop_order = self.mapping_t, cost_t, cost_optimus_t, cost_timeloop_t, loop_block_t, loop_order_t

        cost = cost_timeloop         
        print("Optimal Result: - Mapping:", mapping, loop_block, "Performance Cost:", cost_timeloop)

        return cost, [loop_block], [loop_order], self.sfil_fit
    
     
    def _find_uni_layer_schedule_optimus(self, g):
        layer = self.network[g]
        cost, loop_block, loop_order, is_filter_fit = float('inf'), [None], [None], False

        if isinstance(layer, LocalRegionLayer):
            cost = 0
            return cost, loop_block, loop_order, is_filter_fit

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count
        if s > layer.total_filter_size:
            is_filter_fit = True

        loop_lower_bound = self.loop_lower_bound(layer)

        if self.network.net_name == "SqueezeNet":
            for scheduling in _unilayer_schedule_list_v1:
                cost_t, loop_block_t, loop_order_t = scheduling(layer, s, loop_lower_bound)
                if cost_t < cost:
                    cost, loop_block, loop_order = cost_t, loop_block_t, loop_order_t
        else:
            for scheduling in _unilayer_schedule_list_v2:
                cost_t, loop_block_t, loop_order_t, glb_access_cost_t, dram_access_cost_t \
                    = scheduling(layer, self.resource, loop_lower_bound)
                if cost_t != float('inf') :
                    if self.debug_msg :
                        print(g, cost_t, loop_block_t, loop_order_t, dram_access_cost_t)
                if cost_t < cost:
                    cost, loop_block, loop_order = cost_t, loop_block_t, loop_order_t

        print("Optimal Result:",g, loop_block, "Cost:", cost)
        return cost, [loop_block], [loop_order], is_filter_fit

    def _find_uni_layer_schedule_others(self, g, mode):
        layer = self.network[g]
        cost, loop_block, loop_order, is_filter_fit = float('inf'), [None], [None], False

        if isinstance(layer, LocalRegionLayer):
            cost = 0
            return cost, loop_block, loop_order, is_filter_fit

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count
        if s > layer.total_filter_size:
            is_filter_fit = True

        loop_lower_bound = self.loop_lower_bound(layer)

        if self.is_shiDianNao:
            if mode == 0:
                cost, loop_block, loop_order = _filterr_v1(layer, s, loop_lower_bound)
            else:
                cost, loop_block, loop_order, _, _ \
                        = _filterr_v2(layer, self.resource, loop_lower_bound)
        else:
            if mode == 0:
                cost, loop_block, loop_order = _filterr_v1(layer, s, loop_lower_bound)
            else:
                cost, loop_block, loop_order, _, _ \
                        = _filterr_v2(layer, self.resource, loop_lower_bound)

        return cost, [loop_block], [loop_order], is_filter_fit

    def _find_uni_layer_schedule(self, g, mode=0):
        if self.timeloop_cost_model :
            return self._find_uni_layer_schedule_timeloop(g)
        else :    
            if self.d_fusion or self.z_fusion or self.wofusion:
                return self._find_uni_layer_schedule_others(g, mode)
            else :
                return self._find_uni_layer_schedule_optimus(g)

    def _find_multi_layer_schedule(self, fusion_group, mode):
        if self.timeloop_cost_model :
            cost = float('inf')
            input_size_t , weight_size_t , output_size_t = float('inf') , float('inf')  , float('inf')
            
            ilr = InterLayerReuse(self.network, fusion_group, self.resource, self.dataflow, self.loop_lower_bound, z_fusion=self.z_fusion, d_fusion=self.d_fusion, womincost=self.womincost)
            
            if not ilr.valid or self.wofusion:
                return float('inf'), None, None, None, False
            
            # self.firsts , self.lasts, self.network_prevs , vertex_list = ilr.firsts , ilr.lasts , ilr.network.prevs , ilr.dag_vertex_list
            # self.heads , self.tails, self.bodys, self.singles = [] , [] , [] , []
            # ext_input_list , first_list , ext_output_list , ext_output_list , last_list = [] , [] , [] , [] , []
            
            # for ll in self.firsts :
            #     input_layer = self.network_prevs(ll)
            #     if input_layer not in ext_input_list :
            #         ext_input_list.append(input_layer)
            #         first_list.append(ll)
            # self.heads = first_list
            
            # for ll in self.lasts :
            #     if isinstance(self.network[ll], LocalRegionLayer) :
            #         for candidate in self.network_prevs(ll) :
            #             if (candidate is not None) and (candidate in vertex_list) and (isinstance(self.network[candidate], ConvLayer)):
            #                 if (ll not in ext_output_list) and (candidate not in last_list) :
            #                     ext_output_list.append(ll)
            #                     last_list.append(candidate)
            # self.tails = ilr.lasts
            
            # for candidate_layer1 in last_list :
            #     if candidate_layer1 not in self.tails :
            #         self.tails.append(candidate_layer1)
            
            # for candidate_layer2 in self.heads :
            #     if candidate_layer2 in self.tails :
            #         self.singles.append(candidate_layer2)
            
            # # fusion gorup with only single layers
            # if sorted(self.singles) == sorted(vertex_list) :
            #     return float('inf'), None, None, None, False
            
            # for candidate_layer_singlelayer in self.singles :
            #     self.heads.remove(candidate_layer_singlelayer)
            #     self.tails.remove(candidate_layer_singlelayer)
                
            # for candidate_layers_body in vertex_list :
            #     if (candidate_layers_body not in self.heads) and (candidate_layers_body not in self.tails) and (candidate_layers_body not in self.singles) :
            #         self.bodys.append(candidate_layers_body)
            
            # print("Multi Layer Schedule", fusion_group ,"-> Head :" , self.heads , ", Body :" , self.bodys , ", Tail :" , self.tails , ", Single :" , self.singles)
            # print(ilr.firsts , ilr.lasts)
            for weight_reuse in [True, False] :
                for overlap_reuse in [True, False] :
                    ilr.sched_timeloop_cost_model(overlap_reuse,weight_reuse)
                    
                    self.firsts , self.lasts, self.network_prevs , vertex_list = ilr.firsts , ilr.lasts , ilr.network.prevs , ilr.dag_vertex_list
                    self.heads , self.tails, self.bodys, self.singles = [] , [] , [] , []
                    ext_input_list , first_list , ext_output_list , ext_output_list , last_list = [] , [] , [] , [] , []
                    
                    for ll in self.firsts :
                        input_layer = self.network_prevs(ll)
                        if input_layer not in ext_input_list :
                            ext_input_list.append(input_layer)
                            first_list.append(ll)
                    self.heads = first_list
                    
                    for ll in self.lasts :
                        if isinstance(self.network[ll], LocalRegionLayer) :
                            for candidate in self.network_prevs(ll) :
                                if (candidate is not None) and (candidate in vertex_list) and (isinstance(self.network[candidate], ConvLayer)):
                                    if (ll not in ext_output_list) and (candidate not in last_list) :
                                        ext_output_list.append(ll)
                                        last_list.append(candidate)
                    self.tails = ilr.lasts
                    
                    for candidate_layer1 in last_list :
                        if candidate_layer1 not in self.tails :
                            self.tails.append(candidate_layer1)
                    
                    for candidate_layer2 in self.heads :
                        if candidate_layer2 in self.tails :
                            self.singles.append(candidate_layer2)
                    
                    # fusion gorup with only single layers
                    if sorted(self.singles) == sorted(vertex_list) :
                        return float('inf'), None, None, None, False
                    
                    for candidate_layer_singlelayer in self.singles :
                        self.heads.remove(candidate_layer_singlelayer)
                        self.tails.remove(candidate_layer_singlelayer)
                        
                    for candidate_layers_body in vertex_list :
                        if (candidate_layers_body not in self.heads) and (candidate_layers_body not in self.tails) and (candidate_layers_body not in self.singles) :
                            self.bodys.append(candidate_layers_body)
                    
                    print("Multi Layer Schedule", fusion_group ,"-> Head :" , self.heads , ", Body :" , self.bodys , ", Tail :" , self.tails , ", Single :" , self.singles)
                    print(ilr.firsts , ilr.lasts)
            
                    print(weight_reuse,overlap_reuse,ilr.total_tile_num)
                    if ilr.total_tile_num < 1 :
                        # print(0)
                        continue
                    if (overlap_reuse == 1) and (ilr.total_tile_num == 1) :
                        continue
                    
                    cost_initial , cost_optimus , cost_timeloop, cost_ideal = 0 , 0 , 0 , 0
                    cost_initial, loop_block, loop_order, vertex_list = ilr.q, ilr.loop_block, ilr.loop_order, ilr.dag_vertex_list
                    self.H_recompute , self.H_reuse , self.H_epilogue_recompute , self.H_epilogue_reuse = ilr.H_recompute , ilr.H_reuse , ilr.H_epilogue_recompute , ilr.H_epilogue_reuse
                    self.height_tile_num , self.batch_tile_num , self.total_tile_num = ilr.height_tile_num , ilr.batch_tile_num , ilr.total_tile_num
                    
                    # if self.simple_cost_model :
                    analytic_cycle_inner_list, ideal_cycle_inner_list, ops_count_list , input_size_list,  weight_size_list,  output_size_list,  point_list = self.fused_simple_cost_model(vertex_list, loop_block, loop_order, overlap_reuse, weight_reuse)
                    cost_timeloop = self.batch_tile_num * sum(analytic_cycle_inner_list) 
                    cost_ideal = self.batch_tile_num * sum(ideal_cycle_inner_list)
                    input_size = self.batch_tile_num * sum(input_size_list)
                    weight_size = self.batch_tile_num * sum(weight_size_list)
                    output_size = self.batch_tile_num * sum(output_size_list) 
                    ops_count = self.batch_tile_num * sum(ops_count_list)
                    if self.debug_msg : 
                        print(vertex_list, "Reuse(Overlap/Weight):", overlap_reuse, weight_reuse , "Performance Cost:",cost_timeloop , round(cost_ideal/cost_timeloop,3) , 'I/W/O:', input_size/1024 , weight_size/1024 , output_size/1024, (input_size+weight_size+output_size)/1024,'\n')
                    if cost_timeloop < cost :
                        cost = cost_timeloop
                        overlap_reuse_t = overlap_reuse
                        weight_reuse_t = weight_reuse
                        input_size_t = input_size
                        weight_size_t = weight_size
                        output_size_t = output_size
                        self.sfil_fit = ilr.sfil_fit
            if cost == float('inf') :
                return float('inf'), None, None, None, False
            else :
                print(" Optimal Result: - Reuse(Overlap/Weight):", overlap_reuse_t, weight_reuse_t , ", Performance Cost:",cost , ', Access(I/W/O/Total, KB):', input_size_t/1024 , weight_size_t/1024 , output_size_t/1024 , (input_size_t+weight_size_t+output_size_t)/1024)
        else :
            ilr = InterLayerReuse(self.network, fusion_group, self.resource, self.dataflow, self.loop_lower_bound,
                                z_fusion=self.z_fusion, d_fusion=self.d_fusion, womincost=self.womincost)
            
            if not ilr.valid or self.wofusion:
                return float('inf'), None, None, None, False

            ilr.sched_timeloop_cost_model(True,True)
            
            if not ilr.valid : 
                ilr.sched_timeloop_cost_model(True,False)

            if not ilr.valid :
                return float('inf'), None, None, None, False
            else :
                cost, loop_block, loop_order, vertex_list = ilr.q, ilr.loop_block, ilr.loop_order, ilr.dag_vertex_list
                print(" Optimal Result: - ", fusion_group, cost)
                
        return cost, loop_block, loop_order, vertex_list, ilr.sfil_fit

    def fused_simple_cost_model(self, g, loop_block_g, loop_order_g, overlap_reuse, weight_reuse):

        cost_inner_g = [None for _ in g]
        cost_inner_g_ideal = [None for _ in g]
        point_g = [None for _ in g]
        Ops_count , Input_count , Weight_count , Output_count = [None for _ in g] , [None for _ in g] , [None for _ in g] , [None for _ in g]
        t = 0

        for layer_name, loop_block, loop_order, H_recompute, H_reuse, H_epilogue_recompute, H_epilogue_reuse in zip(g, loop_block_g, loop_order_g, self.H_recompute, self.H_reuse, self.H_epilogue_recompute, self.H_epilogue_reuse):
            layer = self.network[layer_name]
            
            Weight_Access = 1
            Input_Access = 1
            Output_Access = 1
            if (layer_name in self.firsts) :
                Input_Access = 1
                Output_Access = 0
            elif (layer_name in self.bodys) :
                Input_Access = 0
                Output_Access = 0
            elif (layer_name in self.lasts) :
                Input_Access = 0
                Output_Access = 1
            elif (layer_name in self.single_layers) :
                Input_Access = 1
                Output_Access = 1
                
            if not isinstance(layer, ConvLayer):
                cost_inner_g[t], cost_inner_g_ideal[t], point_g[t] = 0, 0, None
                Ops_count[t] , Input_count[t] , Weight_count[t] , Output_count[t] = 0 , 0 , 0 , 0
                t += 1
                continue
            
            if self.dataflow == 'OS' and (loop_block[3] < 32) :
                    padded_wofm_loop = 32
                    H_recompute = math.ceil(H_recompute*loop_block[3]/32)
                    H_reuse = math.ceil(H_reuse*loop_block[3]/32)
                    H_epilogue_recompute = math.ceil(H_epilogue_recompute*loop_block[3]/32)
                    H_epilogue_reuse = math.ceil(H_epilogue_reuse*loop_block[3]/32)            
                    loop_block[3] = padded_wofm_loop
                    
            cost_inner_g[t], cost_inner_g_ideal[t], point_g[t] = 0, 0, None
            Ops_count[t] , Input_count[t] , Weight_count[t] , Output_count[t] = 0 , 0 , 0 , 0
            
            if self.height_tile_num == 1 :               
                height_sizes = [H_recompute]
            elif self.height_tile_num == 2 :
                if overlap_reuse :
                    height_sizes = [H_reuse,H_epilogue_reuse]
                else :
                    height_sizes = [H_reuse,H_epilogue_recompute]
            elif self.height_tile_num > 2 :
                if overlap_reuse :
                    height_sizes = [H_reuse] + (self.height_tile_num-2)*[H_reuse] +[H_epilogue_reuse]
                else :
                    height_sizes = [H_reuse] + (self.height_tile_num-2)*[H_recompute] +[H_epilogue_recompute]
            R,S,C,Q,P,M,B = loop_block[0],loop_block[1],loop_block[2],loop_block[3],loop_block[4],loop_block[5],loop_block[6]
            TB = loop_block[6]
            
            if self.debug_msg : 
                print("  ", layer_name, height_sizes , loop_block)
                
            for ith_iteration,height_size in enumerate(height_sizes):
                if weight_reuse == 1 :
                    if ith_iteration == 0 :
                        Weight_Access = 1
                    else :
                        Weight_Access = 0
                
                TP = height_size
                Weight_Size = R*S*C*M
                
                if overlap_reuse :
                    TW, TH = layer.wstd*Q, layer.hstd*TP
                else :
                    TW, TH = layer.wstd*Q, (layer.hstd*(TP-1)+S)
                    
                Input_Size = TB*TW*TH*C
                Output_Size = TB*TP*Q*M
                if self.dataflow == 'WS' :
                    Execution_Cycle = (TB*TP*Q+32)*(R*S*C//32)*(M//32)
                elif self.dataflow == 'OS' :
                    Execution_Cycle = (R*S*C+32)*(TB*TP*Q//32)*(M//32)
                    
                Execution_Cycle = (TB*TP*Q+32)*R*S*(C//32)*(M//32)
                READ_CYCLE = (Input_Access*Input_Size + Weight_Access*Weight_Size)//self.DRAM_BW
                WRITE_CYCLE = (Output_Access*Output_Size)//self.DRAM_BW
                result_cycle = max(Execution_Cycle,READ_CYCLE,WRITE_CYCLE)
                
                cost_inner_g[t] += result_cycle
                cost_inner_g_ideal[t] += Execution_Cycle
                Ops_count[t] += (TB*TP*Q)*R*S*C*M
                Input_count[t] += Input_Access*Input_Size
                Weight_count[t] += Weight_Access*Weight_Size
                Output_count[t] += Output_Access*Output_Size
                if self.debug_msg : 
                    if len(height_sizes) == 1 :
                        print("   Output :", TB,'*',TP,'*',Q,'*',M , "Input :", TB,'*',TW,'*',TP,'*',C ,"Weight:",  M,'*',R,'*',S,'*',C , Execution_Cycle, READ_CYCLE , WRITE_CYCLE)
                    elif len(height_sizes) == 2 :
                        if ith_iteration == 0 or ith_iteration == 1 :
                            print("   Output :", TB,'*',TP,'*',Q,'*',M , "Input :", TB,'*',TW,'*',TP,'*',C ,"Weight:",  M,'*',R,'*',S,'*',C , Execution_Cycle, READ_CYCLE , WRITE_CYCLE)
                    elif len(height_sizes) > 2 :
                        if ith_iteration == 0 or ith_iteration == 1 or ith_iteration == len(height_sizes) - 1 :
                            print("   Output :", TB,'*',TP,'*',Q,'*',M , "Input :", TB,'*',TW,'*',TP,'*',C ,"Weight:",  M,'*',R,'*',S,'*',C , Execution_Cycle, READ_CYCLE , WRITE_CYCLE)
                
            point_g[t] = None
            cost_inner_g[t] *= (B//TB)
            cost_inner_g_ideal[t] *= (B//TB)
            Ops_count[t] *= (B//TB)
            Input_count[t] *= (B//TB)
            Weight_count[t] *= (B//TB)
            Output_count[t] *= (B//TB)
            if self.debug_msg : 
                print("   ",layer_name,cost_inner_g[t],cost_inner_g_ideal[t] , round(cost_inner_g_ideal[t]/cost_inner_g[t],3), Ops_count[t], Input_count[t]/1024 , Weight_count[t]/1024 , Output_count[t]/1024 ,(Input_count[t]+Weight_count[t]+Output_count[t])/1024)
            t += 1

        return cost_inner_g, cost_inner_g_ideal, Ops_count, Input_count, Weight_count, Output_count, point_g
    
    def _find_schedule(self, g, mode=0):
        if len(g) == 1:
            cost, loop_block, loop_order, sfil_fit = self._find_uni_layer_schedule(g[0], mode)
            vertex_list = g
        else:
            cost, loop_block, loop_order, vertex_list, sfil_fit = self._find_multi_layer_schedule(g, mode)
        if cost != float('inf') :
            cost = int(cost)
        print()
        return cost, loop_block, loop_order, vertex_list, sfil_fit

    def mapping_timeloop(self, g, loop_block_g, loop_order_g):
        cost_inner_g = [None for _ in g]
        point_g = [None for _ in g]
        t = 0
        for layer_name, loop_block, loop_order in zip(g, loop_block_g, loop_order_g):
            layer = self.network[layer_name]
            if not isinstance(layer, ConvLayer):
                cost_inner_g[t], point_g[t] = 0, None
                t += 1
                continue

            cost_inner, loop_block_n, loop_order_n = float('inf'), None, None
            sublayer = ConvLayer(loop_block[le.C], loop_block[le.K], [loop_block[le.H], loop_block[le.W]],
                                 sfil=[loop_block[le.R], loop_block[le.D]],
                                 strd=[layer.hstd, layer.wstd], nimg=loop_block[le.B])
            loop_lower_bound = self.loop_lower_bound(sublayer)

            s = self.resource.buffer(0).capacity
            actual_s = s * self.resource.paras[0].count
            point_t, blocking_t, ordering_t, partition_t = None, None, None, None
            
            if self.dataflow == 'WS' :
                schedule_list = _unilayer_schedule_list_v1_ws
            elif self.dataflow == 'OS' :
                schedule_list = _unilayer_schedule_list_v1_os
            else :
                schedule_list = _unilayer_schedule_list_v1
            
            for scheduling in schedule_list:
                cost_t, loop_block_t, loop_order_t = \
                    scheduling(sublayer, actual_s, loop_lower_bound)

                if cost_t < float('inf'):
                    blocking_t = [loop_block_t, loop_block, layer.dimension]

                    loop_order_innermost = [le.NUM - 1] * le.NUM
                    non_max_block = [i for i, e in enumerate(loop_block_t) if (e != 1)]
                    order = 0
                    for i in non_max_block:
                        loop_order_innermost[i] = order
                        order += 1

                    ordering_t = [loop_order_innermost, loop_order_t, loop_order]

                    # TODO support unroll in other levels
                    innermost_partition = [1] * le.NUM
                    innermost_para_dim = []
                    for i, lp in enumerate(self.loop_lower_bound.unroll_loop):
                        innermost_para_dim.append([lp])
                        partition_size = self.loop_lower_bound.loop_lower_bound_init[lp]
                        if loop_block_t[lp] >= partition_size:
                            innermost_partition[lp] = partition_size
                        else:
                            innermost_partition[lp] = loop_block_t[lp]
                            if self.resource.replication and self.loop_lower_bound.replication_loop is not None:
                                para = partition_size // loop_block_t[lp]
                                replp = self.loop_lower_bound.replication_loop[i]
                                if para > 1 and loop_block_t[replp] > 1:
                                    innermost_para_dim[i].append(replp)
                                    innermost_partition[replp] = para \
                                        if loop_block_t[replp] > para else loop_block_t[replp]

                    para_loop_dim_list = [innermost_para_dim, [], []]
                    partition_t = [innermost_partition, [1] * le.NUM, [1] * le.NUM]

                    point_t = MappingPoint(list(zip(*ordering_t)), list(zip(*blocking_t)),
                                           list(zip(*partition_t)), para_loop_dim_list)
                    buffer_access = self.cost_model.get_level_access([point_t], [layer_name], 1)
                    cost_t = math.ceil(reduce(add, buffer_access, 0))
                    # print('  Buffer-PE Mapping:', loop_block_t, loop_order_t, cost_t)
                if cost_t < cost_inner:
                    point_g[t] = point_t
                    cost_inner_g[t] = cost_t
                    cost_inner = cost_t
        
            t += 1

        return cost_inner_g, point_g
    
    def mapping(self, g, loop_block_g, loop_order_g):
        cost_inner_g = [None for _ in g]
        point_g = [None for _ in g]
        t = 0
        for layer_name, loop_block, loop_order in zip(g, loop_block_g, loop_order_g):
            layer = self.network[layer_name]
            if not isinstance(layer, ConvLayer):
                cost_inner_g[t], point_g[t] = 0, None
                t += 1
                continue

            cost_inner, loop_block_n, loop_order_n = float('inf'), None, None
            sublayer = ConvLayer(loop_block[le.C], loop_block[le.K], [loop_block[le.H], loop_block[le.W]],
                                 sfil=[loop_block[le.R], loop_block[le.D]],
                                 strd=[layer.hstd, layer.wstd], nimg=loop_block[le.B])
            loop_lower_bound = self.loop_lower_bound(sublayer)

            s = self.resource.buffer(0).capacity
            actual_s = s * self.resource.paras[0].count
            point_t, blocking_t, ordering_t, partition_t = None, None, None, None
            for scheduling in _unilayer_schedule_list_v1:
                cost_t, loop_block_t, loop_order_t = \
                    scheduling(sublayer, actual_s, loop_lower_bound)

                if cost_t < float('inf'):
                    blocking_t = [loop_block_t, loop_block, layer.dimension]

                    loop_order_innermost = [le.NUM - 1] * le.NUM
                    non_max_block = [i for i, e in enumerate(loop_block_t) if (e != 1)]
                    order = 0
                    for i in non_max_block:
                        loop_order_innermost[i] = order
                        order += 1

                    ordering_t = [loop_order_innermost, loop_order_t, loop_order]

                    # TODO support unroll in other levels
                    innermost_partition = [1] * le.NUM
                    innermost_para_dim = []
                    for i, lp in enumerate(self.loop_lower_bound.unroll_loop):
                        innermost_para_dim.append([lp])
                        partition_size = self.loop_lower_bound.loop_lower_bound_init[lp]
                        if loop_block_t[lp] >= partition_size:
                            innermost_partition[lp] = partition_size
                        else:
                            innermost_partition[lp] = loop_block_t[lp]
                            if self.resource.replication and self.loop_lower_bound.replication_loop is not None:
                                para = partition_size // loop_block_t[lp]
                                replp = self.loop_lower_bound.replication_loop[i]
                                if para > 1 and loop_block_t[replp] > 1:
                                    innermost_para_dim[i].append(replp)
                                    innermost_partition[replp] = para \
                                        if loop_block_t[replp] > para else loop_block_t[replp]

                    para_loop_dim_list = [innermost_para_dim, [], []]
                    partition_t = [innermost_partition, [1] * le.NUM, [1] * le.NUM]

                    point_t = MappingPoint(list(zip(*ordering_t)), list(zip(*blocking_t)),
                                           list(zip(*partition_t)), para_loop_dim_list)
                    buffer_access = self.cost_model.get_level_access([point_t], [layer_name], 1)
                    cost_t = math.ceil(reduce(add, buffer_access, 0))
                if cost_t < cost_inner:
                    point_g[t] = point_t
                    cost_inner_g[t] = cost_t
                    cost_inner = cost_t
            t += 1

        return cost_inner_g, point_g

    def res_parse_timeloop(self, section_info):

        off_chip_overall = 0
        total_cost = 0
        costs = [0, 0, 0, 0, 0]
        ifmap, ofmap, filter = 0, 0, 0

        utilization = 0
        total_ops = 0
        alu = 0
        
        # for section_info in schedule_info_list:
        fusion_group, loop_blocking_list, loop_ordering_list, off_chip_access_test, is_filter_fit = section_info
        cost_inner_list, point_list = self.mapping_timeloop(fusion_group, loop_blocking_list, loop_ordering_list)

        access_list, levels_cost, noc_cost, ops, cost = self.cost_model.get_cost(point_list, fusion_group, is_filter_fit)
        total_cost += cost

        num_levels = self.resource.buffer_levels()

        for i in range(num_levels):
            costs[i] += sum(levels_cost[i])
        costs[3] += noc_cost
        costs[4] += ops

        off_chip_access_breakdown, off_chip_access \
            = access_list[2], math.ceil(reduce(add, access_list[2], 0))
        ifmap += off_chip_access_breakdown[0]
        ofmap += off_chip_access_breakdown[1]
        filter += off_chip_access_breakdown[2]
        off_chip_overall += off_chip_access

        utilization_p = 0
        total_ops_p = 0
        for layer, point in zip(fusion_group, point_list):
            if point:
                x = point.para_loop_dim[0][0]
                y = point.para_loop_dim[0][1]
                xx = 1
                yy = 1
                for idx in x:
                    xx *= point.loop_partitionings[idx][0]
                for idx in y:
                    yy *= point.loop_partitionings[idx][0]
                utilization += xx * yy * self.network[layer].total_ops / self.resource.para_count_list[0]
                utilization_p += xx * yy * self.network[layer].total_ops / self.resource.para_count_list[0]
                total_ops += self.network[layer].total_ops
                total_ops_p += self.network[layer].total_ops
        if total_ops_p:
            u = utilization_p / total_ops_p
            alu += total_ops_p / u

        access = off_chip_overall * self.resource.precision / 8 / 1024 / 1024
        energy = total_cost
        return energy, access
    
    def timeloop_model_uni_layer(self, g, layer, mapping_t, blocking_t, ordering_t, innermost_partition):
        loop_block_t, loop_block, loop_block_outer = blocking_t
        loop_order_innermost, loop_order_t, loop_order = ordering_t
        
        Total_Energy, DRAM_Energy, MAC_Util, EnergyperMAC, Ideal_Cycle, Slowdown, Total_Cycle_Timeloop = 0,0,0,0,0,0,0
        
        path1 = self.path + "/" + self.dataflow
        if not os.path.exists(path1):
            os.makedirs(path1)
        os.chdir(path1)

        path2 = './' + g + "/"+self.mapping_t
        
        if os.path.exists(path2):
            shutil.rmtree(path2)
        os.makedirs(path2)
        os.chdir(path2)
        
        loop_block_PE_SPAD = [1] * le.NUM
        loop_block_Buffer_Spatial = [1] * le.NUM
        loop_block_Buffer_Temporal = [1] * le.NUM
        loop_block_DRAM = [1] * le.NUM
        
        loop_order_PE_SPAD = [le.NUM - 1] * le.NUM
        loop_order_Buffer_Spatial = [le.NUM - 1] * le.NUM
        loop_order_Buffer_Temporal = [le.NUM - 1] * le.NUM
        loop_order_DRAM = [le.NUM - 1] * le.NUM
        
        for i, v in enumerate(loop_block_PE_SPAD):
            loop_block_PE_SPAD[i] = loop_block_t[i]//innermost_partition[i]
            loop_block_Buffer_Spatial[i] = innermost_partition[i]
            loop_block_Buffer_Temporal[i] = loop_block[i]//loop_block_t[i]
            loop_block_DRAM[i] = layer.dimension[i]//loop_block[i]
        
        order_PE_SPAD = 0
        order_Buffer_Spatial = 0
        for i, v in enumerate(loop_order_PE_SPAD):
            if loop_block_PE_SPAD[i] == 1 :
                loop_order_PE_SPAD[i] = le.NUM - 1
            else :
                loop_order_PE_SPAD[i] = order_PE_SPAD
                order_PE_SPAD += 1
            if loop_block_Buffer_Spatial[i] == 1 :
                loop_order_Buffer_Spatial[i] = le.NUM - 1
            else :
                loop_order_Buffer_Spatial[i] = order_Buffer_Spatial
                order_Buffer_Spatial += 1
                
            loop_order_Buffer_Temporal[i] = loop_order_t[i]
            loop_order_DRAM[i] = loop_order[i]
                        
        loop_block_by_level = [loop_block_PE_SPAD] + [loop_block_Buffer_Spatial] + [loop_block_Buffer_Temporal] + [loop_block_DRAM]
        loop_order_by_level = [loop_order_PE_SPAD] + [loop_order_Buffer_Spatial] + [loop_order_Buffer_Temporal] + [loop_order_DRAM]
        
        
        factor_string = [None]* 4
        permutation_string = [None] * 4
        
        notation = ['R','S','C','Q','P','M','N']
        for level in range(0,4) :
            factor_string_t = [None] * le.NUM
            permutation_string_t = [None] * le.NUM
            
            for loopblock, looporder, loopnotation in zip(loop_block_by_level[level], loop_order_by_level[level], notation) :
                
                if factor_string_t[looporder] == None : 
                    if looporder == 6 :
                        factor_string_t[looporder] = ' {}{} '.format(loopnotation,loopblock)
                    else :
                        factor_string_t[looporder] = '{}{} '.format(loopnotation,loopblock)
                else :
                    if loopnotation == 'N' :
                        factor_string_t.append('{}{}'.format(loopnotation,loopblock))
                    else :
                        factor_string_t.append('{}{} '.format(loopnotation,loopblock))
                        
                if permutation_string_t[looporder] == None : 
                    if looporder == 6 :
                        permutation_string_t[looporder] = " " + loopnotation
                    else :
                        permutation_string_t[looporder] = loopnotation
                else : 
                    permutation_string_t.append(loopnotation)
            
            factor_string[level] = ''.join([i for i in factor_string_t if i not in [None]])
            permutation_string[level] = ''.join([i for i in permutation_string_t if i not in [None]])
        
        print("     ",loop_block_t, innermost_partition, loop_block, loop_block_outer)
        print("     ",loop_block_PE_SPAD,loop_block_Buffer_Spatial,loop_block_Buffer_Temporal,loop_block_DRAM)
        print("     ",loop_order_PE_SPAD,loop_order_Buffer_Spatial,loop_order_Buffer_Temporal,loop_order_DRAM)
        print("     ",factor_string)
        print("     ",permutation_string)

        
        Actual_BW = self.DRAM_BW
        Ideal_BW = 128
        DRAM_Type , DRAM_BW = 'LPDDR4' , Actual_BW

        Input_Size = loop_block[2] * (layer.wstd*(loop_block[3]-1)+loop_block[0]) * (layer.hstd*(loop_block[4]-1)+loop_block[1]) * loop_block[6]//64 + 1
        Output_Size = loop_block[3]*loop_block[4]*loop_block[5]*loop_block[6]//64 + 1
        Weight_Size = loop_block[0]*loop_block[1]*loop_block[2]*loop_block[5]//64 + 1
        
        archi_data = {'architecture': {'version': 0.3,  
                        'subtree': [{'name': 'system', 'local': [
                            {'name': 'DRAM', 'class': 'DRAM', 'attributes': {'type': DRAM_Type, 'block-size': 64, 'width': 512, 'word-bits': 8, 'read_bandwidth': DRAM_BW, 'write_bandwidth': DRAM_BW}}],
                        'subtree': [{'name': 'Chip', 'attributes': {'technology': '45nm'}, 'local': [
                            {'name': 'Input_Buffer', 'class': 'smartbuffer_SRAM', 'attributes': {'word-bits': 8, 'block-size': 64, 'memory_width': 512, 'memory_depth': Input_Size, 'read_bandwidth': 32, 'write_bandwidth': 32, 'multiple-buffering': 1}}, 
                            {'name': 'Output_Buffer', 'class': 'smartbuffer_SRAM', 'attributes': {'word-bits': 8, 'block-size': 64, 'memory_width': 512, 'memory_depth': Output_Size, 'read_bandwidth': 32, 'write_bandwidth': 32, 'multiple-buffering': 1}}, 
                            {'name': 'Weight_Buffer', 'class': 'smartbuffer_SRAM', 'attributes': {'word-bits': 8, 'block-size': 64, 'memory_width': 512, 'memory_depth': Weight_Size, 'read_bandwidth': 32, 'write_bandwidth': 32, 'multiple-buffering': 1}}, 
                            {'name': 'NoC', 'class': 'SRAM', 'attributes': {'block-size': 1, 'width': 8, 'depth': 0, 'word-bits': 8}}],
                        'subtree': [{'name': 'PE[0..1023]', 'local': [
                            {'name': 'PE_SPAD', 'class': 'smartbuffer_RF', 'attributes': {'block-size': 1, 'memory_depth': 1, 'memory_width': 8, 'meshX': 32, 'word-bits': 8}},
                            {'name': 'weight_reg', 'class': 'reg_storage', 'attributes': {'block-size': 1, 'depth': 1, 'meshX': 32, 'width': 8, 'word-bits': 8}},
                            {'name': 'input_activation_reg', 'class': 'reg_storage', 'attributes': {'block-size': 1, 'depth': 1, 'meshX': 32, 'width': 8, 'word-bits': 8}},
                            {'name': 'output_activation_reg', 'class': 'reg_storage', 'attributes': {'block-size': 1, 'depth': 1, 'meshX': 32, 'width': 8, 'word-bits': 8}},
                            {'name': 'mac', 'class': 'intmac', 'attributes': {'meshX': 32, 'word-bits': 8, 'dataflow' : self.dataflow}}]}]}]}]}}
        
        R,S,C,Q,P,M,N = loop_block_outer[0],loop_block_outer[1],loop_block_outer[2],loop_block_outer[3],loop_block_outer[4],loop_block_outer[5],loop_block_outer[6]
        probelm_data = {'problem': {'instance': {'R': R, 'S': S, 'C': C, 'Q': Q, 'P': P, 'M': M, 'N': N, 'Hdilation': 1, 'Hstride': layer.hstd, 'Wdilation': 1, 'Wstride': layer.wstd}, 
                            'shape': {'coefficients': [{'default': 1, 'name': 'Wstride'}, {'default': 1, 'name': 'Hstride'}, {'default': 1, 'name': 'Wdilation'}, {'default': 1, 'name': 'Hdilation'}], 
                            'data-spaces': [{'name': 'Weights', 'projection': [[['C']], [['M']], [['R']], [['S']]]}, 
                            {'name': 'Inputs', 'projection': [[['N']], [['C']], [['R', 'Wdilation'], ['Q', 'Wstride']], [['S', 'Hdilation'], ['P', 'Hstride']]]}, 
                            {'name': 'Outputs', 'projection': [[['N']], [['M']], [['Q']], [['P']]], 'read-write': True}], 'dimensions': ['C', 'M', 'R', 'S', 'N', 'P', 'Q'], 'name': 'Convolution'}}} 
        
        if self.dataflow == 'WS' :            
            mapping_data = {'mapping': [
                            {'bypass': ['Weights', 'Inputs'], 'keep': ['Outputs'], 'target': 'output_activation_reg', 'type': 'datatype'}, 
                            {'bypass': ['Weights', 'Outputs'], 'keep': ['Inputs'], 'target': 'input_activation_reg', 'type': 'datatype'}, 
                            {'bypass': ['Inputs', 'Outputs'], 'keep': ['Weights'], 'target': 'weight_reg', 'type': 'datatype'}, 
                            {'bypass': ['Inputs', 'Outputs'], 'keep': ['Weights'], 'target': 'PE_SPAD', 'type': 'datatype'},
                            {'bypass': ['Weights', 'Inputs', 'Outputs'], 'keep': [], 'target': 'NoC', 'type': 'datatype'},  
                            {'bypass': ['Inputs', 'Outputs'], 'keep': ['Weights'], 'target': 'Weight_Buffer', 'type': 'datatype'}, 
                            {'bypass': ['Weights', 'Inputs'], 'keep': ['Outputs'], 'target': 'Output_Buffer', 'type': 'datatype'}, 
                            {'bypass': ['Weights', 'Outputs'], 'keep': ['Inputs'], 'target': 'Input_Buffer', 'type': 'datatype'}, 
                            {'bypass': [], 'keep': ['Weights', 'Inputs', 'Outputs'], 'target': 'DRAM', 'type': 'datatype'},
                            
                            {'target': 'output_activation_reg', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'input_activation_reg', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'weight_reg', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'PE_SPAD', 'type': 'temporal', 'factors': factor_string[0], 'permutation': permutation_string[0]}, 
                            {'target': 'NoC', 'type': 'spatial', 'factors': factor_string[1], 'permutation': permutation_string[1], 'split': 1}, 
                            {'target': 'NoC', 'type': 'temporal', 'factors': factor_string[2], 'permutation': permutation_string[2]}, 
                            {'target': 'Weight_Buffer', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'Output_Buffer', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'Input_Buffer', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'DRAM', 'type': 'temporal', 'factors': factor_string[3], 'permutation': permutation_string[3]}]}

        elif self.dataflow == 'OS' :
            mapping_data = {'mapping': [
                            {'bypass': ['Weights', 'Inputs'], 'keep': ['Outputs'], 'target': 'output_activation_reg', 'type': 'datatype'}, 
                            {'bypass': ['Weights', 'Outputs'], 'keep': ['Inputs'], 'target': 'input_activation_reg', 'type': 'datatype'}, 
                            {'bypass': ['Inputs', 'Outputs'], 'keep': ['Weights'], 'target': 'weight_reg', 'type': 'datatype'}, 
                            {'bypass': ['Inputs', 'Weights'], 'keep': ['Outputs'], 'target': 'PE_SPAD', 'type': 'datatype'},
                            {'bypass': ['Weights', 'Inputs', 'Outputs'], 'keep': [], 'target': 'NoC', 'type': 'datatype'},  
                            {'bypass': ['Inputs', 'Outputs'], 'keep': ['Weights'], 'target': 'Weight_Buffer', 'type': 'datatype'}, 
                            {'bypass': ['Weights', 'Inputs'], 'keep': ['Outputs'], 'target': 'Output_Buffer', 'type': 'datatype'}, 
                            {'bypass': ['Weights', 'Outputs'], 'keep': ['Inputs'], 'target': 'Input_Buffer', 'type': 'datatype'}, 
                            {'bypass': [], 'keep': ['Weights', 'Inputs', 'Outputs'], 'target': 'DRAM', 'type': 'datatype'},
                            
                            {'target': 'output_activation_reg', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'input_activation_reg', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'weight_reg', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'PE_SPAD', 'type': 'temporal', 'factors': factor_string[0], 'permutation': permutation_string[0]}, 
                            {'target': 'NoC', 'type': 'spatial', 'factors': factor_string[1], 'permutation': permutation_string[1], 'split': 1}, 
                            {'target': 'NoC', 'type': 'temporal', 'factors': factor_string[2], 'permutation': permutation_string[2]}, 
                            {'target': 'Weight_Buffer', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'Output_Buffer', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'Input_Buffer', 'type': 'temporal', 'factors': 'C1 M1 R1 S1 N1 P1 Q1', 'permutation': 'CMRSNPQ'}, 
                            {'target': 'DRAM', 'type': 'temporal', 'factors': factor_string[3], 'permutation': permutation_string[3]}]}
            
        compo_data = {'compound_components': {'version': 0.3, 'classes': [{'name': 'smartbuffer_SRAM', 'attributes': {'technology': '45nm', 'memory_depth': 1024, 'memory_width': 256, 'n_rdwr_ports': 2, 'n_banks': 1, 'n_buffets': 1}, 'subcomponents': [{'name': 'storage', 'class': 'SRAM', 'attributes': {'technology': 'technology', 'width': 'memory_width', 'depth': 'memory_depth', 'n_rdwr_ports': 'n_rdwr_ports', 'n_banks': 'n_banks'}}, {'name': 'address_generators[0..1]', 'class': 'intadder', 'attributes': {'technology': 'technology', 'width': 'log(memory_depth)'}}], 'actions': [{'name': 'write', 'arguments': {'data_delta': '0..1', 'address_delta': '0..n_banks'}, 'subcomponents': [{'name': 'storage', 'actions': [{'name': 'write', 'arguments': {'data_delta': 'data_delta', 'address_delta': 'address_delta'}}]}, {'name': 'address_generators[0]', 'actions': [{'name': 'count'}]}, {'name': 'address_generators[1]', 'actions': [{'name': 'idle'}]}]}, {'name': 'read', 'arguments': {'data_delta': '0..1', 'address_delta': '0..n_banks'}, 'subcomponents': [{'name': 'storage', 'actions': [{'name': 'read', 'arguments': {'data_delta': 'data_delta', 'address_delta': 'address_delta'}}]}, {'name': 'address_generators[1]', 'actions': [{'name': 'add'}]}, {'name': 'address_generators[0]', 'actions': [{'name': 'idle'}]}]}, {'name': 'idle', 'subcomponents': [{'name': 'storage', 'actions': [{'name': 'idle'}]}, {'name': 'address_generators[0..1]', 'actions': [{'name': 'idle'}]}]}]}, {'name': 'smartbuffer_RF', 'attributes': {'technology': '45nm', 'memory_depth': 'memory_depth', 'memory_width': 'memory_width', 'n_rdwr_ports': 2, 'n_banks': 1, 'n_buffets': 1}, 'subcomponents': [{'name': 'storage', 'class': 'regfile', 'attributes': {'technology': 'technology', 'width': 'memory_width', 'depth': 'memory_depth', 'n_rdwr_ports': 'n_rdwr_ports', 'n_banks': 'n_banks'}}, {'name': 'address_generators[0..1]', 'class': 'intadder', 'attributes': {'technology': 'technology', 'width': 'log(memory_depth)'}}], 'actions': [{'name': 'write', 'arguments': {'data_delta': '0..1', 'address_delta': '0..n_banks'}, 'subcomponents': [{'name': 'storage', 'actions': [{'name': 'write', 'arguments': {'data_delta': 'data_delta', 'address_delta': 'address_delta'}}]}, {'name': 'address_generators[0]', 'actions': [{'name': 'add'}]}, {'name': 'address_generators[1]', 'actions': [{'name': 'idle'}]}]}, {'name': 'read', 'arguments': {'data_delta': '0..1', 'address_delta': '0..n_banks'}, 'subcomponents': [{'name': 'storage', 'actions': [{'name': 'read', 'arguments': {'data_delta': 'data_delta', 'address_delta': 'address_delta'}}]}, {'name': 'address_generators[1]', 'actions': [{'name': 'add'}]}, {'name': 'address_generators[0]', 'actions': [{'name': 'idle'}]}]}, {'name': 'idle', 'subcomponents': [{'name': 'storage', 'actions': [{'name': 'idle'}]}, {'name': 'address_generators[0..1]', 'actions': [{'name': 'idle'}]}]}]}, {'name': 'reg_storage', 'attributes': {'technology': '45nm', 'latency': '1ns', 'width': 8, 'depth': 1}, 'subcomponents': [{'name': 'single_register', 'class': 'reg', 'attributes': {'technology': 'technology', 'latency': 'latency', 'datawidth': 'width'}}], 'actions': [{'name': 'read', 'subcomponents': [{'name': 'single_register', 'actions': [{'name': 'read'}]}]}, {'name': 'write', 'subcomponents': [{'name': 'single_register', 'actions': [{'name': 'write'}]}]}, {'name': 'idle', 'subcomponents': [{'name': 'single_register', 'actions': [{'name': 'idle'}]}]}]}]}}
     
       
        archi_yaml = g + '_archi.yaml'
        mapping_yaml = g + "_" + self.mapping_t+'_mapping.yaml'
        problem_yaml = g + "_" + 'problem.yaml'
        component_yaml = 'components.yaml'
        
        with open(problem_yaml, "w") as file:
            yaml.dump(probelm_data, file)
        with open(mapping_yaml, "w") as file:
            yaml.dump(mapping_data, file)
        with open(archi_yaml, "w") as file:
            yaml.dump(archi_data, file)
        with open(component_yaml, "w") as file:
            yaml.dump(compo_data, file)
            
        subprocess.call(['timeloop-model', archi_yaml, component_yaml, problem_yaml, mapping_yaml])
        stats = parse_timeloop_output.parse_timeloop_stats(os.getcwd())
        Total_Cycle_Timeloop = stats['actual_cycles']
        
        # Ideal_Cycle = stats['cycles']
        # Slowdown = round(stats['slowdown'],3)
        
        Ideal_Cycle = layer.total_ops//(self.resource.array_width*self.resource.array_width)
        Slowdown = round(Ideal_Cycle/Total_Cycle_Timeloop,3)
        
        Total_Energy = int(stats['energy_pJ'])
        EnergyperMAC = 0

        print("    ", g ,"- Timeloop Result :", Total_Cycle_Timeloop, stats['cycles'], Slowdown, Total_Energy)
        os.chdir('/home/min/workspace/ASD/ASD_Layer_Fusion_Optimization')
        
        return Total_Cycle_Timeloop, Ideal_Cycle, Slowdown, Total_Energy, EnergyperMAC
        
def _bounded_factor(n, start, end):
    # TODO: start bound
    f = []
    for i in range(1, min(int(n ** 0.5) + 1, end + 1)):
        if n % i == 0 and n // i <= end and i >= start and (n // i) >= start:
            f.__iadd__([i, int(n // i)])
        elif n % i == 0 and n // i <= end and (n // i) >= start:
            f.__iadd__([int(n // i)])
        elif n % i == 0 and i >= start:
            f.__iadd__([i])

    return set(f)


def _bhw_factorization(layer, n, loop_lower_bound):
    b = h = w = 0
    while n > 0 and b * h * w == 0:
        for bb in _bounded_factor(n, loop_lower_bound.b, layer.nimg):
            for ww in _bounded_factor(n / bb, loop_lower_bound.w, layer.wofm):
                if n / (bb * ww) <= layer.hofm:
                    b, w, h = bb, ww, n / (bb * ww)
                    break
                if b * h * w > 0:
                    break
            if b * h * w > 0:
                break
        n -= 1
    n += 1
    return int(n), int(b), int(h), int(w)


def loop_order_generator(layer, loop_block, irrelevant):
    loop_order = [le.NUM - 1] * le.NUM

    order = 0
    for i in irrelevant:
        if loop_block[i] < layer.dimension[i]:
            loop_order[i] = order
            order += 1
    non_max_block = [i for i, e in enumerate(loop_block) if (e < layer.dimension[i] and i not in irrelevant)]
    for i in non_max_block:
        loop_order[i] = order
        order += 1

    return loop_order


def blocking_upper_bound(layer, block, q, resource, loop_lower_bound):

    k, c, bhw, r, d = block
    q0, q1, q2 = q
    s = resource.buffer(1).capacity

    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w

    if k > layer.nofm and c > layer.nifm and bhw > bhw_upper_bound:
        k = layer.nofm
        c = layer.nifm
        bhw = bhw_upper_bound
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif k > layer.nofm and c > layer.nifm:
        k = layer.nofm
        c = layer.nifm
        bhw = min((s - r*d*c*k)/(c*layer.hstd*layer.wstd + k), bhw_upper_bound)
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif k > layer.nofm and bhw > bhw_upper_bound:
        k = layer.nofm
        bhw = bhw_upper_bound
        c = min((s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k), layer.nifm)
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif c > layer.nifm and bhw > bhw_upper_bound:
        c = layer.nifm
        bhw = bhw_upper_bound
        k = min((s - bhw*c*layer.hstd*layer.wstd)/(bhw + r*d*c), layer.nofm)
        return math.floor(k), math.floor(c), math.floor(bhw)
    elif k > layer.nofm:
        k = layer.nofm
        a = (0.5 / (layer.hstd * layer.wstd)) * (k + r * d * k * q1 / q2)
        if loop_lower_bound.c > 1:
            c = max(min(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a, layer.nifm),
                    loop_lower_bound.c)
            bhw = min((s - r * d * c * k) / (c * layer.hstd * layer.wstd + k), bhw_upper_bound)
        else:
            bhw = max(min(q2*(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a)/q1, bhw_upper_bound),
                      bhw_lower_bound)
            c = min((s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k), layer.nifm)

        return math.floor(k), math.floor(c), math.floor(bhw)

    elif c > layer.nifm:
        c = layer.nifm
        a = 0.5 * (c * layer.hstd * layer.wstd + r * d * c * q0 / q2)

        if loop_lower_bound.k > 1:
            k = max(min(math.sqrt(q0 * s / q2 + a ** 2) - a, layer.nofm), loop_lower_bound.k)
            bhw = min((s - r * d * c * k) / (c * layer.hstd * layer.wstd + k), bhw_upper_bound)
        else:
            bhw = max(min(q2*(math.sqrt(q0 * s / q2 + a ** 2) - a)/q0, bhw_upper_bound), bhw_lower_bound)
            k = min((s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c), layer.nofm)

        return math.floor(k), math.floor(c), math.floor(bhw)

    elif bhw > bhw_upper_bound:
        bhw = bhw_upper_bound
        a = (0.5/(r*d)) * (bhw*layer.hstd*layer.wstd + q0*bhw/q1)

        if loop_lower_bound.k > 1:
            k = max(min(math.sqrt(s * q0 / (q1 * r * d) + a ** 2) - a, layer.nofm), loop_lower_bound.k)
            c = min((s - k * bhw) / (bhw * layer.hstd * layer.wstd + r * d * k), layer.nifm)
        else:
            c = max(min(q1*(math.sqrt(s * q0 / (q1 * r * d) + a ** 2) - a)/q0, layer.nifm), loop_lower_bound.c)
            k = min((s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c), layer.nofm)
        return math.floor(k), math.floor(c), math.floor(bhw)

    return math.floor(k), math.floor(c), math.floor(bhw)


def blocking_lower_bound(layer, block, q, resource, loop_lower_bound):

    k, c, bhw, r, d = block
    q0, q1, q2 = q
    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    if k < loop_lower_bound.k and c < loop_lower_bound.c and bhw < bhw_lower_bound:
        return k, c, bhw
    elif k < loop_lower_bound.k and c < loop_lower_bound.c:
        k = loop_lower_bound.k
        c = loop_lower_bound.c
        bhw = (s - r*d*c*k)/(c*layer.hstd*layer.wstd + k)
        return k, c, bhw
    elif k < loop_lower_bound.k and bhw < bhw_lower_bound:
        k = loop_lower_bound.k
        bhw = bhw_lower_bound
        c = (s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k)
        return k, c, bhw
    elif c < loop_lower_bound.c and bhw < bhw_lower_bound:
        c = loop_lower_bound.c
        bhw = bhw_lower_bound
        k = (s - bhw*c*layer.hstd*layer.wstd)/(bhw + r*d*c)
        return k, c, bhw
    elif k < loop_lower_bound.k:
        k = loop_lower_bound.k
        a = (0.5 / (layer.hstd * layer.wstd)) * (k + r * d * k * q1 / q2)
        if loop_lower_bound.c > 1:
            c = max(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a, loop_lower_bound.c)
            bhw = (s - r * d * c * k) / (c * layer.hstd * layer.wstd + k)
        else:
            bhw = max(q2*(math.sqrt(s * q1 / (q2 * layer.hstd * layer.wstd) + a ** 2) - a)/q1, bhw_lower_bound)
            c = (s - k*bhw)/(bhw*layer.hstd*layer.wstd + r*d*k)
        return k, c, bhw
    elif c < loop_lower_bound.c:
        c = loop_lower_bound.c
        a = 0.5 * (c * layer.hstd * layer.wstd + r * d * c * q0 / q2)

        if loop_lower_bound.k > 1:
            k = max(math.sqrt(q0 * s / q2 + a ** 2) - a, loop_lower_bound.k)
            bhw = (s - r * d * c * k) / (c * layer.hstd * layer.wstd + k)
        else:
            bhw = max(q2*(math.sqrt(q0 * s / q2 + a ** 2) - a)/q0, bhw_lower_bound)
            k = (s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c)
        return k, c, bhw
    elif bhw < bhw_lower_bound:
        bhw = bhw_lower_bound
        a = (0.5/(r*d)) * (bhw*layer.hstd*layer.wstd + q0*bhw/q1)
        k = math.sqrt(s*q0/(q1*r*d) + a**2) - a
        c = q1*k / q0

        return k, c, bhw
    return k, c, bhw


# def _psumsr_v2(layer, resource, loop_lower_bound):
#     irrelevant = [le.C]

#     p2 = resource.access_cost[2]
#     p1 = resource.access_cost[1]
#     r = layer.hfil
#     d = layer.wfil
#     q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
#     q1 = 2 * p1[1] / (r * d)
#     q2 = (p2[2] + p1[2])
#     s = resource.buffer(1).capacity

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
#     args = (q0, q1, q2)
#     cons_args = (loop_lower_bound.k, layer.nofm,
#                  loop_lower_bound.c, layer.nifm,
#                  bhw_lower_bound, bhw_upper_bound,
#                  layer.hfil*layer.wfil, 1, layer.hstd*layer.wstd, s)
#     # constrain
#     cons = con(cons_args)

#     # init
#     q = min([q0, q1, q2])/2
#     x0 = np.asarray((q0/q, q1/q, q2/q))

#     # optimize
#     res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

#     k, c, bhw = res.x
#     k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
#     bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if not res.success or any(i <= 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order, \
#                [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
#     else:
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size

#     if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
#     fi_glb_access_cost = p1[2] * layer.total_ops / bhw
#     if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_dram_access_cost = p2[1] * layer.total_ofmap_size
#     fi_dram_access_cost = p2[2] * layer.total_ops / bhw
#     glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
#     dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

#     return q, loop_block, loop_order, glb_access_cost, dram_access_cost


# def _filterr_v2(layer, resource, loop_lower_bound):
#     irrelevant = [le.W, le.H, le.B]

#     p2 = resource.access_cost[2]
#     p1 = resource.access_cost[1]
#     r = layer.hfil
#     d = layer.wfil
#     q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
#     q1 = 2 * (p2[1] + p1[1]) / (r * d)
#     q2 = p1[2]
#     s = resource.buffer(1).capacity

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
#     args = (q0, q1, q2)
#     cons_args = (loop_lower_bound.k, layer.nofm,
#                  loop_lower_bound.c, layer.nifm,
#                  bhw_lower_bound, bhw_upper_bound,
#                  r*d, 1, layer.hstd*layer.wstd, s)
#     # constrain
#     cons = con(cons_args)

#     # init
#     q = min([q0, q1, q2])/2
#     x0 = np.asarray((q0/q, q1/q, q2/q))

#     # optimize
#     res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

#     k, c, bhw = res.x
#     k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
#     bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if not res.success or any(i <= 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order, \
#                [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
#     else:
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[2] * layer.total_filter_size \
#         - (p2[1] + p1[1]) * layer.total_ofmap_size

#     if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
#     fi_glb_access_cost = p1[2] * layer.total_ops / bhw
#     if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
#     fi_dram_access_cost = p2[2] * layer.total_filter_size
#     glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
#     dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

#     return q, loop_block, loop_order, glb_access_cost, dram_access_cost


# def _ifmapr_v2(layer, resource, loop_lower_bound):
#     # irrelevant loop: k
#     irrelevant = [le.K]

#     p2 = resource.access_cost[2]
#     p1 = resource.access_cost[1]
#     r = layer.hfil
#     d = layer.wfil
#     q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
#     q1 = 2 * (p2[1] + p1[1]) / (r * d)
#     q2 = p2[2] + p1[2]
#     s = resource.buffer(1).capacity

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
#     args = (q0, q1, q2)
#     cons_args = (loop_lower_bound.k, layer.nofm,
#                  loop_lower_bound.c, layer.nifm,
#                  bhw_lower_bound, bhw_upper_bound,
#                  r*d, 1, layer.hstd*layer.wstd, s)
#     # constrain
#     cons = con(cons_args)

#     # init
#     q = min([q0, q1, q2])/2
#     x0 = np.asarray((q0/q, q1/q, q2/q))

#     # optimize
#     res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

#     k, c, bhw = res.x
#     k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
#     bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if not res.success or any(i <= 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order, \
#                [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
#     else:
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[0] * layer.total_ifmap_size \
#         - (p2[1] + p1[1]) * layer.total_ofmap_size

#     if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
#     fi_glb_access_cost = p1[2] * layer.total_ops / bhw
#     if_dram_access_cost = p2[0] * layer.total_ifmap_size
#     of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
#     fi_dram_access_cost = p2[2] * layer.total_ops / bhw
#     glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
#     dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

#     return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _cwr_c_v2(layer, resource, loop_lower_bound):
    # irrelevant loop: k r d
    irrelevant = [le.D, le.R, le.K]
    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = p1[2] + p2[2]

    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (loop_lower_bound.k, layer.nofm,
                 layer.nifm, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 r*d, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops \
        + p2[0] * layer.total_ifmap_size + (p2[1] - p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _cwr_k_v2(layer, resource, loop_lower_bound):
    # irrelevant loop: c r d
    irrelevant = [le.D, le.R, le.C]
    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = p1[2] + p2[2]

    s = resource.buffer(1).capacity

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm
    args = (q0, q1, q2)
    cons_args = (layer.nofm, layer.nofm,
                 loop_lower_bound.c, layer.nifm,
                 bhw_lower_bound, bhw_upper_bound,
                 r*d, 1, layer.hstd*layer.wstd, s)
    # constrain
    cons = con(cons_args)

    # init
    q = min([q0, q1, q2])/2
    x0 = np.asarray((q0/q, q1/q, q2/q))

    # optimize
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)

    k, c, bhw = res.x
    k, c, bhw = math.floor(k), math.floor(c), math.floor(bhw)
    bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if not res.success or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, \
               [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops \
        + p2[0] * layer.total_ifmap_size + (p2[1] - p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost

def _cwr_c_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: k r d
    irrelevant = [le.D, le.R, le.K]
    c = layer.nifm
    k = loop_lower_bound.k
    r = layer.hfil
    d = layer.wfil

    s = capacity - k * c * r * d
    rho = min(math.ceil(s / (c * layer.hstd * layer.wstd + k)), layer.nimg * layer.hofm * layer.wofm)
    rho, b, h, w = _bhw_factorization(layer, rho, loop_lower_bound)

    k = min(math.ceil((capacity - b*h*w*c*layer.hstd*layer.wstd) / (b*h*w + r*d*c)), layer.nofm)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if s < 0 or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = layer.total_ifmap_size + math.ceil(layer.total_ops / rho) + layer.total_ofmap_size

    return q, loop_block, loop_order


def _cwr_k_v1(layer, capacity, loop_lower_bound):

    # irrelevant loop: c r d
    irrelevant = [le.D, le.R, le.C]
    k = layer.nofm
    c = loop_lower_bound.c
    r = layer.hfil
    d = layer.wfil

    s = capacity - k * c * r * d

    rho = min(math.ceil(s / (c * layer.hstd * layer.wstd + k)), layer.nimg * layer.hofm * layer.wofm)
    rho, b, h, w = _bhw_factorization(layer, rho, loop_lower_bound)

    c = min(math.ceil((capacity - b * h * w * k) / (b * h * w * layer.hstd * layer.wstd + r * d * k)), layer.nifm)
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if s < 0 or any(i <= 0 for i in loop_block):
        return float('inf'), loop_block, loop_order
    else:
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = layer.total_ifmap_size + math.ceil(layer.total_ops / rho) + layer.total_ofmap_size

    return q, loop_block, loop_order

# def _psumsr_v1(layer, capacity, loop_lower_bound):

#     # irrelevant loop: c
#     irrelevant = [le.C]
#     c = loop_lower_bound.c
#     r = layer.hfil
#     d = layer.wfil
#     a = layer.hstd * layer.wstd * (c + 1) / 2
#     f = r * d / (layer.hstd * layer.wstd)
#     s = capacity

#     k = max(math.sqrt(s/f + a**2) - a, 1)
#     bhw = math.floor(k * r * d / a)
#     k = math.floor(k)

#     # upper bound
#     if k > layer.nofm and bhw > layer.nimg * layer.hofm * layer.wofm:
#         k = layer.nofm
#         bhw = layer.nimg * layer.hofm * layer.wofm
#     elif k > layer.nofm:
#         k = layer.nofm
#         bhw = min(math.floor((s-r*d*c*k)/(k+c*layer.hstd*layer.wstd)), layer.nimg * layer.hofm * layer.wofm)
#     elif bhw > layer.nimg * layer.hofm * layer.wofm:
#         bhw = layer.nimg * layer.hofm * layer.wofm
#         k = min(math.floor((s-bhw*c*layer.hstd*layer.wstd)/(bhw+r*d*c)), layer.nofm)

#     # lower bound
#     if k * bhw > loop_lower_bound.k * loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w:
#         if k < loop_lower_bound.k and bhw > loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w:
#             k = loop_lower_bound.k
#             bhw = min(math.floor((s - r * d * c * k) / (k + c * layer.hstd * layer.wstd)),
#                       layer.nimg * layer.hofm * layer.wofm)
#         elif k > loop_lower_bound.k and bhw < loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w:
#             bhw = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#             k = min(math.floor((s - bhw * c * layer.hstd * layer.wstd) / (bhw + r * d * c)), layer.nofm)

#     bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)

#     c = min(math.floor((s - bhw*k) / (bhw*layer.hstd*layer.wstd + r*d*k)), layer.nifm)

#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if any(i <= 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order
#     else:
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     rho = bhw * k * r * d / (bhw * layer.hstd * layer.wstd + k * r * d)
#     q = math.ceil(layer.total_ops / rho) + layer.total_ofmap_size

#     return q, loop_block, loop_order

# def _filterr_v1(layer, capacity, loop_lower_bound):

#     # irrelevant loop: b h w
#     irrelevant = [le.W, le.H, le.B]
#     b = loop_lower_bound.b
#     h = loop_lower_bound.h
#     w = loop_lower_bound.w
#     r = layer.hfil
#     d = layer.wfil

#     f = r * d / (layer.hstd * layer.wstd)
#     a = b * h * w * 3 / (4 * f)

#     s = capacity

#     k = max(math.sqrt(s / (2 * f) + a ** 2) - a, 1)
#     c = math.ceil(2 * k / (layer.hstd * layer.wstd))

#     # upper bound
#     if k > layer.nofm and c > layer.nifm:
#         k = layer.nofm
#         c = layer.nifm
#     elif k > layer.nofm:
#         k = layer.nofm
#         c = min(math.ceil((s-b*h*w*k)/(r*d*k+b*h*w*layer.hstd*layer.wstd)), layer.nifm)
#     elif c > layer.nifm:
#         c = layer.nifm
#         k = min(math.ceil((s-b*h*w*c*layer.hstd*layer.wstd)/(r*d*c+b*h*w)), layer.nofm)

#     # lower bound
#     if k * c > loop_lower_bound.k * loop_lower_bound.c:
#         if k < loop_lower_bound.k and c > loop_lower_bound.c:
#             k = loop_lower_bound.k
#             c = min(math.ceil((s - b * h * w * k) / (r * d * k + b * h * w * layer.hstd * layer.wstd)), layer.nifm)
#         elif k > loop_lower_bound.k and c < loop_lower_bound.c:
#             c = loop_lower_bound.c
#             k = min(math.ceil((s - b * h * w * c * layer.hstd * layer.wstd) / (r * d * c + b * h * w)), layer.nofm)

#     k = math.floor(k)

#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if any(i <= 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order
#     else:
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     rho = c * k * r * d / (c * layer.hstd * layer.wstd + (2 * k - 1))
#     q = layer.total_filter_size + math.ceil(layer.total_ops / rho)

#     return q, loop_block, loop_order

# def _ifmapr_v1(layer, capacity, loop_lower_bound):

#     # irrelevant loop: k
#     irrelevant = [le.K]
#     r = layer.hfil
#     d = layer.wfil
#     k = loop_lower_bound.k
#     f = r * d / (layer.hstd * layer.wstd)
#     a = k * 3 * f / 4

#     s = capacity

#     bhw = max(math.ceil(math.sqrt(s*f/2 + a**2) - a), 1)
#     c = math.ceil(2 * bhw / (r * d))

#     # upper bound
#     if bhw > layer.nimg * layer.hofm * layer.wofm and c > layer.nifm:
#         bhw = layer.nimg * layer.hofm * layer.wofm
#         c = layer.nifm
#     if bhw > layer.nimg * layer.hofm * layer.wofm:
#         bhw = layer.nimg * layer.hofm * layer.wofm
#         c = min(math.ceil((s-bhw*k)/(r*d*k+bhw*layer.hstd*layer.wstd)), layer.nifm)
#     elif c > layer.nifm:
#         c = layer.nifm
#         bhw = min(math.ceil((s-r*d*c*k)/(k+c*layer.hstd*layer.wstd)), layer.nimg * layer.hofm * layer.wofm)

#     # lower bound
#     if bhw * c > loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w * loop_lower_bound.c:
#         if bhw < loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w and c > loop_lower_bound.c:
#             bhw = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#             c = min(math.ceil((s - bhw * k) / (r * d * k + bhw * layer.hstd * layer.wstd)), layer.nifm)
#         elif bhw > loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w and c < loop_lower_bound.c:
#             c = loop_lower_bound.c
#             bhw = min(math.ceil((s - r * d * c * k) / (k + c * layer.hstd * layer.wstd)),
#                       layer.nimg * layer.hofm * layer.wofm)

#     bhw, b, h, w = _bhw_factorization(layer, bhw, loop_lower_bound)

#     k = min(math.ceil((s - b*h*w*c*layer.hstd*layer.wstd) / (b*h*w + r*d*c)), layer.nofm)
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if any(i <= 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order
#     else:
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     rho = bhw * c * r * d / (c * r * d + (2 * bhw - 1))
#     q = math.ceil(layer.total_ops / rho) + layer.total_ifmap_size

#     return q, loop_block, loop_order



def fun(args):
    q0, q1, q2 = args
    v = lambda x: q0 / x[0] + q1 / x[1] + q2 / x[2]
    return v


def con(args):
    x1min, x1max, x2min, x2max, x3min, x3max, a, b, c, xmax = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
            {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
            {'type': 'ineq', 'fun': lambda x: -x[2] + x3max},
            {'type': 'ineq', 'fun': lambda x: -(a*x[0]*x[1] + b*x[0]*x[2] + c*x[1]*x[2]) + xmax})
    return cons

def get_divisor(input,Array_Dim):
    data = set()

    for i in range(1, int(input ** (1 / 2)) + 1):
        if input % i == 0:
            data.add(Array_Dim*i)
            data.add(Array_Dim*input//i)
    return sorted(data)



def _filterr_v2(layer, resource, loop_lower_bound):
    irrelevant = [le.W, le.H, le.B]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * (p2[1] + p1[1]) / (r * d)
    q2 = p1[2]
    s = resource.buffer(1).capacity
    q = min([q0, q1, q2])/2
    
    c,w,h,k,b = 0,0,0,0,0
    hh, ww, bb = 1 , 1 , layer.nimg
    
    access = float('inf')

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

    # constrain & optimize
    for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
        for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
            max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
            if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
                for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
                    for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
                        for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
                        # ww = layer.wofm
                            if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
                                bhw_temp = bb*hh*ww
                                access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + p2[2] * layer.total_filter_size - (p2[1] + p1[1]) * layer.total_ofmap_size
                                if access_temp < access :
                                    access = access_temp
                                    k = kk
                                    c = cc
                                    b = bb
                                    h = hh
                                    w = ww
                                    bhw = bhw_temp
                                    
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i == 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else :
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[2] * layer.total_filter_size \
        - (p2[1] + p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_filter_size
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]
    return q, loop_block, loop_order, glb_access_cost, dram_access_cost

def _psumsr_v2(layer, resource, loop_lower_bound):
    irrelevant = [le.C]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = (p2[2] + p1[2])
    q = min([q0, q1, q2])/2
    s = resource.buffer(1).capacity

    c,w,h,k,b = 0,0,0,0,0
    hh, ww, bb = 1 , 1 , layer.nimg
    access = float('inf')
    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

    # constrain & optimize
    for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
        for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
    # for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
    #     for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
            max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
            if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
                for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
                    for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
                        for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
                        # ww = layer.wofm
                            if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
                                bhw_temp = bb*hh*ww
                                access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size
                                # print(cc,kk,hh,ww,bb,access_temp)
                                if access_temp < access :
                                    access = access_temp
                                    k = kk
                                    c = cc
                                    b = bb
                                    h = hh
                                    w = ww
                                    bhw = bhw_temp
                                    
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i == 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else :
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]
    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _ifmapr_v2(layer, resource, loop_lower_bound):
    # irrelevant loop: k
    irrelevant = [le.K]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * (p2[1] + p1[1]) / (r * d)
    q2 = (p2[2] + p1[2])
    s = resource.buffer(1).capacity
    q = min([q0, q1, q2])/2
    c,w,h,k,b = 0,0,0,0,0
    hh, ww, bb = 1 , 1 , layer.nimg
    access = float('inf')

    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

    # constrain & optimize
    for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
        for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
            max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
            if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
                for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
                    for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
                        for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
                            if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
                                bhw_temp = bb*hh*ww
                                access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + p2[0] * layer.total_ifmap_size - (p2[1] + p1[1]) * layer.total_ofmap_size
                                if access_temp < access :
                                    access = access_temp
                                    k = kk
                                    c = cc
                                    b = bb
                                    h = hh
                                    w = ww
                                    bhw = bhw_temp

    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i == 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else :
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[0] * layer.total_ifmap_size \
        - (p2[1] + p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost

def _filterr_v2_os(layer, resource, loop_lower_bound):
    irrelevant = [le.W, le.H, le.B]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * (p2[1] + p1[1]) / (r * d)
    q2 = p1[2]
    s = resource.buffer(1).capacity
    q = min([q0, q1, q2])/2
    
    c,w,h,k,b = 0,0,0,0,0
    hh, ww, bb = 1 , 1 , layer.nimg
    
    access = float('inf')
    
    loop_lower_bound_c = 32
    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

    # constrain & optimize
    for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
        for cc in get_divisor(layer.nifm//loop_lower_bound_c,loop_lower_bound_c) :
            max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
            if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
                for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
                    for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
                        for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
                        # ww = layer.wofm
                            if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
                                bhw_temp = bb*hh*ww
                                access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + p2[2] * layer.total_filter_size - (p2[1] + p1[1]) * layer.total_ofmap_size
                                if access_temp < access :
                                    access = access_temp
                                    k = kk
                                    c = cc
                                    b = bb
                                    h = hh
                                    w = ww
                                    bhw = bhw_temp
                                    
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i == 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else :
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[2] * layer.total_filter_size \
        - (p2[1] + p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_filter_size
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]
    return q, loop_block, loop_order, glb_access_cost, dram_access_cost

def _psumsr_v2_os(layer, resource, loop_lower_bound):
    irrelevant = [le.C]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * p1[1] / (r * d)
    q2 = (p2[2] + p1[2])
    q = min([q0, q1, q2])/2
    s = resource.buffer(1).capacity

    c,w,h,k,b = 0,0,0,0,0
    hh, ww, bb = 1 , 1 , layer.nimg
    access = float('inf')
    
    loop_lower_bound_c = 32
    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

    # constrain & optimize
    for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
        for cc in get_divisor(layer.nifm//loop_lower_bound_c,loop_lower_bound_c) :
    # for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
    #     for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
            max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
            if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
                for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
                    for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
                        for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
                        # ww = layer.wofm
                            if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
                                bhw_temp = bb*hh*ww
                                access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size
                                # print(cc,kk,hh,ww,bb,access_temp)
                                if access_temp < access :
                                    access = access_temp
                                    k = kk
                                    c = cc
                                    b = bb
                                    h = hh
                                    w = ww
                                    bhw = bhw_temp
                                    
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i == 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else :
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_dram_access_cost = p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]
    return q, loop_block, loop_order, glb_access_cost, dram_access_cost


def _ifmapr_v2_os(layer, resource, loop_lower_bound):
    # irrelevant loop: k
    irrelevant = [le.K]

    p2 = resource.access_cost[2]
    p1 = resource.access_cost[1]
    r = layer.hfil
    d = layer.wfil
    q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
    q1 = 2 * (p2[1] + p1[1]) / (r * d)
    q2 = (p2[2] + p1[2])
    s = resource.buffer(1).capacity
    q = min([q0, q1, q2])/2
    c,w,h,k,b = 0,0,0,0,0
    hh, ww, bb = 1 , 1 , layer.nimg
    access = float('inf')

    loop_lower_bound_c = 32
    bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
    bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

    # constrain & optimize
    for cc in get_divisor(layer.nifm//loop_lower_bound_c,loop_lower_bound_c) :
        for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
            max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
            if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
                for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
                    for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
                        for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
                            if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
                                bhw_temp = bb*hh*ww
                                access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + p2[0] * layer.total_ifmap_size - (p2[1] + p1[1]) * layer.total_ofmap_size
                                if access_temp < access :
                                    access = access_temp
                                    k = kk
                                    c = cc
                                    b = bb
                                    h = hh
                                    w = ww
                                    bhw = bhw_temp

    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    if any(i == 0 for i in loop_block):
        return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
    else :
        loop_order = loop_order_generator(layer, loop_block, irrelevant)

    q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[0] * layer.total_ifmap_size \
        - (p2[1] + p1[1]) * layer.total_ofmap_size

    if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
    of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
    fi_glb_access_cost = p1[2] * layer.total_ops / bhw
    if_dram_access_cost = p2[0] * layer.total_ifmap_size
    of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
    fi_dram_access_cost = p2[2] * layer.total_ops / bhw
    glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
    dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

    return q, loop_block, loop_order, glb_access_cost, dram_access_cost

# def _psumsr_v1_os(layer, capacity, loop_lower_bound):

#     # irrelevant loop: c
#     irrelevant = [le.C,le.D,le.R]
#     bb = 1
#     hh = 1
#     ww = 1
    
#     r = layer.hfil
#     d = layer.wfil
#     cc = layer.nifm
#     kk = loop_lower_bound.k
    
#     s = capacity
#     access = float('inf')
    
#     bhw_temp = bb*hh*ww
#     rho_temp = (bhw_temp * kk * r * d) / (bhw_temp * layer.hstd * layer.wstd + kk * r * d)
#     access_temp = math.ceil(layer.total_ops / rho_temp) + layer.total_ofmap_size
#     if access_temp < access :
#         access = access_temp
#         rho = rho_temp
#         k = kk
#         c = cc
#         b = bb
#         h = hh
#         w = ww
#         bhw = bhw_temp

#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     loop_order = loop_order_generator(layer, loop_block, irrelevant)
#     q = access
    
#     return q, loop_block, loop_order

def _psumsr_v1_os(layer, capacity, loop_lower_bound):

    # irrelevant loop: c,d,r
    irrelevant = [le.D, le.R, le.C]
    bb = 1
    hh = 1
    ww = loop_lower_bound.w
    
    r = layer.hfil
    d = layer.wfil
    cc = layer.nifm
    kk = loop_lower_bound.k
    
    s = capacity
    access = float('inf')
    
    bhw_temp = bb*hh*ww
    rho_temp = (bhw_temp * kk * r * d) / (bhw_temp * layer.hstd * layer.wstd + kk * r * d)
    access_temp = math.ceil(layer.total_ops / rho_temp) + layer.total_ofmap_size
    if access_temp < access :
        access = access_temp
        rho = rho_temp
        k = kk
        c = cc
        b = bb
        h = hh
        w = ww
        bhw = bhw_temp

    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    loop_order = loop_order_generator(layer, loop_block, irrelevant)
    q = access
    
    return q, loop_block, loop_order

def _filterr_v1_ws(layer, capacity, loop_lower_bound):

    # irrelevant loop: b h w
    irrelevant = [le.W, le.H, le.B]
    bb = layer.nimg
    hh = layer.hofm
    ww = layer.wofm
    
    r = 1
    d = 1
    cc = loop_lower_bound.c
    kk = loop_lower_bound.k
    
    s = capacity
    access = float('inf')
                    
    rho_temp = (cc * kk * r * d) / (cc * layer.hstd * layer.wstd + (2 * kk - 1))
    access_temp = layer.total_filter_size + math.ceil(layer.total_ops / rho_temp)
    if access_temp < access :
        access = access_temp
        rho = rho_temp
        k = kk
        c = cc
        b = bb
        h = hh
        w = ww
                    
    loop_block = [d, r, c, w, h, k, b]
    loop_order = [le.NUM - 1] * le.NUM
    loop_order = loop_order_generator(layer, loop_block, irrelevant)
    q = access

    return q, loop_block, loop_order

# def _psumsr_v1(layer, capacity, loop_lower_bound):

#     # irrelevant loop: c
#     irrelevant = [le.C]
#     c = loop_lower_bound.c
#     # r = layer.hfil
#     # d = layer.wfil
#     r = 1
#     d = 1
#     a = layer.hstd * layer.wstd * (c + 1) / 2
#     f = r * d / (layer.hstd * layer.wstd)
#     s = capacity

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

#     access = float('inf')

#     bb = 1
#     for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
#         for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
#             if (cc//loop_lower_bound.c)*(kk//loop_lower_bound.k)*r*d <= s :
#                 for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
#                     for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
#                         if (bb*hh*ww <= bhw_upper_bound) :
#                             bhw_temp = bb*hh*ww
#                             rho_temp = (bhw_temp * kk * r * d) / (bhw_temp * layer.hstd * layer.wstd + kk * r * d)
#                             access_temp = math.ceil(layer.total_ops / rho_temp) + layer.total_ofmap_size
#                             if access_temp < access :
#                                 access = access_temp
#                                 rho = rho_temp
#                                 k = kk
#                                 c = cc
#                                 b = bb
#                                 h = hh
#                                 w = ww
#                                 bhw = bhw_temp

#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     loop_order = loop_order_generator(layer, loop_block, irrelevant)
    
#     q = access
    
#     return q, loop_block, loop_order

# def _filterr_v1(layer, capacity, loop_lower_bound):

#     # irrelevant loop: b h w
#     irrelevant = [le.W, le.H, le.B]
#     b = loop_lower_bound.b
#     h = loop_lower_bound.h
#     w = loop_lower_bound.w
#     # r = layer.hfil
#     # d = layer.wfil
#     r = 1
#     d = 1

#     f = r * d / (layer.hstd * layer.wstd)
#     a = b * h * w * 3 / (4 * f)

#     s = capacity
    
#     access = float('inf')
                        
#     bb = 1
#     hh = 1
#     ww = 1
#     for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
#         for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
#             if (cc//loop_lower_bound.c)*(kk//loop_lower_bound.k)*r*d <= s :
#                 rho_temp = (cc * kk * r * d) / (cc * layer.hstd * layer.wstd + (2 * kk - 1))
#                 access_temp = layer.total_filter_size + math.ceil(layer.total_ops / rho_temp)
#                 if access_temp < access :
#                     access = access_temp
#                     rho = rho_temp
#                     k = kk
#                     c = cc
#                     b = bb
#                     h = hh
#                     w = ww
                    
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     loop_order = loop_order_generator(layer, loop_block, irrelevant)
#     q = access

#     return q, loop_block, loop_order

# def _ifmapr_v1(layer, capacity, loop_lower_bound):

#     # irrelevant loop: k
#     irrelevant = [le.K]
#     # r = layer.hfil
#     # d = layer.wfil
#     r = 1
#     d = 1
#     k = loop_lower_bound.k
#     f = r * d / (layer.hstd * layer.wstd)
#     a = k * 3 * f / 4

#     s = capacity

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

#     access = float('inf')
    
#     bb = 1
#     kk = loop_lower_bound.k
#     for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
#         if (cc//loop_lower_bound.c)*(kk//loop_lower_bound.k)*r*d <= s :
#             for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
#                 for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
#                     if (bb*hh*ww <= bhw_upper_bound) :
#                         bhw_temp = bb*hh*ww
#                         rho_temp = (bhw_temp * cc * r * d) / (cc * r * d + (2 * bhw_temp - 1))
#                         access_temp = math.ceil(layer.total_ops / rho_temp) + layer.total_ifmap_size
#                         if access_temp <= access :
#                             access = access_temp
#                             rho = rho_temp
#                             k = kk
#                             c = cc
#                             b = bb
#                             h = hh
#                             w = ww
#                             bhw = bhw_temp

#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     loop_order = loop_order_generator(layer, loop_block, irrelevant)
#     q = access

#     return q, loop_block, loop_order

# _unilayer_schedule_list_v1 = [_cwr_c_v1, _cwr_k_v1, _filterr_v1, _ifmapr_v1, _psumsr_v1]
# _unilayer_schedule_list_v2 = [_psumsr_v2, _filterr_v2, _ifmapr_v2, _cwr_c_v2, _cwr_k_v2]

# _unilayer_schedule_list_v1 = [_ifmapr_v1, _psumsr_v1, _filterr_v1]
_unilayer_schedule_list_v2 = [_ifmapr_v2, _psumsr_v2, _filterr_v2]

_unilayer_schedule_list_v2_ws = [_ifmapr_v2, _psumsr_v2, _filterr_v2]
_unilayer_schedule_list_v2_os = [_ifmapr_v2_os, _psumsr_v2_os, _filterr_v2_os]

_unilayer_schedule_list_v1 = [_filterr_v1_ws, _psumsr_v1_os]
_unilayer_schedule_list_v1_ws = [_filterr_v1_ws]
_unilayer_schedule_list_v1_os = [_psumsr_v1_os]
# _unilayer_schedule_list_v2 = [_filterr_v2, _psumsr_v2]


# def _psumsr_v2(layer, resource, loop_lower_bound):
#     irrelevant = [le.C]

#     p2 = resource.access_cost[2]
#     p1 = resource.access_cost[1]
#     r = layer.hfil
#     d = layer.wfil
#     q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
#     q1 = 2 * p1[1] / (r * d)
#     q2 = (p2[2] + p1[2])
#     q = min([q0, q1, q2])/2
#     s = resource.buffer(1).capacity

#     c,w,h,k,b = 0,0,0,0,0
#     hh, ww = 1 , 1
#     bb = layer.nimg
#     access = float('inf')
#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

#     # constrain & optimize
#     for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
#         for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
#             max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
#             if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
#                 for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
#                     for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
#                         if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
#                             bhw_temp = bb*hh*ww
#                             access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size
#                             if access_temp < access :
#                                 access = access_temp
#                                 k = kk
#                                 c = cc
#                                 b = bb
#                                 h = hh
#                                 w = ww
#                                 bhw = bhw_temp
                                    
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if any(i == 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
#     else :
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + (p2[1]-p1[1]) * layer.total_ofmap_size

#     if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
#     fi_glb_access_cost = p1[2] * layer.total_ops / bhw
#     if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_dram_access_cost = p2[1] * layer.total_ofmap_size
#     fi_dram_access_cost = p2[2] * layer.total_ops / bhw
#     glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
#     dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

#     return q, loop_block, loop_order, glb_access_cost, dram_access_cost

# def _filterr_v2(layer, resource, loop_lower_bound):
#     irrelevant = [le.W, le.H, le.B]

#     p2 = resource.access_cost[2]
#     p1 = resource.access_cost[1]
#     r = layer.hfil
#     d = layer.wfil
#     q0 = (p2[0] + p1[0]) * layer.hstd * layer.wstd / (r * d)
#     q1 = 2 * (p2[1] + p1[1]) / (r * d)
#     q2 = p1[2]
#     s = resource.buffer(1).capacity
#     q = min([q0, q1, q2])/2
#     c,w,h,k,b = 0,0,0,0,0
#     hh, ww = 1 , 1
#     bb = layer.nimg
#     access = float('inf')

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

#     # constrain & optimize
#     for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
#         for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
#             max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
#             if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
#                 for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
#                     for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
#                         for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
#                             if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
#                                 bhw_temp = bb*hh*ww
#                                 access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + p2[2] * layer.total_filter_size - (p2[1] + p1[1]) * layer.total_ofmap_size
#                                 if access_temp < access :
#                                     access = access_temp
#                                     k = kk
#                                     c = cc
#                                     b = bb
#                                     h = hh
#                                     w = ww
#                                     bhw = bhw_temp
                                    
#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if any(i == 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
#     else :
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[2] * layer.total_filter_size \
#         - (p2[1] + p1[1]) * layer.total_ofmap_size

#     if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
#     fi_glb_access_cost = p1[2] * layer.total_ops / bhw
#     if_dram_access_cost = p2[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
#     fi_dram_access_cost = p2[2] * layer.total_filter_size
#     glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
#     dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

#     return q, loop_block, loop_order, glb_access_cost, dram_access_cost

# def _ifmapr_v2(layer, resource, loop_lower_bound):
#     # irrelevant loop: k
#     irrelevant = [le.K]

#     p2 = resource.access_cost[2]
#     p1 = resource.access_cost[1]
#     r = layer.hfil
#     d = layer.wfil
#     q0 = p1[0] * layer.hstd * layer.wstd / (r * d)
#     q1 = 2 * (p2[1] + p1[1]) / (r * d)
#     q2 = (p2[2] + p1[2])
#     s = resource.buffer(1).capacity
#     q = min([q0, q1, q2])/2
#     c,w,h,k,b = 0,0,0,0,0
#     bb ,hh, ww = 1 , 1 , 1
#     access = float('inf')

#     bhw_lower_bound = loop_lower_bound.b * loop_lower_bound.h * loop_lower_bound.w
#     bhw_upper_bound = layer.nimg * layer.hofm * layer.wofm

#     # constrain & optimize
#     for cc in get_divisor(layer.nifm//loop_lower_bound.c,loop_lower_bound.c) :
#         for kk in get_divisor(layer.nofm//loop_lower_bound.k,loop_lower_bound.k) :
#             max_bhw_temp = min(int((s-layer.hfil*layer.wfil*cc*kk)/(kk+layer.hstd*layer.wstd*cc)) , bhw_upper_bound)
#             if (max_bhw_temp > bhw_lower_bound) and (max_bhw_temp <= bhw_upper_bound) :
#                 for bb in get_divisor(layer.nimg//loop_lower_bound.b,loop_lower_bound.b) :
#                     for hh in get_divisor(layer.hofm//loop_lower_bound.h,loop_lower_bound.h) :
#                         for ww in get_divisor(layer.wofm//loop_lower_bound.w,loop_lower_bound.w) :
#                             if (bb*hh*ww <= max_bhw_temp) and (bb*hh*ww*kk + r*d*cc*kk + bb*(layer.hstd*(hh-1)+r)*(layer.wstd*(ww-1)+d)*cc <= s) :
#                                 bhw_temp = bb*hh*ww
#                                 access_temp = (q0/kk + q1/cc + q2/bhw_temp)* layer.total_ops + p2[0] * layer.total_ifmap_size - (p2[1] + p1[1]) * layer.total_ofmap_size
#                                 if access_temp < access :
#                                     access = access_temp
#                                     k = kk
#                                     c = cc
#                                     b = bb
#                                     h = hh
#                                     w = ww
#                                     bhw = bhw_temp

#     loop_block = [d, r, c, w, h, k, b]
#     loop_order = [le.NUM - 1] * le.NUM
#     if any(i == 0 for i in loop_block):
#         return float('inf'), loop_block, loop_order, [float('inf'), float('inf'), float('inf')], [float('inf'), float('inf'), float('inf')]
#     else :
#         loop_order = loop_order_generator(layer, loop_block, irrelevant)

#     q = (q0 / k + q1 / c + q2 / bhw) * layer.total_ops + p2[0] * layer.total_ifmap_size \
#         - (p2[1] + p1[1]) * layer.total_ofmap_size

#     if_glb_access_cost = p1[0] * layer.hstd * layer.wstd * layer.total_ops / (r * d * k)
#     of_glb_access_cost = 2 * p1[1] * layer.total_ops / (r * d * c) - p1[1] * layer.total_ofmap_size
#     fi_glb_access_cost = p1[2] * layer.total_ops / bhw
#     if_dram_access_cost = p2[0] * layer.total_ifmap_size
#     of_dram_access_cost = 2 * p2[1] * layer.total_ops / (r * d * c) - p2[1] * layer.total_ofmap_size
#     fi_dram_access_cost = p2[2] * layer.total_ops / bhw
#     glb_access_cost = [if_glb_access_cost, of_glb_access_cost, fi_glb_access_cost]
#     dram_access_cost = [if_dram_access_cost, of_dram_access_cost, fi_dram_access_cost]

#     return q, loop_block, loop_order, glb_access_cost, dram_access_cost