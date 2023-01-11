import math
import numpy as np
from scipy.optimize import minimize
import copy

from collections import namedtuple

from .. import util
from .layer import ConvLayer, LocalRegionLayer
from .network import Network
from .resource import Resource
from . import schedule_generator
from . import loop_enum as le


class InterLayerReuse(object):
    """
    Inter-layer reuse.
    """

    SchedIndex = namedtuple('SchedIndex', ['sp_idx', 'tm_idx'])
    Scale = namedtuple('Scale', ['s_h', 's_w'])
    MinSize = namedtuple('MinSize', ['h', 'w'])

    def __init__(self, network, fusion_group, resource, dataflow, loop_lower_bound, topological_order=True,
                 z_fusion=False, d_fusion=False, womincost=True, timeloop_cost_model=False):
        if not isinstance(network, Network):
            raise TypeError('InterLayerReuse: network must be '
                            'a Network instance.')

        if not isinstance(fusion_group, list):
            raise TypeError('InterLayerReuse: fusion_group must be '
                            'a list.')

        if not isinstance(resource, Resource):
            raise TypeError('InterLayerPipeline: resource must be '
                            'a Resource instance.')

        self.network = network
        self.fusion_group = fusion_group
        self.resource = resource
        self.loop_lower_bound = loop_lower_bound
        self.topological_order = topological_order
        self.z_fusion = z_fusion
        self.d_fusion = d_fusion
        self.womincost = womincost
        self.timeloop_cost_model = timeloop_cost_model
        self.dataflow = dataflow
        self.height_tile_num = 0
        self.batch_tile_num = 0
        self.total_tile_num = 0

        if self.timeloop_cost_model :
            self.valid2 = self._prepare()
            if not self.valid2:
                return
        else :
            self.valid = self._prepare()
            if not self.valid:
                return

        # self._calc_sched_dag()
        
        # self.valid = self._init_alternate_pair()
        # if not self.valid:
        #     return

    def sched(self, mode):
        if self.valid:
            self._calc_sched_dag()
            self.valid = self._init_alternate_pair(mode)

    def sched_timeloop_cost_model(self, overlap_reuse=True, weight_reuse=True):
        # if self.valid:
        self._calc_sched_dag()
        self.valid = self._init_alternate_pair_weight_iverlap_reuse_considered(overlap_reuse,weight_reuse)

    def _prepare(self):
        self.firsts = []
        self.lasts = []
        self.ext_inputs_dict = dict()
        self.ext_outputs = set()
        self.fused_weight_size = 0
        self.fused_input_size = 0
        self.fused_output_size = 0
        self.Required_Buffer_Weight_Full_Reuse_OS = 0
        
        if self.d_fusion:
            for layer in self.fusion_group:
                if len(self.network.nexts(layer)) > 1 or len(self.network.prevs(layer)) > 1:
                    self.valid = False
                    return False

        for layer in self.fusion_group:
            tmp = tuple()
            for nx in self.network.nexts(layer):
                if nx not in self.fusion_group:
                    # print(layer, nx)
                    tmp += (nx, )
                    self.ext_outputs.add(layer)
                    # self.lasts.append(layer)
            if tmp == self.network.nexts(layer):
                self.lasts.append(layer)
            
            tmp = tuple()
            for pre in self.network.prevs(layer):
                if pre not in self.fusion_group:
                    # print(layer, pre)
                    tmp += (pre, )
                    if pre not in self.ext_inputs_dict:
                        self.ext_inputs_dict[pre] = [layer]
                    else:
                        self.ext_inputs_dict[pre].append(layer)
            if tmp == self.network.prevs(layer):
                if isinstance(self.network[layer], LocalRegionLayer):
                    return False
                self.firsts.append(layer)
                
            if isinstance(self.network[layer], ConvLayer):
                self.fused_weight_size += self.network[layer].total_filter_size
                self.Required_Buffer_Weight_Full_Reuse_WS = self.resource.paras[0].count
                self.Required_Buffer_Weight_Full_Reuse_OS = max( (self.network[layer].total_filter_size//self.network[layer].nofm)*32 , self.Required_Buffer_Weight_Full_Reuse_OS )
                # print(layer,self.network[layer],self.Required_Buffer_Weight_Full_Reuse_WS,self.Required_Buffer_Weight_Full_Reuse_OS)

        for ip in self.ext_inputs_dict:
            if ip is None:
                self.fused_input_size += self.network[self.network.INPUT_LAYER_KEY].total_ofmap_size
            else:
                self.fused_input_size += self.network[ip].total_ofmap_size
        for op in self.ext_outputs:
            self.fused_output_size += self.network[op].total_ofmap_size
        return True

    def _calc_sched_dag(self):

        # The DAG vertex list in the topological order.
        if self.topological_order:
            self.dag_vertex_list = self._topological_order()
            # print(self.dag_vertex_list)
        else:
            self.dag_vertex_list = self.fusion_group

        # Make a directory from layer name to DAG vertex index.
        self.dag_vertex_dict = {}

        for vidx, layer_name in enumerate(self.dag_vertex_list):
            assert layer_name not in self.dag_vertex_dict
            self.dag_vertex_dict[layer_name] = vidx

        # The previous and next relationship of the DAG vertices.
        self.dag_prev_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))
        self.dag_next_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))

        for layer_name in self.fusion_group:
            vidx = self.dag_vertex_dict[layer_name]

            # Previous layers.
            for p in self.network.prevs(layer_name):
                if not p or p not in self.fusion_group:
                    continue
                pvidx = self.dag_vertex_dict[p]
                if pvidx != vidx:
                    self.dag_prev_dict[vidx].add(pvidx)

            # Next layers.
            for n in self.network.nexts(layer_name):
                if not n or n not in self.fusion_group:
                    continue
                nvidx = self.dag_vertex_dict[n]
                if nvidx != vidx:
                    self.dag_next_dict[vidx].add(nvidx)

        self.ext_inputs_idx = {}
        for vidx, layer_name in enumerate(self.ext_inputs_dict.keys()):
            assert layer_name not in self.dag_vertex_dict
            self.ext_inputs_idx[layer_name] = vidx + len(self.dag_vertex_list)

    def _topological_order(self):

        # The visited layers in the DFS order.
        visited = []
        # The unseen pending layers.
        unseen = set(self.fusion_group)
        # The layers that have been seen, but not visited due to unvisited
        # previous layers.
        seen = set()

        def _dfs(vertex):
            assert vertex not in seen
            if vertex in visited:
                return

            unseen.discard(vertex)
            seen.add(vertex)

            next_vertices = []

            for n in reversed(self.network.nexts(vertex)):
                if n and n not in next_vertices and n in unseen:
                    next_vertices.append(n)

            for nv in next_vertices:
                _dfs(nv)

            visited.append(vertex)
            seen.remove(vertex)

        # Start from the first layers.
        for v in self.firsts:
            _dfs(v)
        assert not unseen
        assert not seen

        return list(reversed(visited))

    def ordered_layer_list(self):

        return list(sum(self.dag_vertex_list, tuple()))

    def _init_scale(self):
        scale_tmp = [None for _ in self.dag_vertex_list]

        for idx, l in enumerate(self.dag_vertex_list):
            layer = self.network[l]
            if l in self.firsts:
                scale_tmp[idx] = [layer.hstd, layer.wstd]
                continue

            max_hs, max_ws = 0, 0
            for src in self.dag_prev_dict[idx]:
                src_scale = scale_tmp[src]
                assert src_scale
                max_hs = src_scale[0] if src_scale[0] > max_hs else max_hs
                max_ws = src_scale[1] if src_scale[1] > max_ws else max_ws
            scale_tmp[idx] \
                = [max_hs * layer.hstd, max_ws * layer.wstd]

        self.scale = [None for _ in self.dag_vertex_list]

        last_h = []
        last_w = []
        for l in self.lasts:
            idx = self.dag_vertex_dict[l]
            last_h.append(scale_tmp[idx][0])
            last_w.append(scale_tmp[idx][1])
        s_h = util.lcm(*last_h)
        s_w = util.lcm(*last_w)
        # print(s_h,s_w)
        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            if l in self.lasts:
                # print(l)
                self.scale[idx] = \
                    InterLayerReuse.Scale(s_h / scale_tmp[idx][0], s_w / scale_tmp[idx][1])
                continue

            s_h_tmp, s_w_tmp = None, None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_scale = self.scale[dst_idx]
                assert dst_scale
                if s_h_tmp is None and s_w_tmp is None:
                    s_h_tmp, s_w_tmp = dst_layer.hstd * dst_scale.s_h, dst_layer.wstd * dst_scale.s_w
                else:
                    assert s_h_tmp == dst_layer.hstd * dst_scale.s_h \
                           and s_w_tmp == dst_layer.wstd * dst_scale.s_w
            self.scale[idx] = \
                InterLayerReuse.Scale(s_h_tmp, s_w_tmp)

        self.minSize = [None for _ in self.dag_vertex_list]
        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            if l in self.lasts:
                self.minSize[idx] = InterLayerReuse.MinSize(self.scale[idx].s_h, self.scale[idx].s_w)
                continue

            h_tmp, w_tmp = None, None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_minsize = self.minSize[dst_idx]
                assert dst_minsize
                if isinstance(dst_layer, LocalRegionLayer):
                    hreg, wreg = dst_layer.hreg, dst_layer.wreg
                else:
                    hreg, wreg = dst_layer.hfil, dst_layer.wfil
                if h_tmp is None and w_tmp is None:
                    h_tmp = (dst_minsize.h - 1) * dst_layer.hstd + hreg
                    w_tmp = (dst_minsize.w - 1) * dst_layer.wstd + wreg
                    h_tmp = layer.hofm if h_tmp > layer.hofm else h_tmp
                    w_tmp = layer.wofm if w_tmp > layer.wofm else w_tmp
                else:
                    if (dst_minsize.h - 1) * dst_layer.hstd + hreg > h_tmp:
                        h_tmp = (dst_minsize.h - 1) * dst_layer.hstd + hreg
                        h_tmp = layer.hofm if h_tmp > layer.hofm else h_tmp
                    if (dst_minsize.w - 1) * dst_layer.wstd + wreg > w_tmp:
                        w_tmp = (dst_minsize.w - 1) * dst_layer.wstd + wreg
                        w_tmp = layer.wofm if w_tmp > layer.wofm else w_tmp
            self.minSize[idx] = InterLayerReuse.MinSize(h_tmp, w_tmp)
        # print(self.scale , self.minSize)

    def _init_alternate_pair_weight_iverlap_reuse_considered(self, overlap_reuse, weight_reuse):        
        
        irrelevant = [le.D, le.R, le.K, le.C]
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self._init_scale()
        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint , self.line_buffer_footprint = self._alternate_line_buffer_included()

        if self.is_full_buffer is None:
            # print(1)
            return False

        level = self.resource.buffer_levels() - 2
        total_buffer_size = self.resource.buffer(level).capacity * self. resource.paras[level].count
        size_temp = 0
        if overlap_reuse :
            reuse_buffer = self.line_buffer_footprint 
            min_feature_buffer = self.min_feature_footprint + self.line_buffer_footprint
        else :
            reuse_buffer = 0
            min_feature_buffer = self.min_feature_footprint
            
        if total_buffer_size <= min_feature_buffer:
            # print(2)
            return False

        if weight_reuse :
            if total_buffer_size >= min_feature_buffer + self.fused_weight_size :
                self.sfil_fit = True
                weight_buffer = self.fused_weight_size
            else :
                # print(3)
                return False
        else :
            self.sfil_fit = False
            if self.dataflow == 'OS' :
                weight_buffer = self.Required_Buffer_Weight_Full_Reuse_OS
            if self.dataflow == 'WS' :
                weight_buffer = 1024
        
        feature_buffer_temp = total_buffer_size - weight_buffer - reuse_buffer
        
        output_h_min = self.network.input_layer().hofm
        output_w_min = self.network.input_layer().wofm
        for l in self.lasts:
            if self.network[l].hofm < output_h_min:
                output_h_min = self.network[l].hofm
            if self.network[l].wofm < output_w_min:
                output_w_min = self.network[l].wofm

        line_num = math.ceil((feature_buffer_temp - self.min_feature_footprint) / self.add_one_line_footprint)
        # print(line_num)
        if line_num < 1 :
            # print(4)
            return False
        
        if line_num > output_h_min:
            h = output_h_min
            b_temp = feature_buffer_temp//((output_h_min-1)*self.add_one_line_footprint+self.min_feature_footprint)
            
            if b_temp <= 1 :
                b = 1
            elif b_temp > 1 :
                b = min(b_temp,self.network.input_layer().nimg)
        
        else:
            h = line_num
            b = 1
            
        output_tile_h_min = self.network.input_layer().hofm
        output_tile_h_recompute = [None for _ in self.dag_vertex_list]
        output_tile_h_reuse = [None for _ in self.dag_vertex_list]
        output_tile_h_epilogue_reuse = [None for _ in self.dag_vertex_list]
        output_tile_h_epilogue_recompute = [None for _ in self.dag_vertex_list]
        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            if l in self.lasts:
                output_tile_h_reuse[idx] = int(self.scale[idx].s_h * h)
                output_tile_h_recompute[idx] = int(self.scale[idx].s_h * h)
                if output_tile_h_reuse[idx] < output_tile_h_min :
                    output_tile_h_min = output_tile_h_reuse[idx]
                continue
            h_tmp_recompute , h_tmp_reuse = None , None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_h_recompute = output_tile_h_recompute[dst_idx]
                dst_h_reuse = output_tile_h_reuse[dst_idx]
                if isinstance(dst_layer, LocalRegionLayer):
                    hreg = dst_layer.hreg
                else:
                    hreg = dst_layer.hfil
                if h_tmp_recompute is None:
                    h_tmp_recompute = min((dst_h_recompute - 1) * dst_layer.hstd + hreg, layer.hofm)
                    h_tmp_reuse = min(dst_h_reuse * dst_layer.hstd , layer.hofm)
                else:
                    if (dst_h_recompute - 1) * dst_layer.hstd + hreg > h_tmp_recompute :
                        h_tmp_recompute = min((dst_h_recompute - 1) * dst_layer.hstd + hreg, layer.hofm)
                    if dst_h_reuse * dst_layer.hstd  > h_tmp_reuse :
                        h_tmp_reuse = min(dst_h_reuse * dst_layer.hstd , layer.hofm)
            output_tile_h_recompute[idx] = math.floor(h_tmp_recompute)
            output_tile_h_reuse[idx] = math.floor(h_tmp_reuse)
        
        self.height_tile_num = math.ceil( (output_h_min/output_tile_h_min) )
        self.batch_tile_num = math.ceil( self.network.input_layer().nimg/b)
        self.total_tile_num = self.height_tile_num * self.batch_tile_num 

        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            output_tile_h_epilogue_reuse[idx] = layer.wofm - output_tile_h_reuse[idx]*\
                (self.height_tile_num - 1)
            output_tile_h_epilogue_recompute[idx] = output_tile_h_epilogue_reuse[idx]+\
                (output_tile_h_recompute[idx] - output_tile_h_reuse[idx])

        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            if isinstance(layer, LocalRegionLayer):
                if l in self.firsts:
                    size_temp += layer.wofm * output_tile_h_recompute[idx] * layer.nofm * b
                continue
            c = layer.nifm
            k = layer.nofm
            size_temp += layer.wofm * output_tile_h_recompute[idx] * layer.nofm * b
            
            self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, output_tile_h_reuse[idx], k, b]
            self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)
            
        self.H_recompute = output_tile_h_recompute
        self.H_reuse = output_tile_h_reuse
        self.H_epilogue_recompute = output_tile_h_epilogue_recompute
        self.H_epilogue_reuse = output_tile_h_epilogue_reuse
        
        # print('  Buffer(total , weight, reuse , available :' , int(total_buffer_size//1024) , weight_buffer//1024 , reuse_buffer//1024 , int(feature_buffer_temp//1024)) 
        # print('  Buffer(min_feature , one_line_size, residue) :' ,  self.min_feature_footprint//1024, self.add_one_line_footprint//1024 , h , int((feature_buffer_temp-self.min_feature_footprint-(h-1)*(self.add_one_line_footprint))//1024) )
        # print('  Batch , Output_Min_H, Batch_Tile(Size) , Output_H_Tile(Size) :', self.network.input_layer().nimg, output_h_min, b, h)
        # print('  Num of Tiles(Total,B,H) :', self.total_tile_num, self.batch_tile_num, self.height_tile_num )
        
        DRAM_Access_Cost = self.resource.access_cost[2]
        Buffer_Access_Cost = self.resource.access_cost[1]
        Input_DRAM_Cost , Output_DRAM_Cost , Weight_DRAM_Cost = DRAM_Access_Cost[0] , DRAM_Access_Cost[1] , DRAM_Access_Cost[2]
        Input_Buffer_Cost , Output_Buffer_Cost , Weight_Buffer_Cost = Buffer_Access_Cost[0] , Buffer_Access_Cost[1] , Buffer_Access_Cost[2]
        
        input_cost = self.fused_input_size * Input_DRAM_Cost
        output_cost = self.fused_output_size * Output_DRAM_Cost
        
        if self.sfil_fit == True :
            weight_cost = self.fused_weight_size * (Weight_DRAM_Cost + Weight_Buffer_Cost)
        elif self.sfil_fit == False :
            weight_cost = self.batch_tile_num * self.height_tile_num * self.fused_weight_size * (Weight_DRAM_Cost + Weight_Buffer_Cost)
        q = input_cost + output_cost + weight_cost

        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            if isinstance(layer, LocalRegionLayer):
                continue
            q += (Input_Buffer_Cost *  layer.total_ifmap_size * (layer.nofm / self.loop_block[idx][5]) )
            q += (Output_Buffer_Cost * layer.total_ofmap_size * (layer.nifm / self.loop_block[idx][2]) )
        
        self.q = q
        return True
        
    def _init_alternate_pair_optimus(self):
        size_temp = 0
        self._init_scale()
        irrelevant = [le.D, le.R, le.K, le.C]
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint = self._alternate_2()
        # print(self.min_feature_footprint, self.add_one_line_footprint)

        if self.is_full_buffer is None:
            return False

        level = self.resource.buffer_levels() - 2
        total_buffer_size = self.resource.buffer(level).capacity * self. resource.paras[level].count

        if total_buffer_size <= self.min_feature_footprint:
            return False

        if total_buffer_size >= self.fused_weight_size + self.min_feature_footprint:
            self.sfil_fit = True
            self.height_tile_num = 1

            output_h_min = self.network.input_layer().hofm
            output_w_min = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < output_h_min:
                    output_h_min = self.network[l].hofm
                if self.network[l].wofm < output_w_min:
                    output_w_min = self.network[l].wofm

            total_buffer_size = total_buffer_size - self.fused_weight_size
            line_num = math.floor((total_buffer_size - self.min_feature_footprint) /
                                  self.add_one_line_footprint) + 1
            if line_num > output_h_min:
                h = output_h_min
                b = int(max(total_buffer_size // ((output_h_min - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                # print(1, h_tmp)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                k = layer.nofm
                c = layer.nifm
                # kk = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                # k = layer.nofm if self.is_full_buffer[idx] else kk
                # c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                # for pre in self.dag_prev_dict[idx]:
                #     if self.is_full_buffer[pre]:
                #         c = layer.nifm

                # if k < layer.nofm and c < layer.nifm:
                #     c = layer.nifm
                size_temp += layer.wofm * H[idx] * k * b
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

        else:
            if self.z_fusion or self.d_fusion:
                return False
            self.sfil_fit = False
            # output_h_min : Height of Minimum Output Layer
            output_h_min = self.network.input_layer().hofm
            output_w_min = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < output_h_min:
                    output_h_min = self.network[l].hofm
                if self.network[l].wofm < output_w_min:
                    output_w_min = self.network[l].wofm

            # Line_Num : Height of Max Output Tile Size
            line_num = math.floor((total_buffer_size - self.min_feature_footprint) / self.add_one_line_footprint) + 1
            
            if line_num > output_h_min:
                # print(1)
                h = output_h_min
                # b = int(min(total_buffer_size // ((output_h_min - 1) * self.add_one_line_footprint + self.min_feature_footprint), self.network.input_layer().nimg))
                b = 1
                # h = output_h_min
                # b = 1
            else:
                # print(2)
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    print(2, H[idx])
                    continue                

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                
                H[idx] = math.floor(h_tmp)
                print(2, H[idx])
                # h_tmp = None
                # for dst_idx in self.dag_next_dict[idx]:
                #     dst = self.dag_vertex_list[dst_idx]
                #     dst_layer = self.network[dst]
                #     dst_h = H[dst_idx]
                #     assert dst_h is not None
                #     if isinstance(dst_layer, LocalRegionLayer):
                #         hreg = dst_layer.hreg
                #     else:
                #         hreg = dst_layer.hfil
                #     if h_tmp is None:
                #         h_tmp = min(dst_h, layer.hofm)
                #     else:
                #         if dst_h > h_tmp:
                #             h_tmp = min(dst_h, layer.hofm)
                # print(h_tmp)
                # H[idx] = math.floor(h_tmp)
                
            
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    if l in self.firsts:
                        c = layer.nifm
                        k = layer.nofm
                        size_temp += layer.wofm * H[idx] * k * b
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                k = layer.nofm
                c = layer.nifm
                
                size_temp += layer.wofm * H[idx] * k * b
                
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)
                # irrelevant = [le.D, le.R, le.K, le.C] -> DRKC WHB -> DRKCHB W
                # self.height_tile_num = math.ceil(output_h_min * self.network.input_layer().nimg / (b * h))
            self.height_tile_num = math.ceil(output_h_min * self.network.input_layer().nimg / (b * h))
            # print('Test:',self.resource.buffer(level).capacity * self. resource.paras[level].count , size_temp, self.resource.buffer(level).capacity * self. resource.paras[level].count - size_temp)
        self.q = self.fused_weight_size * self.height_tile_num + self.fused_input_size + self.fused_output_size
        
        # print(self.q,self.sfil_fit,line_num,self.height_tile_num, H)
        
        p2 = self.resource.access_cost[2]
        p1 = self.resource.access_cost[1]

        q0, q1, q2 = p1[0], p1[1], p2[2] + p1[2]
        # q0, q1, q2 = p1[0], p1[1], p1[2]
        # print(self.dag_vertex_list)
        # print(self.height_tile_num)
        f_args = (q0, q1, q2, b, output_h_min)
        fun = self.fun(f_args)
        c_args = (b, output_h_min, self.idx_dict)
        con = self.con(c_args)
        x0 = [1 for _ in range(len(set(self.idx_dict.values())) + 1)]
        if b > 1:
            x0[0] = b
        else:
            x0[0] = h
        for idx in self.idx_dict:
            if idx < len(self.dag_vertex_list):
                layer = self.network[self.dag_vertex_list[idx]]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                x0[self.idx_dict[idx]] = loop_lower_bound.k

        x0 = np.asarray(x0)
        res = minimize(fun, x0, method='COBYLA', constraints=con)
        
        if res.success:
            size_temp = 0
            # import pdb
            # pdb.set_trace()
            # print(x0, res.x)
            #x0[0] = h , res.x = optimized_h
            if b > 1:
                b = math.ceil(res.x[0])
                h = output_h_min
            else:
                b = 1
                # h = math.ceil(res.x[0])
                h = math.floor(res.x[0])
                
            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    print(3, l, H[idx])
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                
                H[idx] = math.floor(h_tmp)
                print(3, l, H[idx])
                
                # h_tmp = None
                # for dst_idx in self.dag_next_dict[idx]:
                #     dst = self.dag_vertex_list[dst_idx]
                #     dst_layer = self.network[dst]
                #     dst_h = H[dst_idx]
                #     assert dst_h is not None
                #     if isinstance(dst_layer, LocalRegionLayer):
                #         hreg = dst_layer.hreg
                #     else:
                #         hreg = dst_layer.hfil
                #     if h_tmp is None:
                #         h_tmp = min(dst_h, layer.hofm)
                #     else:
                #         if dst_h > h_tmp:
                #             h_tmp = min(dst_h, layer.hofm)
                # print(h_tmp)
                # H[idx] = math.floor(h_tmp)
            
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    if l in self.firsts:
                        c = layer.nifm
                        k = layer.nofm
                        size_temp += layer.wofm * H[idx] * k * b
                    continue
                c = layer.nifm
                k = layer.nofm
                
                # if idx in self.idx_dict:
                #     k = res.x[self.idx_dict[idx]]
                # else:
                #     k = layer.nofm

                # if self.dag_prev_dict[idx] and list(self.dag_prev_dict[idx])[0] in self.idx_dict:
                #     c = self.idx_dict[list(self.dag_prev_dict[idx])[0]]
                # else:
                #     c = layer.nifm
                    
                size_temp += layer.wofm * H[idx] * math.ceil(k) * b
                self.loop_block[idx] = \
                    [layer.wfil, layer.hfil, math.ceil(c), layer.wofm, H[idx], math.ceil(k), b]

                self.loop_order[idx] = \
                    schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

        # q = self.fused_weight_size * math.ceil(self.network.input_layer().nimg * output_h_min * q2 / (b * h)) \
            # + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]  
        
        # if self.sfil_fit == True :
        #     weight_cost = self.fused_weight_size * q2
        # elif self.sfil_fit == False :
        #     weight_cost = self.fused_weight_size * math.ceil(self.network.input_layer().nimg * output_h_min * q2 / (b * h))
        # q = weight_cost + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]
        
        input_cost = self.fused_input_size * p2[0]
        output_cost = self.fused_output_size * p2[1]
        self.height_tile_num = math.ceil(output_h_min * self.network.input_layer().nimg / (b * h))
        if self.sfil_fit == True :
            weight_cost = self.fused_weight_size * (p1[2] + p2[2])
        elif self.sfil_fit == False :
            weight_cost = self.fused_weight_size * self.height_tile_num * (p1[2] + p2[2])

        q = weight_cost + input_cost + output_cost
        # print(weight_cost,input_cost,output_cost)
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            if isinstance(layer, LocalRegionLayer):
                continue
            # print(q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2] , q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            q += (q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2])
            q += (q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            # q += 0
        
        self.q = q
        # print(self.q,self.sfil_fit,line_num,self.height_tile_num, H)
        # print(total_buffer_size , size_temp)
        # print(self.q , self.sfil_fit , self.height_tile_num , line_num, math.floor(res.x[0]), H)
        # print('Test:',self.resource.buffer(level).capacity * self. resource.paras[level].count , size_temp, self.resource.buffer(level).capacity * self. resource.paras[level].count - size_temp)
        return True

    def _init_alternate_pair_others(self, mode):
        self._init_scale()
        irrelevant = [le.D, le.R, le.K, le.C]
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint = self._alternate()

        if self.is_full_buffer is None:
            return False

        level = self.resource.buffer_levels() - 2
        total_buffer_size = self.resource.buffer(level).capacity * self. resource.paras[level].count

        if total_buffer_size <= self.min_feature_footprint:
            return False

        if total_buffer_size >= self.fused_weight_size + self.min_feature_footprint:
            self.sfil_fit = True
            self.height_tile_num = 1

            output_h_min = self.network.input_layer().hofm
            output_w_min = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < output_h_min:
                    output_h_min = self.network[l].hofm
                if self.network[l].wofm < output_w_min:
                    output_w_min = self.network[l].wofm

            total_buffer_size = total_buffer_size - self.fused_weight_size
            line_num = math.floor((total_buffer_size - self.min_feature_footprint) /
                                  self.add_one_line_footprint) + 1
            if line_num > output_h_min:
                h = output_h_min
                # b = int(max(total_buffer_size // ((output_h_min - 1) * self.add_one_line_footprint + self.min_feature_footprint),self.network.input_layer().nimg))
                b = 1
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                kk = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                k = layer.nofm if self.is_full_buffer[idx] else kk
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm

                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

        else:
            if self.z_fusion or self.d_fusion:
                return False
            self.sfil_fit = False
            output_h_min = self.network.input_layer().hofm
            output_w_min = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < output_h_min:
                    output_h_min = self.network[l].hofm
                if self.network[l].wofm < output_w_min:
                    output_w_min = self.network[l].wofm

            line_num = math.floor((total_buffer_size - self.min_feature_footprint) / self.add_one_line_footprint) + 1
            if line_num > output_h_min:
                h = output_h_min
                b = int(min(total_buffer_size // ((output_h_min - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)

                k = layer.nofm if self.is_full_buffer[idx] \
                    else min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm
                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = \
                    schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

                self.height_tile_num = math.ceil(output_h_min * self.network.input_layer().nimg / (b * h))
        self.q = self.fused_weight_size * self.height_tile_num + self.fused_input_size + self.fused_output_size

        if mode == 1:
            p2 = self.resource.access_cost[2]
            p1 = self.resource.access_cost[1]

            q0, q1, q2 = p1[0], p1[1], p2[2] + p1[2]

            f_args = (q0, q1, q2, b, output_h_min)
            fun = self.fun(f_args)
            c_args = (b, output_h_min, self.idx_dict)
            con = self.con(c_args)
            x0 = [1 for _ in range(len(set(self.idx_dict.values())) + 1)]
            if b > 1:
                x0[0] = b
            else:
                x0[0] = h
            for idx in self.idx_dict:
                if idx < len(self.dag_vertex_list):
                    layer = self.network[self.dag_vertex_list[idx]]
                    if isinstance(layer, LocalRegionLayer):
                        continue
                    loop_lower_bound = self.loop_lower_bound(layer)
                    x0[self.idx_dict[idx]] = loop_lower_bound.k

            x0 = np.asarray(x0)
            res = minimize(fun, x0, method='COBYLA', constraints=con)

            if res.success:
                if b > 1:
                    b = math.ceil(res.x[0])
                    h = output_h_min
                else:
                    b = 1
                    h = math.ceil(res.x[0])
                H = [None for _ in self.dag_vertex_list]
                for l in reversed(self.dag_vertex_list):
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if l in self.lasts:
                        H[idx] = int(self.scale[idx].s_h * h)
                        continue

                    h_tmp = None
                    for dst_idx in self.dag_next_dict[idx]:
                        dst = self.dag_vertex_list[dst_idx]
                        dst_layer = self.network[dst]
                        dst_h = H[dst_idx]
                        assert dst_h is not None
                        if isinstance(dst_layer, LocalRegionLayer):
                            hreg = dst_layer.hreg
                        else:
                            hreg = dst_layer.hfil
                        if h_tmp is None:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                        else:
                            if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                                h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    H[idx] = math.floor(h_tmp)

                for l in self.dag_vertex_list:
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if isinstance(layer, LocalRegionLayer):
                        continue

                    if idx in self.idx_dict:
                        k = res.x[self.idx_dict[idx]]
                    else:
                        k = layer.nofm

                    if self.dag_prev_dict[idx] and list(self.dag_prev_dict[idx])[0] in self.idx_dict:
                        c = self.idx_dict[list(self.dag_prev_dict[idx])[0]]
                    else:
                        c = layer.nifm
                    self.loop_block[idx] = \
                        [layer.wfil, layer.hfil, math.ceil(c), layer.wofm, H[idx], math.ceil(k), b]

                    self.loop_order[idx] = \
                        schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

            q = self.fused_weight_size * self.network.input_layer().nimg * output_h_min * q2 / (b * h) \
                + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                q += (q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2])
                q += (q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            self.q = q
            self.height_tile_num = math.ceil(output_h_min * self.network.input_layer().nimg / (b * h))

        return True

    def _init_alternate_pair(self, mode):
        if self.d_fusion or self.z_fusion:
            return self._init_alternate_pair_others(mode)
        else:
            return self._init_alternate_pair_optimus()
        
    def fun(self, args):

        q0, q1, q2, b, output_h_min = args
        expr = ''

        fidx = 1
        idx_dict = dict()
        if b > 1:
            p2 = q2 * self.fused_weight_size * self.network.input_layer().nimg
            expr += '+ {p2} / x[0] '.format(p2=p2)
        else:
            p2 = q2 * self.fused_weight_size * self.network.input_layer().nimg * output_h_min
            expr += '+ {p2} / x[0] '.format(p2=p2)
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]

            if isinstance(layer, ConvLayer):
                assert len(self.dag_prev_dict[idx]) <= 1
                p0 = q0 * layer.total_ifmap_size * layer.nofm
                p1 = q1 * layer.total_ofmap_size * layer.nifm

                k = True if self.is_full_buffer[idx] else False
                c = False
                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = True
                if not k and not c:
                    c = True

                if not k:
                    if idx in idx_dict:
                        cur_fidx = idx_dict[idx]
                    else:
                        cur_fidx = fidx
                        idx_dict[idx] = fidx
                        fidx += 1
                    expr += '+ {p0} / x[{idx}] '.format(p0=p0, idx=cur_fidx)

                if not c:

                    if len(self.dag_prev_dict[idx]) == 1:
                        pidx = list(self.dag_prev_dict[idx])[0]
                        if pidx in idx_dict:
                            cur_fidx = idx_dict[pidx]
                        else:
                            cy_idx = pidx
                            while len(self.dag_prev_dict[cy_idx]) == 1:
                                if isinstance(self.network[self.dag_vertex_list[cy_idx]], ConvLayer):
                                    break
                                cy_idx = list(self.dag_prev_dict[cy_idx])[0]
                            if len(self.dag_prev_dict[cy_idx]) == 1:
                                if cy_idx in idx_dict:
                                    cur_fidx = idx_dict[cy_idx]
                                    idx_dict[pidx] = cur_fidx
                                else:
                                    cur_fidx = fidx
                                    idx_dict[cy_idx] = cur_fidx
                                    idx_dict[pidx] = cur_fidx
                                    fidx += 1
                            elif len(self.dag_prev_dict[cy_idx]) == 0:
                                continue

                            else:
                                cur_fidx = fidx
                                idx_dict[pidx] = cur_fidx
                                fidx += 1

                    else:
                        continue

                    expr += '+ {p1} / x[{idx}] '.format(p1=p1, idx=cur_fidx)

        self.idx_dict = idx_dict
        expr = expr[1:]
        v = lambda x: eval(expr)
        # print(expr)
        return v

    def con(self, args):
        b, output_h_min, idx_dict = args

        ineq_cons = []
        if b > 1:
            ineq_cons.append('x[0] - 1')
            ineq_cons.append('-x[0] + {nimg}'.format(nimg=self.network.input_layer().nimg))
        else:
            ineq_cons.append('x[0] - 1')
            ineq_cons.append('-x[0] + {hh}'.format(hh=output_h_min))

        ext_inputs = set(self.ext_inputs_dict.keys())
        ss = ''
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            sca = self.scale[idx]
            minsize = self.minSize[idx]
            loop_lower_bound = self.loop_lower_bound(layer)

            if l in self.firsts:
                if not self.is_full_buffer[idx]:
                    for src in self.network.prevs(l):
                        if src in ext_inputs:
                            if src is None:
                                src_layer = self.network.input_layer()
                            else:
                                src_layer = self.network[src]
                            m_h = min((minsize.h - 1) * layer.hstd + layer.hfil, src_layer.hofm)
                            s_h = sca.s_h * layer.hstd
                            if b > 1:
                                ss += '+x[0]*{hh}*{w}*{k}'\
                                    .format(hh=layer.hifm, w=layer.wifm, k=src_layer.nofm)
                            else:
                                ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}' \
                                    .format(m_h=m_h, s_h=s_h, w=layer.wifm, k=src_layer.nofm)
                            ext_inputs.remove(src)

                if isinstance(layer, LocalRegionLayer) and self.is_full_buffer[idx]:
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)

            if isinstance(layer, ConvLayer):
                if self.is_full_buffer[idx]:
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*{k}' \
                            .format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}' \
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)

                else:
                    cur_fidx = idx_dict[idx]
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*x[{idx}]' \
                            .format(hh=layer.hofm, w=layer.wofm, idx=cur_fidx)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*x[{idx}]' \
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, idx=cur_fidx)

                    ineq_cons.append('x[{idx}] - {k}'.format(idx=cur_fidx, k=loop_lower_bound.k))
                    ineq_cons.append('-x[{idx}] + {nofm}'.format(idx=cur_fidx, nofm=layer.nofm))

        for src in ext_inputs:
            if src is None:
                src_layer = self.network.input_layer()
            else:
                src_layer = self.network[src]
            loop_lower_bound = self.loop_lower_bound(src_layer)
            pidx = self.ext_inputs_idx[src]
            if pidx in idx_dict:
                cur_fidx = idx_dict[pidx]
                ineq_cons.append('x[{pidx}] - {k}'.format(pidx=cur_fidx, k=loop_lower_bound.k))
                ineq_cons.append('-x[{pidx}] + {nofm}'.format(pidx=cur_fidx, nofm=src_layer.nofm))

        total_buffer_size = self.resource.buffer(1).capacity
        #bug fix
        if self.sfil_fit == True :
            total_buffer_size = total_buffer_size -  self.fused_weight_size
        ss = '-(' + ss[1:] + ')+{}'.format(total_buffer_size)
        ineq_cons.append(ss)
        cons = ()
        for ineq in ineq_cons:
            # print(ineq)
            cons += ({'type': 'ineq', 'fun': lambda x, ineq=ineq: eval(ineq)}, )

        cons_res = copy.copy(cons)
        
        return cons_res

    def _alternate(self):

        level = self.resource.buffer_levels() - 2
        total_buffer_size = self.resource.buffer(level).capacity * self.resource.paras[level].count

        min_feature_footprint_t = total_buffer_size - 0.0000001
        add_one_line_footprint_t = float('inf')

        is_full_buffer_t = None
        # for start in [True, False]:
        # Full_Buffer True  -> NO BRR
        start = True
        is_full_buffer = [None for _ in range(len(self.dag_vertex_list))]
        min_feature_footprint = 0
        add_one_line_footprint = 0
        ext_inputs = set(self.ext_inputs_dict.keys())
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            sca = self.scale[idx]
            minsize = self.minSize[idx]
            if l in self.firsts:
                if self.womincost:
                    is_full_buffer[idx] = True
                else:
                    if start:
                        is_full_buffer[idx] = True
                    else:
                        is_full_buffer[idx] = False

                if not is_full_buffer[idx]:
                    for src in self.network.prevs(l):
                        if src in ext_inputs:
                            if src is None:
                                src_layer = self.network.input_layer()
                            else:
                                src_layer = self.network[src]

                            min_feature_footprint += \
                                (src_layer.nofm
                                    * min(((minsize.h - 1) * layer.hstd + layer.hfil), src_layer.hofm)
                                    * layer.wifm)
                            add_one_line_footprint += \
                                (src_layer.nofm
                                    * sca.s_h * layer.hstd
                                    * layer.wifm)
                            ext_inputs.remove(src)
                if isinstance(layer, LocalRegionLayer) and is_full_buffer[idx]:
                    min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                    add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm

            for src_idx in self.dag_prev_dict[idx]:
                assert is_full_buffer[src_idx] is not None
                if self.womincost:
                    is_full_buffer[idx] = True
                else:
                    if isinstance(layer, LocalRegionLayer):
                        if is_full_buffer[idx] is None:
                            is_full_buffer[idx] = is_full_buffer[src_idx]
                        else:
                            is_full_buffer[idx] \
                                = is_full_buffer[idx] or is_full_buffer[src_idx]
                    else:
                        if not is_full_buffer[src_idx]:
                            is_full_buffer[idx] = True
                        else:
                            is_full_buffer[idx] = False
            # print(layer,is_full_buffer[idx])
            if isinstance(layer, ConvLayer):
                if is_full_buffer[idx]:
                    min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                    add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm
                    # if start:
                        # print(layer.nofm * minsize.h * layer.wofm , layer.nofm * sca.s_h * layer.wofm)

                else:
                    loop_lower_bound = self.loop_lower_bound(layer)
                    k = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                    min_feature_footprint += k * minsize.h * layer.wofm
                    add_one_line_footprint += k * sca.s_h * layer.wofm

        if (total_buffer_size - min_feature_footprint) > 0 \
                and (add_one_line_footprint / (total_buffer_size - min_feature_footprint)) \
                < (add_one_line_footprint_t / (total_buffer_size - min_feature_footprint_t)):
            min_feature_footprint_t = min_feature_footprint
            is_full_buffer_t = is_full_buffer
            add_one_line_footprint_t = add_one_line_footprint
                
        return min_feature_footprint_t, is_full_buffer_t, add_one_line_footprint_t
    
    def _alternate_line_buffer_included(self):

        level = self.resource.buffer_levels() - 2
        total_buffer_size = self.resource.buffer(level).capacity * self.resource.paras[level].count

        is_full_buffer_t = None
        min_feature_footprint_t, add_one_line_footprint_t, line_buffer_footprint_t = 0,0,0
        min_feature_footprint_input, add_one_line_footprint_input, line_buffer_footprint_input = 0 , 0 , 0
        is_full_buffer = [True for _ in range(len(self.dag_vertex_list))]
        min_feature_footprint, add_one_line_footprint, line_buffer_footprint = 1,1,1

        ext_inputs = set(self.ext_inputs_dict.keys())
        # print(self.scale,self.minSize)
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            sca = self.scale[idx]
            minsize = self.minSize[idx]
                
            if l in self.firsts:
                for src in self.network.prevs(l):
                    if (sca.s_h - minsize.h == 0) and layer.hfil == 1 :
                        min_feature_footprint_input = 0
                        add_one_line_footprint_input = 0
                        line_buffer_footprint_input = 0
                        continue
                    else :
                        if src in ext_inputs:
                            if src is None:
                                src_layer = self.network.input_layer()
                            else:
                                src_layer = self.network[src]
                            if min_feature_footprint_input == None :
                                min_feature_footprint_input = src_layer.nofm * \
                                    min((minsize.h-1)*layer.hstd+layer.hfil,src_layer.hofm)*src_layer.wofm
                                add_one_line_footprint_input = src_layer.nofm * sca.s_h * layer.hstd * src_layer.wofm
                                line_buffer_footprint_input = src_layer.nofm * src_layer.wofm * \
                                    (min((minsize.h-1)*layer.hstd+layer.hfil,src_layer.hofm) - sca.s_h * layer.hstd)
                                # print(src, line_buffer_footprint_input)
                            else :
                                if min_feature_footprint_input < src_layer.nofm * min((minsize.h-1)*layer.hstd+layer.hfil,src_layer.hofm)*src_layer.wofm :
                                    min_feature_footprint_input = src_layer.nofm * \
                                        min((minsize.h-1)*layer.hstd+layer.hfil,src_layer.hofm)*src_layer.wofm
                                    add_one_line_footprint_input = src_layer.nofm * sca.s_h * layer.hstd * src_layer.wofm
                                    line_buffer_footprint_input = src_layer.nofm * src_layer.wofm * \
                                        (min((minsize.h-1)*layer.hstd+layer.hfil,src_layer.hofm) - sca.s_h * layer.hstd)
                                    # print(src, line_buffer_footprint_input)
                if isinstance(layer, LocalRegionLayer) :
                    min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                    add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm
                    line_buffer_footprint += layer.nofm * (minsize.h - sca.s_h) * layer.wofm
                    # print(l, (minsize.h - sca.s_h), layer.nofm * (minsize.h - sca.s_h) * layer.wofm)
                    
            if isinstance(layer, ConvLayer):
                min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm
                line_buffer_footprint += layer.nofm * (minsize.h - sca.s_h) * layer.wofm
                # print(l , (minsize.h - sca.s_h), layer.nofm * (minsize.h - sca.s_h) * layer.wofm)
                
        min_feature_footprint += min_feature_footprint_input
        add_one_line_footprint += add_one_line_footprint_input
        line_buffer_footprint += line_buffer_footprint_input
        if (total_buffer_size - min_feature_footprint) > 0 :
            is_full_buffer_t = is_full_buffer
            min_feature_footprint_t = int(min_feature_footprint)
            add_one_line_footprint_t = int(add_one_line_footprint)
            line_buffer_footprint_t = int(line_buffer_footprint)
            
        return min_feature_footprint_t, is_full_buffer_t, add_one_line_footprint_t, line_buffer_footprint_t
    
    