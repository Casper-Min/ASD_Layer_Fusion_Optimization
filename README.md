# HYU 2022 Fall AI System Design Final Project - Layer Fusion Optimization #
This is the code of final project of HYU ECE9125 2022 Fall Semester

## Requirment ##
python==3.6
matplotlib==3.1.1
numpy==1.13.3
scipy==1.4.0

# Getiing Started Guide #
1. Open Layer_Fusion_Optimization.ipynb
2. set predefined network resnet18-B2/3/4/5
3. set batch 1/4 (or value you want)
4. set arch as 'fusion/arch/1024PE.json'
5. set buffer_size_KB as you want
6. set dataflow 'fusion/dataflow/1_weight_stationary.json' or 'fusion/dataflow/1_output_stationary.json'
7. set DRAM_BW 1/2/4(or value you want) to check performance difference due to Memory-Bound/Coumpute-Bound issue
8. set path as './Result'

### Reminder ###
1. Simple per iteration bandwidth/execution cycle based performance model is applied
2. Timeloop Wraper + Cost Model is excluded from source code due to build/dependency issue
3. If you are looking for Timeloop Wraper + Cost Model, send e-mail to ycivil93@hanyang.ac.kr
