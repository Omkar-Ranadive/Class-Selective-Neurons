#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python compare_models_cka.py --exp_name rn50_2_e51 --m1 rn50_2 --m2 rn50_a20e5_1 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=1 python compare_models_cka.py --exp_name rn50_2_e52 --m1 rn50_2 --m2 rn50_a20e5_2 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=1 python compare_models_cka.py --exp_name rn50_3_e53 --m1 rn50_3 --m2 rn50_a20e5_3 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=1 python compare_models_cka.py --exp_name rn50_4_e54 --m1 rn50_4 --m2 rn50_a20e5_4 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=1 python compare_models_cka.py --exp_name rn50_5_e55 --m1 rn50_5 --m2 rn50_a20e5_5 --arc resnet50 ;
