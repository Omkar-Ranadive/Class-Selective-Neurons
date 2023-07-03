#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_2_4 --m1 rn50_2 --m2 rn50_4 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_2_5 --m1 rn50_2 --m2 rn50_5 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_3_4 --m1 rn50_3 --m2 rn50_4 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_3_5 --m1 rn50_3 --m2 rn50_5 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_4_5 --m1 rn50_4 --m2 rn50_5 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_2_e01 --m1 rn50_2 --m2 rn50_a20e0_1 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_2_e02 --m1 rn50_2 --m2 rn50_a20e0_2 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_3_e03 --m1 rn50_3 --m2 rn50_a20e0_3 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_4_e04 --m1 rn50_4 --m2 rn50_a20e0_4 --arc resnet50 ;
CUDA_VISIBLE_DEVICES=0 python compare_models_cka.py --exp_name rn50_5_e05 --m1 rn50_5 --m2 rn50_a20e0_5 --arc resnet50 ;
