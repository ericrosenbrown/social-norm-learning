#!/bin/bash

python multimodal_feedback.py --mode test --dataset data/action_only_0.pkl
python multimodal_feedback.py --mode test --dataset data/action_only_1.pkl
python multimodal_feedback.py --mode test --dataset data/action_only_2.pkl

python multimodal_feedback.py --mode test --dataset data/scalar_only_0.pkl
python multimodal_feedback.py --mode test --dataset data/scalar_only_1.pkl
python multimodal_feedback.py --mode test --dataset data/scalar_only_2.pkl

python multimodal_feedback.py --mode test --dataset data/human_choice_0.pkl
python multimodal_feedback.py --mode test --dataset data/human_choice_1.pkl
python multimodal_feedback.py --mode test --dataset data/human_choice_2.pkl
python multimodal_feedback.py --mode test --dataset data/human_choice_3.pkl
python multimodal_feedback.py --mode test --dataset data/human_choice_4.pkl
