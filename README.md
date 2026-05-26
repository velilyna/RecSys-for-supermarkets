# Recommender Systems for Supermarkets

This repository contains a research project on **next-basket recommendation** for supermarket retail. The goal is to predict which products a customer is likely to buy in the next shopping basket based on previous purchase history.

## Overview

Supermarket recommendation is strongly affected by repeat purchases: customers often buy the same everyday products again and again. Therefore, the project compares different approaches to understand which models are most suitable for grocery recommendation.

## Models

The project compares:

- Top Popular
- Top Personal
- TIFU-KNN
- BERT4Rec-NBR
- SAFERec-AE
- SAFERec-T
- Gen
- H-Gen

## Datasets

Experiments are conducted on two public grocery datasets:

- TaFeng Grocery Dataset
- Dunnhumby: The Complete Journey

## Metrics

Models are evaluated using:

- Recall@K
- NDCG@K
- UN@K

## Main Result

The results show that models using explicit repeat-purchase information perform better for supermarket recommendation than pure generative modelling.

The best top-10 recommendation quality is achieved by **SAFERec-AE**, while generative models produce more novel but less accurate recommendations.
